# SPDX-License-Identifier: GPL-3.0-or-later
"""Rc0 network (encoding v1: 41 input planes, 121 policy planes, 14x14).

forward(x) -> (policy_logits [B, 23716], wdl_logits [B, 3], moves_left [B, 1])
The C++ backend (src/backend_torch.cpp) depends on exactly this tuple; the
policy is indexed plane*196 + rank*14 + file, matching src/encoding.cpp.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PLANES = 41
BOARD = 14
CELLS = BOARD * BOARD
POLICY_PLANES = 121
POLICY_SIZE = POLICY_PLANES * CELLS
ENCODING_VERSION = 1


def in_planes_of(sd):
    """How many input planes a state dict's stem expects (41 pre-step-d, 210 with history)."""
    return int(sd["stem.0.weight"].shape[1])


def has_aux(sd):
    """does this state dict carry the tactical auxiliary heads?"""
    return any(k.startswith("kcap.") for k in sd)


def net_for(sd, blocks, filters, se=True):
    """Build a ZeroNet shaped for THIS state dict, so a 41-plane and a 210-plane checkpoint both load."""
    return ZeroNet(blocks, filters, se, in_planes=in_planes_of(sd), aux=has_aux(sd))


class SE(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, 2 * ch)
        self.ch = ch

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = self.fc2(s)
        w, b = s[:, : self.ch], s[:, self.ch :]
        return x * torch.sigmoid(w)[:, :, None, None] + b[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, ch, se=True):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
        self.se = SE(ch) if se else nn.Identity()

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        y = self.se(y)
        return F.relu(x + y)


class ZeroNet(nn.Module):
    def __init__(self, blocks=10, filters=128, se=True, in_planes=N_PLANES, aux=False):
        super().__init__()
        self.blocks_n, self.filters, self.in_planes, self.aux = blocks, filters, in_planes, aux
        self.stem = nn.Sequential(nn.Conv2d(in_planes, filters, 3, padding=1, bias=False), nn.BatchNorm2d(filters), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(filters, se) for _ in range(blocks)])
        # policy: conv tower -> 121 planes per cell (indexed by FROM square)
        self.p_conv = nn.Sequential(nn.Conv2d(filters, filters, 3, padding=1, bias=False), nn.BatchNorm2d(filters), nn.ReLU(inplace=True))
        self.p_out = nn.Conv2d(filters, POLICY_PLANES, 1)
        # value trunk (normalised, per the RASA post-mortem: never a bare ReLU on a Linear here)
        self.v_conv = nn.Sequential(nn.Conv2d(filters, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.v_fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * CELLS, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1, inplace=True))
        self.wdl = nn.Linear(256, 3)
        self.score = nn.Linear(256, 1)      # auxiliary: MCTS root Q
        self.moves_left = nn.Linear(256, 1)
        self.mat = nn.Linear(256, 1)        # auxiliary (from-random line): final material balance / 20, mover-team view
        # ownership auxiliary (KataGo): 25 classes per cell, 10 plies ahead
        self.own = nn.Conv2d(filters, 25, 1)
        # tactical auxiliary heads . TRAIN ONLY - deliberately absent from
        # forward(), so the traced TorchScript the C++ engine loads is unchanged and they cost the search
        # nothing. Created only when aux=True, so every existing checkpoint still load_state_dict()s
        # strictly; growth/distillation turns them on.
        if aux:
            self.kcap = nn.Linear(256, 4)          # per seat: a legal move of the mover captures that king
            self.hang = nn.Conv2d(filters, 1, 1)   # per cell: our piece attacked and not defended

    def forward(self, x):
        h = self.blocks(self.stem(x))
        p = self.p_out(self.p_conv(h))                        # [B,121,14,14]
        p = p.permute(0, 1, 2, 3).reshape(p.shape[0], -1)    # plane*196 + rank*14 + file
        v = self.v_fc(self.v_conv(h))
        return p, self.wdl(v), F.softplus(self.moves_left(v))

    def forward_train(self, x):
        h = self.blocks(self.stem(x))
        p = self.p_out(self.p_conv(h)).reshape(x.shape[0], -1)
        v = self.v_fc(self.v_conv(h))
        out = {"policy": p, "wdl": self.wdl(v), "score": self.score(v).squeeze(1),
               "moves_left": F.softplus(self.moves_left(v)).squeeze(1), "mat": self.mat(v).squeeze(1), "own": self.own(h)}
        if self.aux:
            out["kcap"] = self.kcap(v)                                  # [B,4] logits
            out["hang"] = self.hang(h).reshape(x.shape[0], -1)          # [B,196] logits
        return out


def export_torchscript(net: ZeroNet, path: str):
    net = net.eval().to(memory_format=torch.channels_last)          # NHWC weights: 1.26x measured 06:17
    ex = torch.zeros(1, getattr(net, "in_planes", N_PLANES), BOARD, BOARD).to(memory_format=torch.channels_last)
    ts = torch.jit.trace(net, ex)
    ts.save(path)
    return path


def export_onnx(net: ZeroNet, path: str):
    """Portable export for the ONNX Runtime backend (release downloads, Windows/DirectML, CPU).
    Dynamic batch; input "planes" [B,C,14,14] float32; outputs "policy", "wdl", "moves_left"."""
    net = net.eval().float().to("cpu")
    ex = torch.zeros(1, getattr(net, "in_planes", N_PLANES), BOARD, BOARD)
    torch.onnx.export(net, ex, path, input_names=["planes"], output_names=["policy", "wdl", "moves_left"],
                      dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "wdl": {0: "batch"}, "moves_left": {0: "batch"}},
                      opset_version=17, dynamo=False)
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="export a random-init net (Stage 0 backend smoke test)")
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--filters", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    net = ZeroNet(a.blocks, a.filters)
    n = sum(p.numel() for p in net.parameters())
    export_torchscript(net, a.out)
    if a.out.endswith(".pt"):
        export_onnx(net, a.out[:-3] + ".onnx")
    print(f"exported {a.blocks}x{a.filters} ({n:,} params) to {a.out}")
