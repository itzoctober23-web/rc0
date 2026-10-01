#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Net2Net growth for ZeroNet: widen 6x64 -> 10x128 by TRANSFERRING the current weights, so the student
starts as an exact functional copy of the teacher and then only has to improve on it.

His rule 7b says "warm-started from the CURRENT WEIGHTS by distilling on the replay window, never from
scratch". Attempts 1-3 initialised the student RANDOMLY and only used a distillation loss, which throws away
every generation of accumulated function and tries to re-learn it in one run - that is "from scratch with a
distillation loss", not a warm start. This does the warm start properly:

  widen  (Net2WiderNet, exact 2x): new channel j copies old channel j % F. Every weight that PRODUCES a
         channel is duplicated; every weight that CONSUMES one is halved, so sums are preserved. A tiny
         relative noise breaks the duplicate-gradient symmetry.
  deepen (residual identity): the extra blocks get b2 (the second BatchNorm) and the SE output layer zeroed,
         so the block computes relu(x + 0) = x exactly - the tower input is always post-ReLU, so relu is a
         no-op there.

  python net2net.py --teacher runs/rand/net_gen18.pt --blocks 10 --filters 128 --out grown.ckpt
"""
import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ZeroNet, N_PLANES, BOARD, net_for, in_planes_of
from train_selfplay import infer_arch


def widen_out(w, g):            # producer: duplicate rows
    return w[g].clone()


def widen_in(w, g, cnt):        # consumer: copy columns and divide by how many times the source was copied
    return (w[:, g] / cnt[g].view(1, -1, *([1] * (w.dim() - 2)))).clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--filters", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--in-planes", type=int, default=0, help="student input planes (0 = same as the teacher). "
                    "Growing them (41 -> 210 for input history) zero-initialises the NEW channels, so the student computes "
                    "exactly the teacher's function whatever the new planes contain.")
    ap.add_argument("--noise", type=float, default=1e-5, help="relative noise on duplicated producer weights (symmetry breaking)")
    ap.add_argument("--aux", action="store_true", help="attach the zero-initialised tactical auxiliary heads")
    ap.add_argument("--tol", type=float, default=1e-4, help="max allowed RELATIVE |student - teacher| per output on the verification batch")
    a = ap.parse_args()
    torch.manual_seed(0)

    tsd, tb, tf = infer_arch(a.teacher)
    if a.filters % tf != 0 or a.filters < tf or a.blocks < tb:
        raise SystemExit(f"teacher {tb}x{tf} cannot be grown to {a.blocks}x{a.filters} by this transfer")
    rep = a.filters // tf
    teacher = net_for(tsd, tb, tf); teacher.load_state_dict({k: v.contiguous() for k, v in tsd.items()}); teacher.eval()
    tin = in_planes_of(tsd)
    sin = a.in_planes or tin
    if sin < tin:
        raise SystemExit(f"cannot shrink input planes {tin} -> {sin}")
    student = ZeroNet(a.blocks, a.filters, in_planes=sin)

    g = torch.arange(a.filters) % tf                       # new trunk channel -> old trunk channel
    cnt = torch.bincount(g, minlength=tf).float()
    r = tf // 4                                            # SE hidden width (old)
    gh = torch.arange(a.filters // 4) % r                  # new SE hidden -> old SE hidden
    cnth = torch.bincount(gh, minlength=r).float()
    sd = student.state_dict()
    t = {k: v for k, v in teacher.state_dict().items()}

    def bn(prefix, gmap):
        for s in ("weight", "bias", "running_mean", "running_var"):
            sd[f"{prefix}.{s}"] = t[f"{prefix}.{s}"][gmap].clone()
        if f"{prefix}.num_batches_tracked" in t:
            sd[f"{prefix}.num_batches_tracked"] = t[f"{prefix}.num_batches_tracked"].clone()

    # stem: produces trunk channels, consumes the fixed 41 input planes
    stem_w = widen_out(t["stem.0.weight"], g)                     # [filters, tin, 3, 3]
    if sin > tin:                                                  # input history: new input planes contribute NOTHING
        padded = torch.zeros(stem_w.shape[0], sin, *stem_w.shape[2:])
        padded[:, :tin] = stem_w
        stem_w = padded
    sd["stem.0.weight"] = stem_w
    bn("stem.1", g)
    # trunk blocks: the first tb are transferred, the rest are residual identities
    for i in range(tb):
        sd[f"blocks.{i}.c1.weight"] = widen_out(widen_in(t[f"blocks.{i}.c1.weight"], g, cnt), g)
        bn(f"blocks.{i}.b1", g)
        sd[f"blocks.{i}.c2.weight"] = widen_out(widen_in(t[f"blocks.{i}.c2.weight"], g, cnt), g)
        bn(f"blocks.{i}.b2", g)
        sd[f"blocks.{i}.se.fc1.weight"] = widen_out(widen_in(t[f"blocks.{i}.se.fc1.weight"], g, cnt), gh)
        sd[f"blocks.{i}.se.fc1.bias"] = t[f"blocks.{i}.se.fc1.bias"][gh].clone()
        w2 = widen_in(t[f"blocks.{i}.se.fc2.weight"], gh, cnth)          # [2*tf, filters/4]
        b2 = t[f"blocks.{i}.se.fc2.bias"]
        sd[f"blocks.{i}.se.fc2.weight"] = torch.cat([w2[:tf][g], w2[tf:][g]], 0).clone()
        sd[f"blocks.{i}.se.fc2.bias"] = torch.cat([b2[:tf][g], b2[tf:][g]], 0).clone()
    for i in range(tb, a.blocks):   # identity: zero the second BN and the SE output -> relu(x + 0) = x
        torch.nn.init.kaiming_normal_(sd[f"blocks.{i}.c1.weight"]); torch.nn.init.kaiming_normal_(sd[f"blocks.{i}.c2.weight"])
        sd[f"blocks.{i}.b2.weight"].zero_(); sd[f"blocks.{i}.b2.bias"].zero_()
        sd[f"blocks.{i}.se.fc2.weight"].zero_(); sd[f"blocks.{i}.se.fc2.bias"].zero_()
    # heads
    sd["p_conv.0.weight"] = widen_out(widen_in(t["p_conv.0.weight"], g, cnt), g)
    bn("p_conv.1", g)
    sd["p_out.weight"] = widen_in(t["p_out.weight"], g, cnt); sd["p_out.bias"] = t["p_out.bias"].clone()
    sd["v_conv.0.weight"] = widen_in(t["v_conv.0.weight"], g, cnt)
    bn("v_conv.1", torch.arange(32))
    sd["own.weight"] = widen_in(t["own.weight"], g, cnt); sd["own.bias"] = t["own.bias"].clone()
    for k in ("v_fc.1.weight", "v_fc.1.bias", "v_fc.2.weight", "v_fc.2.bias", "v_fc.2.running_mean",
              "v_fc.2.running_var", "wdl.weight", "wdl.bias", "score.weight", "score.bias",
              "moves_left.weight", "moves_left.bias", "mat.weight", "mat.bias"):
        if k in t: sd[k] = t[k].clone()
    if "v_fc.2.num_batches_tracked" in t: sd["v_fc.2.num_batches_tracked"] = t["v_fc.2.num_batches_tracked"].clone()
    student.load_state_dict(sd)

    # VERIFY before any noise: the transfer must reproduce the teacher
    student.eval()
    xt = torch.rand(8, tin, BOARD, BOARD)
    xs = torch.rand(8, sin, BOARD, BOARD)        # the EXTRA planes are random on purpose:
    xs[:, :tin] = xt                             # a correct transfer must ignore them entirely
    with torch.no_grad():
        tp, tw, tm = teacher(xt); spv, sw, sm = student(xs)
    # RELATIVE tolerance: the transfer is exact in exact arithmetic (verified at 2e-12 in float64), so what is
    # left in float32 is accumulation from summing 2x as many channels. moves_left is ~1e3, so an absolute
    # threshold flags it spuriously - scale by each output's own magnitude.
    names = ("policy", "wdl", "moves_left")
    rel = [float((s_ - t_).abs().max() / t_.abs().max().clamp(min=1e-6)) for s_, t_ in ((spv, tp), (sw, tw), (sm, tm))]
    absd = [float((s_ - t_).abs().max()) for s_, t_ in ((spv, tp), (sw, tw), (sm, tm))]
    print("transfer check (exact copy): " + "  ".join(f"{n} abs {absd[i]:.3e} rel {rel[i]:.3e}" for i, n in enumerate(names)))
    if max(rel) > a.tol:
        raise SystemExit(f"TRANSFER IS NOT FUNCTION-PRESERVING (max relative {max(rel):.3e} > tol {a.tol:.3e}) - refusing to write")
    # symmetry breaking only after the check passes
    if a.noise > 0:
        with torch.no_grad():
            for k in ("stem.0.weight",) + tuple(f"blocks.{i}.c1.weight" for i in range(tb)) + tuple(f"blocks.{i}.c2.weight" for i in range(tb)):
                p = dict(student.named_parameters())[k]
                p.add_(torch.randn_like(p) * a.noise * p.abs().mean())
        with torch.no_grad():
            spv, sw, sm = student(xs)   # 08:27 fix: this site was missed when x was split into xt/xs, so every
                                        # --noise>0 run died with NameError AFTER the transfer check and never saved
        print(f"after {a.noise:g} symmetry-breaking noise: max|dpolicy| {float((spv - tp).abs().max()):.3e} "
              f"max|dwdl| {float((sw - tw).abs().max()):.3e}")
    # create the tactical auxiliary heads at growth time, ZERO-initialised, so the transfer stays
    # function-preserving (forward() does not touch them, so the traced graph and every verification above are
    # unaffected) and the next retrain can start training them without a separate restart - which is exactly
    # how he asked for them to arrive.
    sd_out = student.state_dict()
    if a.aux and not any(k.startswith("kcap.") for k in sd_out):
        f = a.filters
        sd_out["kcap.weight"] = torch.zeros(4, 256)
        sd_out["kcap.bias"] = torch.zeros(4)
        sd_out["hang.weight"] = torch.zeros(1, f, 1, 1)
        sd_out["hang.bias"] = torch.zeros(1)
        print(f"added order-3 aux heads, zero-initialised: kcap [4,256] and hang [1,{f},1,1]")
    torch.save({"net": sd_out, "step": 0, "blocks": a.blocks, "filters": a.filters, "in_planes": sin}, a.out)
    print(f"wrote {a.out}: {a.blocks}x{a.filters} in_planes {sin}, {sum(p.numel() for p in student.parameters()):,} params "
          f"(teacher {tb}x{tf} in_planes {tin}, {sum(p.numel() for p in teacher.parameters()):,})")


if __name__ == "__main__":
    main()
