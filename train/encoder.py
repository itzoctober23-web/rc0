"""Builds network input planes on the GPU from the 136-byte self-play records (mirror of src/encoding.cpp).
The rotation tables come from the engine itself (`rc0 tables`), so Python and C++ cannot drift apart."""
# SPDX-License-Identifier: GPL-3.0-or-later
import json, os, subprocess, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import N_PLANES, BOARD, CELLS, POLICY_SIZE

ZERO_BIN = os.environ.get("RC0_BIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "rc0"))
REC = 136
SQUARES = 160


def load_tables():
    t = json.loads(subprocess.check_output([ZERO_BIN, "tables"]).decode())
    # The engine reports 41 planes without ZERO_HISTORY and 210 with it (encoding.h). The Python side builds
    # whichever the caller asks for, so accept either and keep asserting the part that must never differ.
    # 2026-10-01: this assert fired the moment build/zero became the 210-plane binary and would have crashed
    # the gen31 retrain, since zero_check.sh launches it unattended.
    assert t["N_PLANES"] in (N_PLANES, GpuEncoder.N_PLANES_HIST), \
        f'engine reports N_PLANES={t["N_PLANES"]}, expected {N_PLANES} or {GpuEncoder.N_PLANES_HIST}'
    assert t["POLICY_SIZE"] == POLICY_SIZE, f'engine POLICY_SIZE={t["POLICY_SIZE"]} != {POLICY_SIZE}'
    seat_inv = np.array(t["SEAT_INV"], dtype=np.int64)            # [4][160] seat sq -> mover-frame sq
    role = np.array(t["ROLE_UNDER"], dtype=np.int64)              # [stm][colour] -> role
    cell = np.array(t["RANK_OF"], dtype=np.int64) * BOARD + np.array(t["FILE_OF"], dtype=np.int64)  # [160]
    return seat_inv, role, cell, t["RULE50_PLIES"]


class GpuEncoder:
    """Same layout as Encoder, built on the GPU from raw records (139 KB per 1024-row batch
    crosses PCIe instead of 32 MB of planes; the CPU workers only slice memmaps)."""
    def __init__(self, dev):
        seat_inv, role, cell, self.r50 = load_tables()
        self.dev = dev
        self.canon_cell = torch.as_tensor(cell[seat_inv], device=dev)          # [4,160]
        self.role = torch.as_tensor(role, device=dev)                          # [4,4]
        mask = torch.zeros(CELLS, device=dev); mask[torch.as_tensor(cell, device=dev)] = 1.0
        self.mask = mask

    # input history: history planes. hist is uint8 [B, HIST_POS, 136] (newest first) and nhist is [B]; pass both to
    # emit N_PLANES_HIST planes instead of N_PLANES. Pass neither and nothing changes for the 41-plane line.
    HIST_POS = 7
    PL_HIST = 41
    PL_HISTVALID = PL_HIST + HIST_POS * 24          # 209
    N_PLANES_HIST = PL_HISTVALID + 1                # 210

    def __call__(self, rec, hist=None, nhist=None):  # rec: uint8 [B,136] -> float [B,planes,14,14]
        B = rec.shape[0]
        rec = rec.long()
        planes_out = self.N_PLANES_HIST if hist is not None else N_PLANES
        x = torch.zeros((B, planes_out, CELLS), device=self.dev)
        stm = rec[:, 0]
        pieces = rec[:, 8:136].view(B, 64, 2)
        valid = pieces[:, :, 0] != 255
        b_idx, p_idx = torch.nonzero(valid, as_tuple=True)
        pc = pieces[b_idx, p_idx, 0]; sq = pieces[b_idx, p_idx, 1]; st = stm[b_idx]
        plane = self.role[st, pc // 6] * 6 + pc % 6
        x[b_idx, plane, self.canon_cell[st, sq]] = 1.0
        cr = rec[:, 1]
        for c in range(4):
            for side in range(2):
                has = ((cr >> (c * 2 + side)) & 1).bool()
                bi = torch.nonzero(has, as_tuple=True)[0]
                if bi.numel():
                    x[bi, 24 + self.role[stm[bi], c] * 2 + side, :] = 1.0
        for c in range(4):
            e = rec[:, 4 + c]
            bi = torch.nonzero(e > 0, as_tuple=True)[0]
            if bi.numel():
                x[bi, 32 + self.role[stm[bi], c], self.canon_cell[stm[bi], e[bi] - 1]] = 1.0
        x[:, 36, :] = torch.clamp(rec[:, 2].float() / float(self.r50), max=1.0)[:, None]
        x[:, 39, :] = self.mask[None, :]
        x[:, 40, :] = (stm % 2 == 0).float()[:, None]
        if hist is not None:
            h = hist.long()                                  # [B, HIST_POS, 136]
            nh = nhist.long() if nhist is not None else torch.full((B,), self.HIST_POS, device=self.dev)
            for k in range(self.HIST_POS):
                live = torch.nonzero(nh > k, as_tuple=True)[0]
                if not live.numel():
                    continue
                hp = h[live, k, 8:136].view(-1, 64, 2)
                ok = hp[:, :, 0] != 255
                bi, pi = torch.nonzero(ok, as_tuple=True)
                if not bi.numel():
                    continue
                pc = hp[bi, pi, 0]; sq = hp[bi, pi, 1]
                st = stm[live][bi]                           # the CURRENT mover's frame, as in encoding.cpp
                pl = self.PL_HIST + k * 24 + self.role[st, pc // 6] * 6 + pc % 6
                x[live[bi], pl, self.canon_cell[st, sq]] = 1.0
            full = torch.nonzero(nh >= self.HIST_POS, as_tuple=True)[0]
            if full.numel():
                x[full, self.PL_HISTVALID, :] = 1.0
        return x.view(B, planes_out, BOARD, BOARD)
