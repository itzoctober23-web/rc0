#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pack a self-play directory into one fuzz_validate seed: [u8 flags] then [u8 file][u32 len][bytes] records.
   python3 tests/fuzz/pack.py SELFPLAY_DIR OUT_SEED [--games-only]"""
import os, struct, sys

NAMES = ["games.txt", "sp_rec.bin", "sp_pol.bin", "sp_res.bin", "sp_q.bin", "sp_ply.bin", "sp_mat.bin", "sp_hist.bin", "sp_aux.bin"]
src, out = sys.argv[1], sys.argv[2]
games_only = "--games-only" in sys.argv
buf = bytes([1 if games_only else 0])
for i, n in enumerate(NAMES):
    if games_only and i > 0:
        break
    p = os.path.join(src, n)
    if os.path.isfile(p):
        b = open(p, "rb").read()
        if games_only:   # match games carry no rows: blank the row column
            b = b"\n".join(b"\t".join(l.split(b"\t")[:6] + [b""]) for l in b.splitlines() if l) + b"\n"
        buf += struct.pack("<BI", i, len(b)) + b
open(out, "wb").write(buf)
print(f"{out}: {len(buf)} bytes")
