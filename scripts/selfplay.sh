#!/bin/bash
# Self-play generator for the current generation (see docs/TRAINING.md for where each number comes from).
#   RUN=runs/main scripts/selfplay.sh        # RUN/NET.txt = current net, RUN/GEN = generation number
# SPDX-License-Identifier: GPL-3.0-or-later
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${RUN:-$ROOT/runs/main}
GEN=$(cat "$RUN/GEN"); NET=$(cat "$RUN/NET.txt"); OUT=$RUN/gen$GEN; mkdir -p "$OUT"
THREADS=${THREADS:-256}    # concurrent games; raise until the GPU batches are full
exec "$ROOT/build/rc0" selfplay --net "$NET" --games 100000000 --threads "$THREADS" \
  --playouts ${PLAYOUTS:-600} --fast-playouts ${FAST_PLAYOUTS:-100} --full-pct ${FULL_PCT:-25} \
  --gumbel 16 --noise-plies 20 --temp-plies 20 --max-plies 400 \
  --adj-thr 0.95 --adj-plies 8 --adj-min-ply 40 --verify-pct ${VERIFY_PCT:-25} \
  --batch-target ${BATCH:-512} --max-batch ${BATCH:-512} --out "$OUT"
