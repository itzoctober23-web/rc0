#!/bin/bash
# Start a run from scratch: a randomly initialised network becomes generation 0.
#   RUN=runs/main BLOCKS=15 FILTERS=192 scripts/init_run.sh
# SPDX-License-Identifier: GPL-3.0-or-later
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${RUN:-$ROOT/runs/main}; PY=${PYTHON:-python3}
mkdir -p "$RUN"
[ -e "$RUN/GEN" ] && { echo "$RUN already holds a run (GEN exists); refusing to overwrite"; exit 1; }
"$PY" "$ROOT/train/model.py" --blocks ${BLOCKS:-15} --filters ${FILTERS:-192} --out "$RUN/net_gen0.pt"
echo "$RUN/net_gen0.pt" > "$RUN/NET.txt"; echo 0 > "$RUN/GEN"
echo "run initialised in $RUN; next: scripts/selfplay.sh"
