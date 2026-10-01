#!/bin/bash
# One generation: retrain on the KataGo-style growing window, then promote the new net.
# Run it when RUN/gen<GEN> holds >= 100,000 new rows; restart scripts/selfplay.sh afterwards.
#   RUN=runs/main PYTHON=python3 scripts/train_cycle.sh
# SPDX-License-Identifier: GPL-3.0-or-later
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN=${RUN:-$ROOT/runs/main}; PY=${PYTHON:-python3}
export RC0_BIN=$ROOT/build/rc0
GEN=$(cat "$RUN/GEN"); NET=$(cat "$RUN/NET.txt"); NEXT=$((GEN + 1)); T=$RUN/train_gen$NEXT; mkdir -p "$T"

DATA=(); for g in $(seq $((GEN > 8 ? GEN - 8 : 0)) "$GEN"); do [ -s "$RUN/gen$g/sp_rec.bin" ] && DATA+=(--data "$RUN/gen$g"); done
NEW=$(( $(stat -c %s "$RUN/gen$GEN/sp_rec.bin") / 136 ))
TOTAL=0; for f in "$RUN"/gen*/sp_rec.bin; do TOTAL=$((TOTAL + $(stat -c %s "$f") / 136)); done

# KataGo: window N = c(1 + B((N_total/c)^A - 1)/A), c=250000 A=0.65 B=0.4; ~8 training samples per new row; batch 128
BATCH=${BATCH:-128}
read -r WINDOW STEPS <<<"$($PY -c "
import sys
tot,new,b=map(int,sys.argv[1:4]); c,A,B=250000,0.65,0.4
print(int(c*(1+B*((max(tot,1)/c)**A-1)/A)), max(200,int(8*new/b)))" "$TOTAL" "$NEW" "$BATCH")"

echo "gen $NEXT: init $NET, $NEW new rows, $TOTAL total, window $WINDOW, $STEPS steps x $BATCH"
"$PY" "$ROOT/train/train_selfplay.py" --init "$NET" "${DATA[@]}" --out "$T" \
  --steps "$STEPS" --batch "$BATCH" --window "$WINDOW" --eval-every 250 | tee "$T/train.log"
[ -f "$T/best.pt" ] || { echo "no net produced, see $T/train.log"; exit 1; }
cp -p "$T/best.pt" "$RUN/net_gen$NEXT.pt"
echo "$RUN/net_gen$NEXT.pt" > "$RUN/NET.txt"; echo "$NEXT" > "$RUN/GEN"
echo "gen $NEXT promoted: $RUN/net_gen$NEXT.pt"
