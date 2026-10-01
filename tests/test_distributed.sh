#!/bin/bash
# End-to-end test of the distributed loop on one machine, CPU only, with a tiny network:
#   server init -> client self-play tasks (validated uploads) -> server trains a candidate
#   -> client plays the gate -> candidate promoted or rejected.
# Needs: build/rc0 built with ONNX Runtime, and a Python with torch + onnx for the trainer.
#   PYTHON=/path/to/python tests/test_distributed.sh
# SPDX-License-Identifier: GPL-3.0-or-later
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON:-python3}
export CUDA_VISIBLE_DEVICES=""          # CPU only, so the test never competes for a GPU
W=$(mktemp -d "${TMPDIR:-/tmp}/rc0-e2e-XXXX")
PORT=${PORT:-18765}
trap 'kill $SRV 2>/dev/null || true; echo "workdir: $W"' EXIT

"$PY" "$ROOT/server/rc0_server.py" init --data "$W/srv" --random 2x16 --engine "$ROOT/build/rc0" --python "$PY" > "$W/init.log"
"$PY" - "$W/srv/config.json" <<'EOF'
import json, sys
p = sys.argv[1]; c = json.load(open(p))
c.update(games_per_task=6, rows_per_generation=150, gate_games=4, match_pairs_per_task=2, gate_playouts=16,
         match_opening_plies=4, reuse=2, train_batch=32)
c["selfplay_args"].update({"playouts": 24, "fast-playouts": 8, "full-pct": 50, "max-plies": 80, "adj-min-ply": 20})
json.dump(c, open(p, "w"), indent=1)
EOF
"$PY" "$ROOT/server/rc0_server.py" serve --data "$W/srv" --host 127.0.0.1 --port $PORT > "$W/server.out" 2>&1 &
SRV=$!
sleep 2
export APPDATA="$W/client" XDG_CONFIG_HOME="$W/client"
CL=("$PY" "$ROOT/client/rc0_client.py" --server "http://127.0.0.1:$PORT" --name e2e --engine "$ROOT/build/rc0" --ep cpu --threads 6 --workdir "$W/work" --once)
HOME="$W/client" "${CL[@]}"
st() { curl -s "http://127.0.0.1:$PORT/api/status"; }
for i in $(seq 1 60); do
  s=$(st)
  gen=$(echo "$s" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d['generation'], d['rows'], d['training'], d['candidate'] is not None, len(d['gates']))")
  echo "status: gen rows training candidate gates = $gen"
  set -- $gen
  [ "$5" -ge 1 ] && break
  HOME="$W/client" "${CL[@]}" || true
  [ "$3" = True ] && sleep 5
done
echo "--- server log"; cat "$W/srv/server.log"
s=$(st)
echo "$s" | "$PY" -c "
import json, sys
d = json.load(sys.stdin)
assert d['rows'] > 0, 'no rows accepted'
assert len(d['gates']) >= 1, 'no gate finished'
g = d['gates'][-1]
print('E2E OK: rows', d['rows'], 'games', d['games'], 'gate', g['score'], '/', g['games'], 'promoted' if g['passed'] else 'rejected', 'generation', d['generation'])
"
