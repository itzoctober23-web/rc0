#!/bin/bash
# Build and run the fuzzers (clang + libFuzzer + AddressSanitizer + UndefinedBehaviorSanitizer).
#   tests/fuzz/run.sh [seconds per target] [seed self-play dir]
# SPDX-License-Identifier: GPL-3.0-or-later
set -eu
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SECS=${1:-600}
SEEDDIR=${2:-}
W=${FUZZ_WORK:-$ROOT/.tmp/fuzz}
mkdir -p "$W/corpus_fen" "$W/corpus_validate" "$W/crashes"
CXX=${CXX:-clang++}
FLAGS="-std=c++20 -O1 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined -I$ROOT/rules -I$ROOT/src"
RULES="$ROOT/rules/geometry.cpp $ROOT/rules/position.cpp $ROOT/rules/movegen.cpp"
$CXX $FLAGS "$ROOT/tests/fuzz/fuzz_fen.cpp" $RULES -o "$W/fuzz_fen"
$CXX $FLAGS "$ROOT/tests/fuzz/fuzz_validate.cpp" "$ROOT/src/validate.cpp" "$ROOT/src/encoding.cpp" $RULES -o "$W/fuzz_validate"
if [ -n "$SEEDDIR" ]; then
  python3 "$ROOT/tests/fuzz/pack.py" "$SEEDDIR" "$W/corpus_validate/real"
  python3 "$ROOT/tests/fuzz/pack.py" "$SEEDDIR" "$W/corpus_validate/real_games_only" --games-only
  head -20 "$SEEDDIR/games.txt" | cut -f1,6 | split -l 1 - "$W/corpus_fen/game_"
fi
export FUZZ_DIR="$W"
cd "$W/crashes"
echo "== fuzz_fen"
"$W/fuzz_fen" -max_total_time=$SECS -max_len=4096 -timeout=10 -rss_limit_mb=2048 "$W/corpus_fen" 2>&1 | tail -4
echo "== fuzz_validate"
"$W/fuzz_validate" -max_total_time=$SECS -max_len=200000 -timeout=30 -rss_limit_mb=2048 "$W/corpus_validate" 2>&1 | tail -4
echo "== crash files:"; ls "$W/crashes"
