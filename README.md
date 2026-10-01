# Rc0 — Revenant Chess Zero

Rc0 is an open-source engine for **4-player chess (Teams)**, built the way [Leela Chess Zero](https://lczero.org)
was built for regular chess: a neural network plus Monte-Carlo tree search, trained **from random play by
self-play only**. It has no opening book, no hand-written evaluation and no data from any other engine. Everything it
knows, it learned by playing against itself.

The goal is the same as Lc0's. Anyone with a GPU should be able to help train it, and the networks, code and
training data stay open.

> Status: early. One machine trains it today (an RTX-class consumer GPU). The code here is that pipeline.
> The next milestone is the **distributed client/server**, so other people's GPUs can add self-play games
> (see [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md)).

## The game

Four armies sit on a 14×14 board with the 3×3 corners removed: Red, Blue, Yellow and Green, moving in that
order. Opposite seats are partners: **Red + Yellow** against **Blue + Green**. A team wins by checkmating, or by
capturing, either enemy king. Partner pieces block each other and can't be captured. The full rules this engine
implements (chess.com Teams) are in [docs/RULES.md](docs/RULES.md).

Turn order changes a lot. A king that is not in check now can be mated before its owner moves again, because the
two enemies each get a move first. Hand-written heuristics do badly at this, which is one reason to learn it from
self-play.

## How it works

| Part | What it is |
|---|---|
| `rules/` | Board, move generation, check, mate, stalemate, repetition and 50-move rules. No search, no evaluation. Verified by perft and by a move-for-move comparison against an independent implementation on more than 15 million positions. |
| `src/` | The engine: input encoding, batched MCTS (PUCT for play, Gumbel for self-play), certainty propagation, NN cache, LibTorch and TensorRT backends, a UCI-style protocol for 4 players (`uci4`), and the self-play generator. |
| `train/` | PyTorch network (residual tower with squeeze-excitation; policy, WDL value, moves-left and auxiliary heads) and the self-play trainer. |
| `scripts/` | The single-machine loop: generate games → retrain on a sliding window → new generation. |
| `docs/` | Rules, architecture, the training recipe and the distributed design. |

The training recipe is copied from published research. We don't tune it ourselves. Settings come from
KataGo (Wu, 2019, *Accelerating Self-Play Learning in Go*) and Lc0. Where a setting had to change for 4 players,
[docs/TRAINING.md](docs/TRAINING.md) says so and explains why.

## Building

Requirements: Linux, CMake ≥ 3.20, a C++20 compiler, Python ≥ 3.10 with PyTorch (CUDA build) for the GPU
backend and for training. TensorRT is optional and about 2× faster for self-play.

```bash
git clone https://github.com/itzoctober23-web/rc0
cd rc0
cmake -S . -B build -DRC0_PYTHON=$(which python3)
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Without PyTorch the engine still builds. It then uses a uniform stand-in evaluator, which is useful only for
testing the rules and the search.

## Running

```bash
build/rc0 --net path/to/net.pt          # UCI4 engine (talk to it from a 4-player GUI or match runner)
build/rc0 bench path/to/net.pt          # search speed
build/rc0 selfplay --net net.pt --out games/ --threads 256 --playouts 600 --fast-playouts 100 --full-pct 25 --gumbel 16
python3 train/train_selfplay.py --init net.pt --data games/ --out train_out/
```

`scripts/selfplay.sh` and `scripts/train_cycle.sh` run the complete loop on one machine.

## Contributing

The most useful contributions right now:

1. **The distributed client and server** (Lc0-style, see [docs/DISTRIBUTED.md](docs/DISTRIBUTED.md)).
2. **Rules and protocol checks.** If you have another 4PC Teams implementation, compare perft and game results with ours.
3. **Testing nets.** Play them and report positions where they go wrong, especially turn-order mates.

Please open an issue before starting large changes.

## License

GPL-3.0-or-later, the same as Lc0 and Stockfish. See [LICENSE](LICENSE). Networks trained by the project are
released under the same terms.
