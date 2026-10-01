# Architecture

Rc0 is built in four layers. Each one depends only on the layers below it.

```
train/   PyTorch: network definition, self-play trainer, export to TorchScript / TensorRT
src/     engine: encoding, MCTS, NN backends, UCI4 protocol, self-play generator
rules/   4PC Teams rules: board, moves, check, game end            (no search, no evaluation)
```

## Rules (`rules/`)

- `geometry.*`: lookup tables built once at startup: square indexing, rays, knight and king targets, pawn
  pushes and attacks per seat, the seat rotation maps (`SEAT_MAP`/`SEAT_INV`), and castling squares.
- `position.*`: the board, Zobrist keys, `do_move`/`undo_move`, check detection, repetition counting and FEN4.
- `movegen.*`: pseudo-legal and legal moves, perft, and `game_result()` (king capture, mate, stalemate,
  50-move rule, repetition).

## Input encoding (`src/encoding.*`)

The board is **rotated so the side to move always sits in Red's seat**. The other armies are ordered by their
role relative to the mover: mover, next enemy, partner, previous enemy. So the network sees every seat the
same way, and rotating a position by one seat produces an identical tensor. Reflection is *not* a symmetry,
because of the king/queen placement and castling, so it is never used for augmentation.

| Planes | Content |
|---|---|
| 0–23 | pieces: role × 6 + piece type |
| 24–31 | castling rights: role × 2 + side |
| 32–35 | en-passant target square, per role |
| 36 | 50-move counter / 200 |
| 37–38 | position seen once / twice before |
| 39 | playable-square mask (the 196-cell grid includes the dead corners) |
| 40 | 1 if the mover's team is Red + Yellow |

There are 41 planes on a 14×14 grid. An optional 210-plane version adds the piece planes of the previous seven
positions, like Lc0 (`-DZERO_HISTORY`).

**Policy:** 121 planes × 196 from-squares, which is 23,716 outputs. Planes 0–103 are slides
(8 directions × distance 1–13), 104–111 are knight moves, and 112–120 are under-promotions
(N/B/R × forward/capture-left/capture-right). Queen promotions use the slide planes.

## Network (`train/model.py`)

A residual tower with squeeze-and-excitation, the Lc0/KataGo design. Self-play currently uses **15 blocks × 192
filters**.

- Policy head: a 3×3 conv, then a 1×1 conv to 121 planes.
- Value trunk: a 1×1 conv to 32 channels, then FC 256, which feeds:
  - **WDL** (win/draw/loss, from the mover's team's point of view)
  - **moves left**
  - training-only auxiliaries: final material balance, search value, ownership, and two *tactical*
    heads computed exactly from the rules. One says which enemy king can be captured next move; the other marks
    which of our pieces are attacked and undefended. The search never reads the auxiliaries. They only shape the
    shared trunk during training, the same idea as KataGo's auxiliary targets.

## Search (`src/mcts.*`)

- PUCT MCTS with batched leaf evaluation (virtual loss, collision handling) and an NN cache keyed by
  position hash.
- **Value sign:** each node's value is from the point of view of the *team* to move at that node. Since the
  team alternates every ply, values flip sign on every edge, which keeps the 4-player tree a two-team zero-sum tree.
- **Certainty propagation** (as in Lc0): terminal positions are proven results, and proofs are passed up the
  tree. If a child is a proven loss for the team to move there, the parent is a proven win; if all children are
  proven wins for the opponent, the parent is a proven loss. A proven mate is therefore searched once and then
  trusted, including turn-order mates where an enemy moves in between.
- **Self-play** uses Gumbel AlphaZero root selection (Top-16, sequential halving) and records the
  completed-Q policy as the training target. This works well even at small playout counts.

## Backends (`src/backend_*.cpp`)

- **LibTorch:** loads the TorchScript export. FP16 is used on the GPU.
- **TensorRT:** loads a serialized `.plan` built by `tools/trt_build_plan.py`. It is about 2× faster than
  LibTorch for the 15×192 net. The `.plan` file sits next to the net and is checked against it, and the engine
  falls back to LibTorch if it is missing or stale.

## Protocol (`src/uci4.*`)

A UCI-style protocol for 4 players: `position fen4 … moves …`, `go nodes|movetime|infinite`, `stop`, and
`bestmove`. It also includes referee commands, so GUIs and match runners can use the engine to check legality
and game results.

## Self-play data (`src/selfplay.*`)

Self-play writes append-only binary files to its output directory:

| File | Per row |
|---|---|
| `sp_rec.bin` | 136-byte position record (side to move, castling, 50-move counter, en passant, up to 64 pieces) |
| `sp_pol.bin` | up to 96 × (policy index int16, probability fp16), the search policy |
| `sp_res.bin` | game result (int8, Red + Yellow view) |
| `sp_q.bin` | root search value (float32, mover-team view) |
| `sp_ply.bin` | plies until the game ended (int16) |
| `sp_mat.bin`, `sp_aux.bin`, `sp_hist.bin` | auxiliary targets and input history |

Only the moves that received the full playout budget are recorded as training rows (playout-cap
randomization, see [TRAINING.md](TRAINING.md)).
