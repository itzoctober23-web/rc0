# Training recipe

Rc0 starts from a randomly initialized network and learns only from its own games. The project rule is
**copy what research has already shown works, instead of inventing settings**. Every number below is taken
from a published source, or marked as a 4-player adaptation with the reason given.

Sources:
- **[KG]** D. Wu, *Accelerating Self-Play Learning in Go*, arXiv:1902.10565 (KataGo).
- **[KG-repo]** github.com/lightvector/KataGo, `python/selfplay/synchronous_loop.sh` and `shuffle.sh`
  (the single-machine loop).
- **[Lc0]** Leela Chess Zero design and training documentation.
- **[Gumbel]** Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022.

## The loop

1. **Self-play** (`build/rc0 selfplay`) with the current network until enough new training rows exist (100,000).
2. **Retrain** (`train/train_selfplay.py`): warm-start from the current network and train on a sliding window of
   recent rows.
3. The new network becomes the next **generation**, and self-play continues with it.

`scripts/train_cycle.sh` runs steps 2–3. `scripts/selfplay.sh` runs step 1.

## Self-play settings

| Setting | Value | Source |
|---|---|---|
| Root selection | Gumbel Top-16 with sequential halving, completed-Q policy target | [Gumbel] |
| Playout-cap randomization | 25% of moves get a full search (600 playouts) and are recorded; the rest get a fast search (100) and are not | [KG] §3.1 |
| Exploration | Dirichlet root noise (α = 0.3 × 64 / legal moves, ε = 0.25) and temperature for the first 20 plies | [KG] §2 / [Lc0]: α scaled inversely with the number of legal moves |
| Adjudication | resign when the search value stays beyond ±0.95 for 8 plies after ply 40; 25% of those games are played out to measure how often resigning is wrong | [Lc0] resign with false-positive checks |
| Game length cap | 400 plies, scored as a draw | 4PC adaptation |
| Proven results | certainty propagation; proven positions use the proven value as the target | [Lc0] |
| Openings | 20% of games start from a shuffled back rank (the same for all four seats) for variety | [Lc0] (Chess960 openings) |

## Optimizer and data

| Setting | Value | Source |
|---|---|---|
| Optimizer | SGD, momentum 0.9, L2 3e-5 | [KG] §2 |
| Learning rate | 6e-5 per sample (2e-5 for the first 5M samples of a fresh net), held flat; per-batch lr = per-sample × batch | [KG] §2 |
| Batch | 128 | [KG-repo] single-machine loop |
| Window | N_window = c·(1 + β((N_total/c)^α − 1)/α), c = 250,000, α = 0.65, β = 0.4 | [KG] App. C, [KG-repo] `shuffle.sh` |
| Sample reuse | each new row is trained on about 8 times | [KG-repo] `-max-train-bucket-per-new-data 8` |
| Value target | 0.5 × game result + 0.5 × root search value, as a soft WDL target | [KG]/[Lc0] blend of outcome and search value |
| Policy target | the Gumbel completed-Q policy (KL loss) | [Gumbel] |
| Auxiliary heads | weight 0.15: moves left, material, and rules-exact tactical labels (capturable kings, undefended pieces) | [KG] App. B (auxiliary targets), [Lc0] moves left |
| Gating | a candidate must win ≥ 100 of 200 games against the current net to replace it | [KG] App. E. *Being implemented* |

## What is 4PC-specific

- Four movers, two teams. Values are always from the mover's team's point of view, and the sign flips every ply.
- The encoding rotates every position into the mover's seat (see [ARCHITECTURE.md](ARCHITECTURE.md)).
- Turn-order tactics: a king that is safe now can be mated before its owner moves again. Certainty propagation
  and the tactical auxiliary labels are there so the network learns these quickly.

## Measuring progress

- **Holdout loss:** the newest 5% of the window is held out every generation.
- **Ladder:** fixed bots of increasing strength. The current network plays each rung in paired-seat matches.
- **Tactics suite:** positions taken from the rules alone (forced king captures and mates), scored by whether
  the engine finds the move.
