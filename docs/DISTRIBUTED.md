# Distributed training (design)

The goal is to let anyone with a GPU contribute self-play games, the same way Lc0 does. One server trains the
network; many clients generate the games. This page is the design. The implementation is the next milestone.

## How Lc0 does it

- A **server** hosts the current network, hands out work, collects games and runs the trainer.
- A **client** (`lc0-client`) downloads the newest network, runs the engine in self-play mode, and uploads the
  training data and a game record. It runs unattended and updates itself to new networks.
- **Test matches** between networks run on clients too, and give the published Elo graph.

Rc0 copies this structure.

## Components

```
            ┌──────────────────────────── server ────────────────────────────┐
 client ──► │  API  ──►  validator  ──►  chunk store  ──►  trainer (GPU)       │
 client ──► │   │                                             │                │
 client ──► │   └──── networks (sha256-addressed) ◄───────────┘                │
            └────────────────────────────────────────────────────────────────┘
```

### Client (`client/rc0_client.py`, planned)

A loop with no interaction needed:

1. `GET /api/task`: the server returns either a **self-play** task (network sha256, playouts, number of games,
   engine version) or a **match** task (two networks and opening seeds).
2. Download the network if it is not already cached, and check its sha256.
3. Run `rc0 selfplay --net … --games N --out tmp/` with the server's settings. The user only chooses the GPU
   and how many CPU threads to use.
4. Upload the chunk (the `sp_*.bin` files plus the move list of every game, compressed) to `POST /api/games`.
5. Repeat. A new network is picked up at the next task.

Clients never choose their own training settings, so all data in a generation is comparable.

### Server (`server/`, planned)

- **API:** tasks, network downloads, uploads and a public stats page. Users can create an account with a token.
- **Validator:** every uploaded game is **replayed through the rules library**. Every recorded move must be
  legal, the recorded result must equal `game_result()`, and every recorded position must match the replay.
  A chunk that fails is rejected. A contributor whose uploads keep failing is ignored until reviewed. This is
  the main defense against broken or malicious clients.
- **Sanity checks:** policy targets must be normalized and cover only legal moves, and the result distribution
  for each network is compared against the server's own control games.
- **Trainer:** the same `train/train_selfplay.py` loop and recipe as on one machine ([TRAINING.md](TRAINING.md)),
  run whenever 100,000 new validated rows arrive.
- **Gating:** a new network must win ≥ 100 of 200 games against the current one (KataGo App. E) before clients
  receive it. The matches run as client match tasks.
- **Publishing:** every network, the training data and the match results are public for download, as with Lc0.

## Formats

- **Network:** TorchScript `.pt`. Each client builds its own TensorRT `.plan` locally, because plans are
  specific to a GPU and TensorRT version. Networks are addressed by sha256.
- **Training chunk:** the self-play files described in [ARCHITECTURE.md](ARCHITECTURE.md) plus `games.txt`
  (one line per game: start FEN4, moves, result). A version byte comes first so the format can change later.

## Order of work

1. Game records in self-play output (`games.txt`), plus a `validate` command in the engine that replays a chunk.
2. Server: tasks, network hosting, upload and validation, with the trainer reading validated chunks.
3. Client: download, self-play, upload, and automatic updates.
4. Match tasks for gating and the public Elo graph.
5. A public dashboard for contributors and networks.

Help is welcome with any of these steps. Please open an issue first.
