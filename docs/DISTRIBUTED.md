# Distributed training

Like Lc0, one server trains the network and many contributors' GPUs generate the games.
Contributors only need the release download; see [CONTRIBUTE.md](CONTRIBUTE.md).

```
 client ──┐   GET  /api/task            self-play task (current network) or gate match task
 client ──┼─► GET  /networks/<sha>.onnx  network download, checked by SHA-256
 client ──┘   POST /api/upload|match    results ──► rc0 validate ──► data/gen<N>/ ──► trainer ──► candidate ──► gate
```

## Client (`client/rc0_client.py`, `rc0-client.exe` on Windows)

The client runs this loop without anyone watching:

1. It asks for a task. The server's answer is either a **self-play** task (a network, the self-play settings
   and a game count) or a **match** task (a candidate network, the current network, playouts and a seed).
2. It downloads any network it doesn't have yet and checks its SHA-256.
3. **Self-play:** it runs `rc0 selfplay` with the server's settings, then checks its own output with
   `rc0 validate`, then uploads a `.tar.gz` of the output directory.
4. **Match:** it plays paired games, where each random opening is played twice with the teams swapped, and
   uploads the move lists.

The server chooses every setting except the number of parallel games and the device. That keeps all data in
a generation comparable.

## Server (`server/rc0_server.py`)

The server uses only the Python standard library. It needs the `rc0` engine and, for training, a Python with
PyTorch.

```bash
python3 server/rc0_server.py init  --data server_data --random 15x192      # or --net existing.pt (+ .onnx)
python3 server/rc0_server.py serve --data server_data --port 8000
```

Put it behind a reverse proxy with HTTPS (for example Caddy) to serve it publicly. `server_data/config.json`
holds every setting: self-play arguments, rows per generation, gate size and so on.

- **Validation.** Every upload is replayed by `rc0 validate` before it is accepted. Every move must be legal,
  and every training row must be the position it claims to be, with the right history, rules-derived labels,
  game result, plies-left and material targets. Policy targets may cover only legal moves and must sum to 1.
  Games that ended by the rules must have the result the rules give. Uploads that fail are rejected and
  counted against the contributor.
- **Data.** Accepted chunks are appended to `data/gen<N>/` in the same row format the single-machine trainer
  reads.
- **Training.** Every `rows_per_generation` new rows (default 100,000), the server retrains with
  `train/train_selfplay.py`. It uses the KataGo window and reuse rule and warm-starts from the current network.
- **Gating.** The new network is a *candidate* until clients have played `gate_games` games (default 200)
  against the current one at 300 playouts. It is promoted only if it scores at least 50% (KataGo, App. E).
  If it fails, training continues on more data.
- **Status.** `/` shows a status page (generation, games, contributors, gates). `/api/status` returns the
  same data as JSON.

## Formats

- **Network:** the server keeps TorchScript `.pt` files for training and serves `.onnx` files to clients.
  Both are named by the SHA-256 of the `.onnx` file.
- **Upload:** a `.tar.gz` containing `games.txt` and the `sp_*.bin` files
  (see [ARCHITECTURE.md](ARCHITECTURE.md)). `games.txt` has one line per game, tab-separated: the start
  FEN4, the number of random opening plies, the ply cap, how the game ended (natural, capped or adjudicated),
  the result, every move, and the ply of each recorded training row.

## Testing

`tests/test_distributed.sh` runs the whole loop on one machine, on the CPU, with a tiny network. It starts a
server, runs self-play tasks through the client, trains a candidate, plays the gate through the client, and
then promotes or rejects the candidate.

## Planned next

- Elo graph from the gate matches.
- Automatic client updates.
- Spot re-play of a random sample of uploads on the server, to catch clients that report fake search
  values.
