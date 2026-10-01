# Help train Rc0 with your GPU

Rc0 learns 4-player chess (Teams) only by playing against itself. Every game your computer plays is
checked by the server and becomes training data for the next network.

## Windows

1. Download `rc0-windows-x64.zip` from the [latest release](https://github.com/itzoctober23-web/rc0/releases/latest)
   and unzip it anywhere.
2. Open a terminal in that folder (Shift + right-click → *Open in Terminal*) and run:
   ```
   rc0-client.exe --server SERVER_ADDRESS --name YOUR_NAME --invite YOUR_INVITE_CODE
   ```
3. Leave it running. Stop it any time with Ctrl+C. Next time, just run `rc0-client.exe`, because your settings
   are saved.

The Windows build runs on any DirectX 12 GPU (NVIDIA, AMD or Intel) through DirectML. You don't need to
install drivers beyond your normal graphics driver.

## Linux

1. Download `rc0-linux-x64.tar.gz` from the [latest release](https://github.com/itzoctober23-web/rc0/releases/latest)
   and unpack it.
2. Run `python3 rc0_client.py --server SERVER_ADDRESS --name YOUR_NAME --invite YOUR_INVITE_CODE` in that folder.

The Linux build uses CUDA if CUDA 12 and cuDNN 9 are installed, and the CPU otherwise.

While the project is new, the server is invite-only: ask the maintainer for an invite code. You only need it on
the first run.

## Options

| Option | Meaning |
|---|---|
| `--threads N` | Games played at once (default 64). More games keep a fast GPU busier. |
| `--ep cuda\|dml\|cpu\|auto` | Which device ONNX Runtime uses (default: auto, which tries CUDA, then DirectML, then the CPU). |
| `--once` | Do one task, then exit (useful for testing). |

## What it does

The client asks the server for a task, downloads the current network (checked by its SHA-256 hash), plays
the games with `rc0`, checks its own output with `rc0 validate`, and uploads it. Some tasks are **test
matches** between a new candidate network and the current one, which decide whether the candidate becomes the
next generation.

Nothing on your computer is read or sent except the games the engine plays. The source code is in this
repository, licensed under GPL-3.0.
