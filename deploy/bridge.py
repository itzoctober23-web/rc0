#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Host side of a containerised hub: publish the trainer's current network to the hub.

Runs on the training machine (outside the container). When the trainer's current network changes, it exports
the portable .onnx copy and writes it with current.json into the inbox, which the hub mounts READ-ONLY. Nothing
flows the other way except validated training data, which the trainer reads as plain binary rows. The trainer
never loads a network file written by the container.

    python3 deploy/bridge.py --net-file runs/main/NET.txt --gen-file runs/main/GEN --inbox ~/rc0-server/inbox
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def publish(net_pt, gen, inbox, python):
    with tempfile.TemporaryDirectory(dir=inbox) as tmp:
        pt = os.path.join(tmp, "net.pt")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")   # export on the CPU: never touches the training GPU
        subprocess.run([python, os.path.join(ROOT, "train", "export_best.py"), net_pt, pt], check=True, env=env,
                       stdout=subprocess.DEVNULL)
        onnx = os.path.join(tmp, "net.onnx")
        sha = sha256_file(onnx)
        final = os.path.join(inbox, sha + ".onnx")
        shutil.move(onnx, final)
        os.chmod(final, 0o644)
    cur = os.path.join(inbox, "current.json")
    with open(cur + ".tmp", "w") as f:
        json.dump({"sha256": sha, "gen": gen, "onnx": sha + ".onnx", "source": os.path.basename(net_pt)}, f)
    os.chmod(cur + ".tmp", 0o644)
    os.replace(cur + ".tmp", cur)
    # keep the newest 3 networks in the inbox
    nets = sorted((p for p in os.listdir(inbox) if p.endswith(".onnx")), key=lambda p: os.path.getmtime(os.path.join(inbox, p)))
    for old in nets[:-3]:
        os.remove(os.path.join(inbox, old))
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net-file", required=True, help="file holding the path of the current .pt network")
    ap.add_argument("--gen-file", required=True, help="file holding the current generation number")
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--interval", type=int, default=60)
    a = ap.parse_args()
    os.makedirs(a.inbox, exist_ok=True)
    last = None
    while True:
        try:
            net = open(a.net_file).read().strip()
            gen = int(open(a.gen_file).read().strip())
            if (net, gen) != last and os.path.isfile(net):
                sha = publish(net, gen, a.inbox, a.python)
                print(time.strftime("%F %T"), f"published gen {gen} ({os.path.basename(net)}) as {sha[:12]}", flush=True)
                last = (net, gen)
        except Exception as e:   # keep running; the next interval retries
            print(time.strftime("%F %T"), f"bridge error: {e!r}", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
