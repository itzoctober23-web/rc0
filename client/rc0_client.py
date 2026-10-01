#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rc0 training client: lend your GPU to the Rc0 project.

It asks the server for work, downloads the current network, plays self-play games (or test matches) with the
rc0 engine, and uploads the results. It runs until you stop it (Ctrl+C). Only the Python standard library is
used, and the Windows download ships it as rc0-client.exe, so no Python install is needed there.

    rc0-client --server https://SERVER --name YOUR_NAME          (first run: registers and saves a token)
    rc0-client                                                    (later runs reuse the saved settings)

Options: --threads (concurrent games, default 64), --engine (path to rc0 / rc0.exe), --ep (cuda | dml | cpu |
auto: which ONNX Runtime device to use), --once (do one task and exit), --workdir (cache for networks).
"""
import argparse, hashlib, io, json, os, platform, random, shutil, subprocess, sys, tarfile, tempfile, time
import urllib.error, urllib.request

CLIENT_VERSION = 1


def here():
    # next to the executable when frozen (rc0-client.exe), else next to this script
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def default_engine():
    exe = "rc0.exe" if os.name == "nt" else "rc0"
    for d in (here(), os.path.join(here(), "..", "build"), os.path.join(here(), "..", "build", "Release")):
        p = os.path.join(d, exe)
        if os.path.isfile(p):
            return os.path.abspath(p)
    return shutil.which(exe) or exe


def config_path():
    base = os.environ.get("APPDATA") if os.name == "nt" else os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base or here(), "rc0", "client.json")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class Server:
    def __init__(self, url, token=None):
        self.url = url.rstrip("/")
        self.token = token

    def _req(self, method, path, body=None, ctype="application/json", timeout=120):
        headers = {"User-Agent": f"rc0-client/{CLIENT_VERSION}"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if body is not None:
            headers["Content-Type"] = ctype
        req = urllib.request.Request(self.url + path, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def json(self, method, path, obj=None, timeout=120):
        body = json.dumps(obj).encode() if obj is not None else None
        return json.loads(self._req(method, path, body, timeout=timeout))

    def download(self, path, dest, sha256):
        tmp = dest + ".part"
        h = hashlib.sha256()
        req = urllib.request.Request(self.url + path, headers={"User-Agent": f"rc0-client/{CLIENT_VERSION}"})
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                h.update(b)
                f.write(b)
        if h.hexdigest() != sha256:
            os.remove(tmp)
            raise RuntimeError(f"network download corrupted (sha256 {h.hexdigest()[:12]} != {sha256[:12]})")
        os.replace(tmp, dest)


def get_network(srv, net, workdir):
    """net = {"sha256": <hash of the .onnx file>, "onnx": "/networks/<sha>.onnx"}; cached by hash."""
    os.makedirs(os.path.join(workdir, "networks"), exist_ok=True)
    path = os.path.join(workdir, "networks", net["sha256"] + ".onnx")
    if not os.path.isfile(path):
        log(f"downloading network {net['sha256'][:12]} ...")
        srv.download(net["onnx"], path, net["sha256"])
    return path


def engine_env(args):
    env = dict(os.environ)
    env["ZERO_BACKEND"] = "onnx"
    env["RC0_ONNX_EP"] = args.ep
    return env


# ---------------------------------------------------------------- self-play
def do_selfplay(srv, task, args):
    net = get_network(srv, task["network"], args.workdir)
    out = tempfile.mkdtemp(prefix="rc0-sp-", dir=args.workdir)
    try:
        cmd = [args.engine, "selfplay", "--net", net, "--out", out, "--games", str(task["games"]),
               "--threads", str(min(args.threads, task["games"])), "--seed", str(random.getrandbits(48))]
        for k, v in task["args"].items():
            cmd += ["--" + k, str(v)]
        log(f"self-play: {task['games']} games with network {task['network']['sha256'][:12]} (gen {task.get('gen')})")
        r = subprocess.run(cmd, env=engine_env(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = r.stdout.strip().splitlines()
        if r.returncode != 0:
            raise RuntimeError("engine failed:\n" + "\n".join(lines[-5:]))
        for l in lines:
            if l.startswith("info string network") or l.startswith("RESULT"):
                log(l[:220])
        # check locally first: a client that produces bad data finds out here, not after uploading it
        v = subprocess.run([args.engine, "validate", out], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if v.returncode != 0:
            raise RuntimeError("self-play output failed validation: " + v.stdout.strip().splitlines()[0])
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name in sorted(os.listdir(out)):
                tar.add(os.path.join(out, name), arcname=name)
        ans = json.loads(srv._req("POST", f"/api/upload?task={task['id']}", buf.getvalue(), "application/gzip", timeout=600))
        log(f"uploaded {len(buf.getvalue()) // 1024} KB: {ans}")
    finally:
        shutil.rmtree(out, ignore_errors=True)


# ---------------------------------------------------------------- matches (gating new networks)
class Engine:
    def __init__(self, exe, net, env, playouts):
        self.p = subprocess.Popen([exe, "--net", net], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        self.send("uci4")
        self.expect("uci4ok")
        self.send(f"setoption name Playouts value {playouts}")
        self.send("isready")
        self.expect("readyok")

    def send(self, line):
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def expect(self, prefix):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            if line.startswith(prefix):
                return line.strip()

    def position(self, fen, moves):
        self.send(f"position fen4 {fen}" + (" moves " + " ".join(moves) if moves else ""))

    def legal(self, fen, moves):
        self.position(fen, moves)
        self.send("legalmoves")
        return self.expect("legalmoves").split()[1:]

    def status(self, fen, moves):
        self.position(fen, moves)
        self.send("status")
        return self.expect("status").split()[1]

    def go(self, fen, moves):
        self.position(fen, moves)
        self.send("go")
        return self.expect("bestmove").split()[1]

    def close(self):
        try:
            self.send("quit")
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def play_game(a, b, fen, opening, a_on_ry, max_plies):
    """The team to move alternates every ply. Returns (moves, result in Red+Yellow's view: 1/0/-1, end)."""
    moves = list(opening)
    first_ry = fen[0] in "RY"
    while len(moves) < max_plies:
        st = a.status(fen, moves)
        if st != "ongoing":
            return moves, {"ry_wins": 1, "bg_wins": -1, "draw": 0}[st], "natural"
        ry_to_move = (len(moves) % 2 == 0) == first_ry
        eng = a if ry_to_move == a_on_ry else b
        m = eng.go(fen, moves)
        if m == "0000":
            raise RuntimeError("engine returned no move")
        moves.append(m)
    st = a.status(fen, moves)
    if st != "ongoing":
        return moves, {"ry_wins": 1, "bg_wins": -1, "draw": 0}[st], "natural"
    return moves, 0, "capped"


def do_match(srv, task, args):
    na = get_network(srv, task["a"], args.workdir)
    nb = get_network(srv, task["b"], args.workdir)
    env = engine_env(args)
    a = Engine(args.engine, na, env, task["playouts"])
    b = Engine(args.engine, nb, env, task["playouts"])
    rng = random.Random(task["seed"])
    games, score = [], 0.0
    fen = task["fen"]
    try:
        log(f"match: {task['pairs']} pairs, {task['a']['sha256'][:12]} vs {task['b']['sha256'][:12]} at {task['playouts']} playouts")
        for _ in range(task["pairs"]):
            opening = []
            for _ in range(task["opening_plies"]):   # one random opening, played twice with the teams swapped
                legal = a.legal(fen, opening)
                if not legal or a.status(fen, opening) != "ongoing":
                    break
                opening.append(rng.choice(legal))
            for a_on_ry in (True, False):
                moves, res, end = play_game(a, b, fen, opening, a_on_ry, task["max_plies"])
                score += 0.5 if res == 0 else (1.0 if (res == 1) == a_on_ry else 0.0)
                games.append({"moves": " ".join(moves), "a_on_ry": a_on_ry, "result": res, "end": end})
        ans = srv.json("POST", f"/api/match?task={task['id']}", {"games": games})
        log(f"match uploaded: A scored {score}/{len(games)}: {ans}")
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description="Rc0 training client")
    ap.add_argument("--server")
    ap.add_argument("--name", help="contributor name shown on the server's stats page")
    ap.add_argument("--invite", help="invite code (needed once, if the server is invite-only)")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--threads", type=int, default=None, help="concurrent self-play games (more = fuller GPU batches)")
    ap.add_argument("--ep", default=None, choices=["auto", "cuda", "dml", "cpu"], help="ONNX Runtime device")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--once", action="store_true", help="do one task, then exit")
    args = ap.parse_args()

    cfg_file = config_path()
    cfg = json.load(open(cfg_file)) if os.path.isfile(cfg_file) else {}
    for k in ("server", "name", "engine", "threads", "ep", "workdir"):
        if getattr(args, k) is None:
            setattr(args, k, cfg.get(k))
    args.engine = args.engine or default_engine()
    args.threads = args.threads or 64
    args.ep = args.ep or "auto"
    args.workdir = args.workdir or os.path.join(os.path.dirname(cfg_file), "work")
    os.makedirs(args.workdir, exist_ok=True)
    if not args.server:
        sys.exit("first run: pass --server URL (and --name). It is saved for later runs.")
    if not os.path.isfile(args.engine) and not shutil.which(args.engine):
        sys.exit(f"engine not found: {args.engine} (pass --engine PATH)")

    srv = Server(args.server, cfg.get("token") if cfg.get("server") == args.server else None)
    if not srv.token:
        try:
            ans = srv.json("POST", "/api/register", {"name": args.name or platform.node() or "anonymous",
                                                      "invite": args.invite or ""})
        except urllib.error.HTTPError as e:
            sys.exit(f"registration refused ({e.code}): {e.read().decode(errors='replace')[:200]}")
        srv.token = ans["token"]
        log(f"registered as {ans['name']}")
    cfg.update({"server": args.server, "name": args.name, "token": srv.token, "engine": args.engine,
                "threads": args.threads, "ep": args.ep, "workdir": args.workdir})
    os.makedirs(os.path.dirname(cfg_file), exist_ok=True)
    with open(cfg_file, "w") as f:
        json.dump(cfg, f, indent=1)

    backoff = 10
    while True:
        try:
            task = srv.json("GET", f"/api/task?version={CLIENT_VERSION}")
            if task["type"] == "selfplay":
                do_selfplay(srv, task, args)
            elif task["type"] == "match":
                do_match(srv, task, args)
            elif task["type"] == "upgrade":
                sys.exit("this client is too old for the server: " + task.get("message", "download the new release"))
            else:
                time.sleep(task.get("seconds", 30))
            backoff = 10
        except KeyboardInterrupt:
            log("stopped")
            return
        except (urllib.error.URLError, OSError, RuntimeError, ValueError, KeyError) as e:
            log(f"error: {e}; retrying in {backoff} s")
            if args.once:
                sys.exit(1)
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
        if args.once:
            return


if __name__ == "__main__":
    main()
