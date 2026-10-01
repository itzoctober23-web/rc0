#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Rc0 training server (the Lc0-style hub). Standard library only, plus the rc0 engine and the trainer.

    python3 server/rc0_server.py init  --data server_data --net runs/main/net_gen0.pt   (or --random 15x192)
    python3 server/rc0_server.py serve --data server_data --port 8000

What it does:
  * hands out tasks: self-play with the current network, or gating matches between a new candidate and it
  * hosts every network (sha256-addressed .onnx for clients, .pt for training)
  * accepts uploads only after `rc0 validate` has replayed every game and checked every training row
  * trains: when enough new validated rows exist, it retrains (train/train_selfplay.py, KataGo recipe) and
    makes the result a candidate; clients play the gate (KataGo: >= 100 of 200 by default) and the candidate
    becomes the next generation only if it passes
  * publishes a status page at / and JSON at /api/status
Put it behind a reverse proxy with HTTPS (Caddy, nginx) for a public deployment.
"""
import argparse, hashlib, html, io, json, os, re, secrets, shutil, subprocess, sys, tarfile, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_CLIENT_VERSION = 1
SP_FILES = ["sp_rec.bin", "sp_pol.bin", "sp_res.bin", "sp_q.bin", "sp_ply.bin", "sp_mat.bin", "sp_hist.bin", "sp_aux.bin"]
ALLOWED = set(SP_FILES) | {"games.txt"}

DEFAULT_CONFIG = {
    "engine": os.path.join(ROOT, "build", "rc0"),
    "python": sys.executable,
    "games_per_task": 32,
    "selfplay_args": {"playouts": 600, "fast-playouts": 100, "full-pct": 25, "gumbel": 16, "noise-plies": 20,
                      "temp-plies": 20, "max-plies": 400, "adj-thr": 0.95, "adj-plies": 8, "adj-min-ply": 40,
                      "verify-pct": 25, "batch-target": 64, "max-batch": 256},
    "rows_per_generation": 100000,     # retrain every 100k new validated rows
    "window_generations": 9,           # the trainer reads the newest N generations of data
    "train_batch": 128,                # KataGo single-box
    "reuse": 8,                        # KataGo: ~8 training samples per new row
    "gating": True,
    "gate_games": 200,                 # KataGo App. E: >= 100 of 200
    "gate_score": 0.5,
    "gate_playouts": 300,
    "match_pairs_per_task": 5,
    "match_opening_plies": 8,
    "task_timeout_s": 7200,
    "max_upload_mb": 400,
    "registrations_per_ip_per_hour": 5,
}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Hub:
    def __init__(self, data):
        self.data = os.path.abspath(data)
        self.lock = threading.RLock()
        self.cfg = dict(DEFAULT_CONFIG)
        self.cfg.update(json.load(open(self.p("config.json"))))
        self.state = json.load(open(self.p("state.json")))
        self.users = json.load(open(self.p("users.json"))) if os.path.exists(self.p("users.json")) else {}
        self.tasks = {}
        self.reg_log = []
        self.log_f = open(self.p("server.log"), "a")

    def p(self, *a):
        return os.path.join(self.data, *a)

    def log(self, msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def save(self):
        with self.lock:
            for name, obj in (("state.json", self.state), ("users.json", self.users)):
                tmp = self.p(name + ".tmp")
                with open(tmp, "w") as f:
                    json.dump(obj, f, indent=1)
                os.replace(tmp, self.p(name))

    def net_ref(self, sha):
        return {"sha256": sha, "onnx": f"/networks/{sha}.onnx"}

    # ------------------------------------------------------------ users
    def register(self, name, ip):
        with self.lock:
            t = time.time()
            self.reg_log = [(i, s) for i, s in self.reg_log if t - s < 3600]
            if sum(1 for i, _ in self.reg_log if i == ip) >= self.cfg["registrations_per_ip_per_hour"]:
                return None
            self.reg_log.append((ip, t))
            name = re.sub(r"[^A-Za-z0-9_.\- ]", "", name or "")[:32].strip() or "anonymous"
            token = secrets.token_hex(16)
            self.users[token] = {"name": name, "created": now(), "games": 0, "rows": 0, "match_games": 0, "rejected": 0}
            self.save()
            return token, name

    # ------------------------------------------------------------ tasks
    def next_task(self, token, version):
        if version < MIN_CLIENT_VERSION:
            return {"type": "upgrade", "message": "download the newest client from the Rc0 releases page"}
        with self.lock:
            t = time.time()
            self.tasks = {k: v for k, v in self.tasks.items() if t - v["issued"] < self.cfg["task_timeout_s"]}
            cand = self.state.get("candidate")
            if cand and self.cfg["gating"]:
                played = len(cand["games"])
                pending = sum(2 * v["pairs"] for v in self.tasks.values() if v["type"] == "match" and v["cand"] == cand["sha"])
                if played + pending < self.cfg["gate_games"]:
                    tid = secrets.token_hex(8)
                    pairs = self.cfg["match_pairs_per_task"]
                    self.tasks[tid] = {"type": "match", "token": token, "issued": t, "pairs": pairs, "cand": cand["sha"]}
                    return {"type": "match", "id": tid, "a": self.net_ref(cand["sha"]), "b": self.net_ref(self.state["current"]),
                            "pairs": pairs, "playouts": self.cfg["gate_playouts"], "seed": secrets.randbits(48),
                            "fen": self.state["start_fen"], "opening_plies": self.cfg["match_opening_plies"],
                            "max_plies": self.cfg["selfplay_args"]["max-plies"]}
            tid = secrets.token_hex(8)
            self.tasks[tid] = {"type": "selfplay", "token": token, "issued": t, "gen": self.state["gen"], "net": self.state["current"]}
            return {"type": "selfplay", "id": tid, "gen": self.state["gen"], "network": self.net_ref(self.state["current"]),
                    "games": self.cfg["games_per_task"], "args": self.cfg["selfplay_args"]}

    def take_task(self, tid, token, kind):
        with self.lock:
            task = self.tasks.get(tid)
            if not task or task["token"] != token or task["type"] != kind:
                return None
            return self.tasks.pop(tid)

    # ------------------------------------------------------------ validation
    def validate(self, d, games_only=False):
        cmd = [self.cfg["engine"], "validate", d] + (["--games-only"] if games_only else [])
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return {"valid": False, "reason": "validation timed out"}
        for line in reversed(r.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    break
        return {"valid": False, "reason": "validator gave no verdict"}

    def accept_selfplay(self, token, tid, body):
        task = self.take_task(tid, token, "selfplay")
        if not task:
            return 404, {"accepted": False, "reason": "unknown or expired task"}
        tmp = tempfile.mkdtemp(prefix="up-", dir=self.p("incoming"))
        try:
            limit = self.cfg["max_upload_mb"] << 20
            total = 0
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
                for m in tar.getmembers():
                    if not m.isfile() or m.name not in ALLOWED:
                        return 400, {"accepted": False, "reason": f"unexpected file in upload: {m.name}"}
                    total += m.size
                    if total > limit:
                        return 413, {"accepted": False, "reason": "upload too large"}
                    src = tar.extractfile(m)
                    with open(os.path.join(tmp, m.name), "wb") as f:
                        shutil.copyfileobj(src, f)
            v = self.validate(tmp)
            user = self.users.get(token, {})
            if not v.get("valid"):
                with self.lock:
                    user["rejected"] = user.get("rejected", 0) + 1
                    self.save()
                self.log(f"REJECTED upload from {user.get('name')}: {v.get('reason')}")
                return 422, {"accepted": False, "reason": v.get("reason")}
            with self.lock:   # append to this generation's data, in the same order as games.txt
                gd = self.p("data", f"gen{task['gen']}")
                os.makedirs(gd, exist_ok=True)
                for name in SP_FILES + ["games.txt"]:
                    with open(os.path.join(tmp, name), "rb") as src, open(os.path.join(gd, name), "ab") as dst:
                        shutil.copyfileobj(src, dst)
                user["games"] = user.get("games", 0) + v["games"]
                user["rows"] = user.get("rows", 0) + v["rows"]
                st = self.state
                st["rows_total"] += v["rows"]
                st["rows_since_train"] += v["rows"]
                st["games_total"] += v["games"]
                st["results"] = {k: st["results"].get(k, 0) + v[k] for k in ("ry", "bg", "draw")}
                self.save()
            self.log(f"accepted {v['games']} games / {v['rows']} rows from {user.get('name')} (gen {task['gen']})")
            return 200, {"accepted": True, "games": v["games"], "rows": v["rows"]}
        except (tarfile.TarError, OSError, EOFError) as e:
            return 400, {"accepted": False, "reason": f"bad archive: {e}"}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def accept_match(self, token, tid, obj):
        task = self.take_task(tid, token, "match")
        if not task:
            return 404, {"accepted": False, "reason": "unknown or expired task"}
        games = obj.get("games", [])
        if len(games) != 2 * task["pairs"]:
            return 400, {"accepted": False, "reason": "wrong number of games"}
        tmp = tempfile.mkdtemp(prefix="match-", dir=self.p("incoming"))
        try:
            cap = self.cfg["selfplay_args"]["max-plies"]
            with open(os.path.join(tmp, "games.txt"), "w") as f:
                for g in games:
                    res, end, moves = int(g["result"]), str(g["end"]), str(g["moves"])
                    if not re.fullmatch(r"[a-n0-9qrbn ]*", moves) or end not in ("natural", "capped"):
                        return 400, {"accepted": False, "reason": "malformed game"}
                    f.write(f"{self.state['start_fen']}\t0\t{cap}\t{end}\t{res}\t{moves}\t\n")
            v = self.validate(tmp, games_only=True)
            if not v.get("valid"):
                self.log(f"REJECTED match upload: {v.get('reason')}")
                return 422, {"accepted": False, "reason": v.get("reason")}
            with self.lock:
                cand = self.state.get("candidate")
                if not cand or cand["sha"] != task["cand"]:
                    return 200, {"accepted": True, "note": "the gate this belonged to has already finished"}
                for g in games:
                    res = int(g["result"])
                    cand["games"].append(0.5 if res == 0 else (1.0 if (res == 1) == bool(g["a_on_ry"]) else 0.0))
                self.users.get(token, {})["match_games"] = self.users.get(token, {}).get("match_games", 0) + len(games)
                if len(cand["games"]) >= self.cfg["gate_games"]:
                    self.finish_gate()
                self.save()
            return 200, {"accepted": True}
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------ generations
    def finish_gate(self):
        cand = self.state["candidate"]
        n = len(cand["games"])
        score = sum(cand["games"])
        passed = score >= self.cfg["gate_score"] * n
        self.state["candidate"] = None
        self.state["gates"].append({"gen": self.state["gen"] + 1, "sha": cand["sha"], "score": score, "games": n,
                                    "passed": passed, "at": now()})
        self.log(f"GATE candidate {cand['sha'][:12]}: {score}/{n} -> {'PROMOTED' if passed else 'rejected'}")
        if passed:
            self.promote(cand["sha"])

    def promote(self, sha):
        with self.lock:
            self.state["gen"] += 1
            self.state["current"] = sha
            self.state["history"].append({"gen": self.state["gen"], "sha": sha, "at": now(), "rows_total": self.state["rows_total"]})
            self.save()
        self.log(f"generation {self.state['gen']} is now {sha[:12]}")

    def train_once(self):
        """One retrain on the KataGo window; the result becomes a candidate (or the new generation if gating is off)."""
        with self.lock:
            st, cfg = self.state, self.cfg
            gen = st["gen"]
            new_rows = st["rows_since_train"]
            st["rows_since_train"] = 0
            st["training"] = True
            self.save()
        out = self.p("training", f"after_gen{gen}_{int(time.time())}")
        os.makedirs(out, exist_ok=True)
        dirs = [self.p("data", f"gen{g}") for g in range(max(0, gen - cfg["window_generations"] + 1), gen + 1)]
        dirs = [d for d in dirs if os.path.isfile(os.path.join(d, "sp_rec.bin"))]
        total = sum(os.path.getsize(os.path.join(d, "sp_rec.bin")) // 136 for d in dirs)
        c, A, B = 250000, 0.65, 0.4      # KataGo window: N = c(1 + B((N_total/c)^A - 1)/A)
        window = int(c * (1 + B * ((max(total, 1) / c) ** A - 1) / A))
        steps = max(50, cfg["reuse"] * new_rows // cfg["train_batch"])
        init = self.p("networks", self.state["current"] + ".pt")
        cmd = [cfg["python"], os.path.join(ROOT, "train", "train_selfplay.py"), "--init", init, "--out", out,
               "--steps", str(steps), "--batch", str(cfg["train_batch"]), "--window", str(window), "--eval-every", str(max(25, steps // 8))]
        for d in dirs:
            cmd += ["--data", d]
        self.log(f"training after gen {gen}: {new_rows} new rows, window {window}, {steps} steps")
        env = dict(os.environ, RC0_BIN=cfg["engine"])
        with open(os.path.join(out, "train.log"), "w") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        best_pt, best_onnx = os.path.join(out, "best.pt"), os.path.join(out, "best.onnx")
        with self.lock:
            self.state["training"] = False
            if r.returncode != 0 or not (os.path.isfile(best_pt) and os.path.isfile(best_onnx)):
                self.log(f"training FAILED (see {out}/train.log)")
                self.state["rows_since_train"] += new_rows
                self.save()
                return
            sha = self.add_network(best_pt, best_onnx)
            if cfg["gating"]:
                self.state["candidate"] = {"sha": sha, "games": [], "since": now()}
                self.log(f"candidate {sha[:12]} ready; gating with {cfg['gate_games']} games")
                self.save()
            else:
                self.save()
                self.promote(sha)

    def add_network(self, pt, onnx):
        sha = sha256_file(onnx)
        shutil.copy2(onnx, self.p("networks", sha + ".onnx"))
        shutil.copy2(pt, self.p("networks", sha + ".pt"))
        return sha

    def trainer_loop(self):
        while True:
            time.sleep(10)
            try:
                with self.lock:
                    ready = (not self.state.get("candidate") and not self.state.get("training")
                             and self.state["rows_since_train"] >= self.cfg["rows_per_generation"])
                if ready:
                    self.train_once()
            except Exception as e:   # never let the trainer thread die silently
                self.log(f"trainer error: {e!r}")
                with self.lock:
                    self.state["training"] = False
                    self.save()

    # ------------------------------------------------------------ status
    def status(self):
        with self.lock:
            st = self.state
            people = sorted(self.users.values(), key=lambda u: -u.get("games", 0))
            return {"generation": st["gen"], "network": st["current"], "games": st["games_total"], "rows": st["rows_total"],
                    "results": st["results"], "rows_until_training": max(0, self.cfg["rows_per_generation"] - st["rows_since_train"]),
                    "training": st.get("training", False),
                    "candidate": ({"sha": st["candidate"]["sha"], "games": len(st["candidate"]["games"]),
                                   "score": sum(st["candidate"]["games"])} if st.get("candidate") else None),
                    "history": st["history"][-50:], "gates": st["gates"][-50:],
                    "contributors": [{"name": u["name"], "games": u.get("games", 0), "rows": u.get("rows", 0),
                                      "match_games": u.get("match_games", 0)} for u in people[:100]]}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rc0 training</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141414;--fg:#eee;--mut:#999;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;width:100%;margin:8px 0 24px}}td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}}
.k{{color:var(--mut)}}code{{font-size:13px}}</style></head><body>
<h1>Rc0 training</h1><p class="k">Leela-style 4-player chess (Teams) engine, trained by self-play from contributors' GPUs.
<a href="https://github.com/itzoctober23-web/rc0">Source and client downloads</a>.</p>
<table><tr><td>Generation</td><td>{gen}</td></tr><tr><td>Network</td><td><code>{net}</code></td></tr>
<tr><td>Games / training rows</td><td>{games:,} / {rows:,}</td></tr>
<tr><td>Results (Red+Yellow / Blue+Green / draw)</td><td>{ry:,} / {bg:,} / {dr:,}</td></tr>
<tr><td>Next training in</td><td>{until:,} rows{training}</td></tr><tr><td>Gate</td><td>{cand}</td></tr></table>
<h2>Contributors</h2><table><tr><th>Name</th><th>Games</th><th>Rows</th><th>Match games</th></tr>{people}</table>
<h2>Gates</h2><table><tr><th>When</th><th>Candidate</th><th>Score</th><th>Result</th></tr>{gates}</table>
</body></html>"""


def render(s):
    esc = html.escape
    people = "".join(f"<tr><td>{esc(c['name'])}</td><td>{c['games']:,}</td><td>{c['rows']:,}</td><td>{c['match_games']:,}</td></tr>"
                     for c in s["contributors"])
    gates = "".join(f"<tr><td>{esc(g['at'])}</td><td><code>{g['sha'][:12]}</code></td><td>{g['score']}/{g['games']}</td>"
                    f"<td>{'promoted' if g['passed'] else 'rejected'}</td></tr>" for g in reversed(s["gates"]))
    c = s["candidate"]
    return PAGE.format(gen=s["generation"], net=s["network"][:16], games=s["games"], rows=s["rows"], ry=s["results"].get("ry", 0),
                       bg=s["results"].get("bg", 0), dr=s["results"].get("draw", 0), until=s["rows_until_training"],
                       training=" (training now)" if s["training"] else "",
                       cand=f"<code>{c['sha'][:12]}</code> {c['score']}/{c['games']}" if c else "none running",
                       people=people, gates=gates)


def make_handler(hub):
    class H(BaseHTTPRequestHandler):
        server_version = "rc0-server/1"

        def log_message(self, fmt, *a):
            pass

        def send(self, code, obj=None, body=None, ctype="application/json"):
            data = body if body is not None else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def token(self):
            a = self.headers.get("Authorization", "")
            t = a[7:] if a.startswith("Bearer ") else ""
            return t if t in hub.users else None

        def body(self, limit):
            n = int(self.headers.get("Content-Length", "0"))
            if n > limit:
                raise ValueError("too large")
            return self.rfile.read(n)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self.send(200, body=render(hub.status()).encode(), ctype="text/html; charset=utf-8")
            if u.path == "/api/status":
                return self.send(200, hub.status())
            m = re.fullmatch(r"/networks/([0-9a-f]{64})\.(onnx|pt)", u.path)
            if m:
                path = hub.p("networks", f"{m.group(1)}.{m.group(2)}")
                if not os.path.isfile(path):
                    return self.send(404, {"error": "no such network"})
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(os.path.getsize(path)))
                self.end_headers()
                with open(path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                return
            if u.path == "/api/task":
                tok = self.token()
                if not tok:
                    return self.send(401, {"error": "register first"})
                return self.send(200, hub.next_task(tok, int(q.get("version", ["0"])[0])))
            self.send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/api/register":
                    obj = json.loads(self.body(4096) or b"{}")
                    r = hub.register(str(obj.get("name", "")), self.client_address[0])
                    if not r:
                        return self.send(429, {"error": "too many registrations from this address"})
                    return self.send(200, {"token": r[0], "name": r[1]})
                tok = self.token()
                if not tok:
                    return self.send(401, {"error": "register first"})
                tid = q.get("task", [""])[0]
                if u.path == "/api/upload":
                    code, obj = hub.accept_selfplay(tok, tid, self.body(hub.cfg["max_upload_mb"] << 20))
                    return self.send(code, obj)
                if u.path == "/api/match":
                    code, obj = hub.accept_match(tok, tid, json.loads(self.body(16 << 20)))
                    return self.send(code, obj)
                self.send(404, {"error": "not found"})
            except ValueError as e:
                self.send(400, {"error": str(e)})
    return H


def cmd_init(a):
    os.makedirs(a.data, exist_ok=True)
    if os.path.exists(os.path.join(a.data, "state.json")):
        sys.exit(f"{a.data} already holds a server; refusing to overwrite")
    for d in ("networks", "data", "incoming", "training"):
        os.makedirs(os.path.join(a.data, d), exist_ok=True)
    cfg = dict(DEFAULT_CONFIG)
    if a.engine:
        cfg["engine"] = os.path.abspath(a.engine)
    if a.python:
        cfg["python"] = a.python
    net = a.net
    if a.random:
        b, f = (int(x) for x in a.random.lower().split("x"))
        net = os.path.join(a.data, "training", "gen0.pt")
        subprocess.run([cfg["python"], os.path.join(ROOT, "train", "model.py"), "--blocks", str(b), "--filters", str(f), "--out", net], check=True)
    if not net:
        sys.exit("pass --net NET.pt (with NET.onnx next to it) or --random 15x192")
    onnx = net[:-3] + ".onnx"
    if not os.path.isfile(onnx):
        sys.exit(f"{onnx} not found: export it with train/export_best.py")
    out = subprocess.run([cfg["engine"]], input="position startpos\nd\nquit\n", stdout=subprocess.PIPE, text=True).stdout
    fen = [l.split(" ", 1)[1] for l in out.splitlines() if l.startswith("fen4 ")][0]
    hub_state = {"gen": 0, "current": None, "start_fen": fen, "rows_total": 0, "rows_since_train": 0, "games_total": 0,
                 "results": {"ry": 0, "bg": 0, "draw": 0}, "candidate": None, "training": False, "history": [], "gates": []}
    with open(os.path.join(a.data, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    sha = sha256_file(onnx)
    shutil.copy2(onnx, os.path.join(a.data, "networks", sha + ".onnx"))
    shutil.copy2(net, os.path.join(a.data, "networks", sha + ".pt"))
    hub_state["current"] = sha
    hub_state["history"].append({"gen": 0, "sha": sha, "at": now(), "rows_total": 0})
    with open(os.path.join(a.data, "state.json"), "w") as f:
        json.dump(hub_state, f, indent=1)
    print(f"initialised {a.data}: generation 0 = {sha[:12]}; edit config.json, then run `serve`")


def cmd_serve(a):
    hub = Hub(a.data)
    hub.state["training"] = False
    threading.Thread(target=hub.trainer_loop, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(hub))
    hub.log(f"serving generation {hub.state['gen']} on {a.host}:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        hub.save()


def main():
    ap = argparse.ArgumentParser(description="Rc0 training server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--data", required=True)
    i.add_argument("--net")
    i.add_argument("--random", help="start from a random network, e.g. 15x192")
    i.add_argument("--engine")
    i.add_argument("--python")
    s = sub.add_parser("serve")
    s.add_argument("--data", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    (cmd_init if a.cmd == "init" else cmd_serve)(a)


if __name__ == "__main__":
    main()
