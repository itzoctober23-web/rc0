#!/bin/bash
# Set up a public Rc0 hub on a Linux machine that also trains (rootless Podman + Cloudflare Tunnel + systemd user units).
#   NET_FILE=runs/main/NET.txt GEN_FILE=runs/main/GEN PYTHON=/path/to/python-with-torch deploy/setup.sh
# Creates: ~/rc0-server/{data,inbox}, image localhost/rc0-hub, units rc0-hub (container), rc0-bridge, rc0-tunnel.
# SPDX-License-Identifier: GPL-3.0-or-later
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE=${RC0_SERVER_DIR:-$HOME/rc0-server}
PY=${PYTHON:-python3}
: "${NET_FILE:?set NET_FILE to the file holding the current network path}"
: "${GEN_FILE:?set GEN_FILE to the file holding the current generation}"
CPUS=${HUB_CPUSET:-0-11}
PORT=${HUB_PORT:-8100}
CLOUDFLARED=${CLOUDFLARED:-$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")}

mkdir -p "$BASE/data" "$BASE/inbox"
chmod 755 "$BASE/inbox"

echo "== building the hub image"
podman build -t localhost/rc0-hub:latest -f "$ROOT/deploy/Containerfile" "$ROOT"

if [ ! -f "$BASE/data/state.json" ]; then
  echo "== initialising server state (external trainer)"
  podman run --rm -v "$BASE/data:/data:U,Z" --entrypoint python3 localhost/rc0-hub:latest \
    /app/server/rc0_server.py init --data /data --external --engine /app/build/rc0
fi

UNITS=$HOME/.config/systemd/user
QUAD=$HOME/.config/containers/systemd
mkdir -p "$UNITS" "$QUAD"
cat > "$QUAD/rc0-hub.container" <<EOF
# Rc0 hub: rootless, read-only, no capabilities, its own user namespace; sees only its data and the read-only inbox.
[Unit]
Description=Rc0 training hub (container)

[Container]
Image=localhost/rc0-hub:latest
ContainerName=rc0-hub
PublishPort=127.0.0.1:$PORT:8000
Volume=$BASE/data:/data:U,Z
Volume=$BASE/inbox:/inbox:ro,Z
ReadOnly=true
Tmpfs=/tmp:rw,size=64m
DropCapability=all
NoNewPrivileges=true
Environment=PYTHONDONTWRITEBYTECODE=1
PodmanArgs=--cpus=2 --memory=2g --pids-limit=256

[Service]
Restart=always
Nice=10
# CPU affinity (inherited by the container): rootless cgroups usually lack the cpuset controller
CPUAffinity=$CPUS

[Install]
WantedBy=default.target
EOF

cat > "$UNITS/rc0-bridge.service" <<EOF
[Unit]
Description=Rc0 bridge: publish the trainer's current network to the hub inbox

[Service]
ExecStart=$PY $ROOT/deploy/bridge.py --net-file $NET_FILE --gen-file $GEN_FILE --inbox $BASE/inbox --python $PY
Restart=always
RestartSec=30
Nice=15
CPUAffinity=$CPUS
IOSchedulingClass=idle

[Install]
WantedBy=default.target
EOF

cat > "$UNITS/rc0-tunnel.service" <<EOF
[Unit]
Description=Rc0 hub public address (Cloudflare Tunnel; no open router port)
After=rc0-hub.service

[Service]
ExecStart=$CLOUDFLARED tunnel --no-autoupdate --url http://127.0.0.1:$PORT
Restart=always
RestartSec=10
CPUAffinity=$CPUS

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
T0=$(date "+%F %T")
systemctl --user restart rc0-hub.service rc0-bridge.service rc0-tunnel.service
systemctl --user enable rc0-bridge.service rc0-tunnel.service >/dev/null
echo "== waiting for the public address"
for i in $(seq 1 30); do
  URL=$(journalctl --user -u rc0-tunnel --since "$T0" -o cat | grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1 || true)
  [ -n "$URL" ] && break
  sleep 2
done
echo "hub:     http://127.0.0.1:$PORT  (local)"
echo "public:  ${URL:-not up yet: journalctl --user -u rc0-tunnel}"
echo "invite:  $PY $ROOT/server/rc0_server.py invite --file $BASE/inbox/invites.json --name NAME"
