#!/usr/bin/env bash
# The private dev site: your working copy of app/web, live, on the tailnet only.
#
#   app/tools/dev.sh up [--data prod|dev|public|URL]   start (or restart with other backends)
#   app/tools/dev.sh down | status | logs
#   app/tools/dev.sh check [URL]                       a browser walk-through (default: the dev site)
#
# Open https://lambda.tail69222b.ts.net:18940/ (any of your devices, phone included). Save a file in
# app/web/src: esbuild rebuilds app.js in ~0.1 s and the open page reloads by itself. A red DEV badge says
# which backends it talks to. The data backend "dev" is a second search API running a branch
# (search/dev.sh); "prod" is the one the public site uses (read-only, so sharing it is safe).
# The queue (rewrites, library) is always the real one: rewrites asked for from dev are real rewrites.
set -euo pipefail
APP="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$APP/.." && pwd)"
PORT=8940
TS_PORT=18940
URL="https://lambda.tail69222b.ts.net:$TS_PORT/"
NODE="$(command -v node)"
CHROMIUM_BIN="${CHROMIUM_BIN:-$(ls -d /nix/store/*-chromium-*/bin/chromium 2>/dev/null | head -1)}"

stop() { systemctl --user stop smr-dev-site smr-dev-build 2>/dev/null || true
         systemctl --user reset-failed smr-dev-site smr-dev-build 2>/dev/null || true; }

case "${1:-status}" in
  up)
    shift
    DATA=prod
    while [ $# -gt 0 ]; do case "$1" in --data) DATA="$2"; shift 2;; *) echo "unknown: $1" >&2; exit 2;; esac; done
    stop
    (cd "$APP" && "$NODE" tools/build.mjs)   # a fresh app.js before serving
    systemd-run --user -q --unit=smr-dev-build --description="sciencemadereadable dev: esbuild --watch" \
      -p WorkingDirectory="$APP" -p Nice=5 "$NODE" tools/build.mjs --watch
    systemd-run --user -q --unit=smr-dev-site --description="sciencemadereadable dev site (127.0.0.1:$PORT, tailnet :$TS_PORT)" \
      -p WorkingDirectory="$REPO" -p Restart=on-failure -p MemoryMax=1G \
      "$REPO/.venv/bin/python" -u app/tools/dev.py --port "$PORT" --data "$DATA"
    tailscale serve --bg --https="$TS_PORT" "http://127.0.0.1:$PORT" >/dev/null
    for _ in $(seq 1 20); do curl -sf -o /dev/null "http://127.0.0.1:$PORT/" && break; sleep 0.3; done
    echo "dev site: $URL   (data: $DATA; private to your tailnet)"
    ;;
  down)
    stop
    tailscale serve --https="$TS_PORT" off >/dev/null 2>&1 || true
    echo "dev site stopped"
    ;;
  status)
    systemctl --user is-active smr-dev-site smr-dev-build 2>/dev/null | paste -sd' ' | sed 's/^/site, build: /'
    journalctl --user -u smr-dev-site -n 1 -o cat --no-pager 2>/dev/null | grep -o 'data -> .*' || true
    echo "$URL"
    ;;
  logs)
    journalctl --user -u smr-dev-site -u smr-dev-build -f -o cat
    ;;
  check)
    TARGET="${2:-http://127.0.0.1:$PORT/}"
    cd "$REPO/../citewalk/science"   # its uv project has a Python that runs Playwright
    CHROMIUM_BIN="$CHROMIUM_BIN" uv run -q --with playwright==1.55.0 python "$APP/tools/check_site.py" "$TARGET"
    ;;
  *) sed -n '2,13p' "$0"; exit 2 ;;
esac
