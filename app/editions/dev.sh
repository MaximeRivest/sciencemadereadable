#!/usr/bin/env bash
# Private staging only. The main dev site's editions_proxy points at :8826.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
case "${1:-status}" in
  up)
    systemctl --user stop smr-reading-dev 2>/dev/null || true
    systemd-run --user --unit=smr-reading-dev --description='Reading editions private staging' \
      -p WorkingDirectory="$HERE" -p MemoryMax=6G -p TasksMax=256 -p CPUQuota=400% -p Restart=on-failure \
      -E SMR_READING_STATE=/tmp/smr-reading-dev -E SMR_READING_DEV=1 \
      -E 'SMR_READING_ORIGINS=http://127.0.0.1:8940,http://localhost:8940,https://lambda.tail69222b.ts.net:18940' \
      -E LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib \
      -E CREDENTIALS_DIRECTORY=/home/maxime/.config/scholarsreadinglist \
      -E SMR_AUDIO_MAX_SECONDS=5400 \
      -E CHROMIUM_BIN="${CHROMIUM_BIN:?Use a stock headless Chromium, not the household launcher}" \
      "$HERE/.venv/bin/python" -m uvicorn service:create_app --factory --host 127.0.0.1 --port 8826 --no-access-log --no-proxy-headers
    ;;
  down) systemctl --user stop smr-reading-dev ;;
  status) systemctl --user status smr-reading-dev --no-pager ;;
  *) echo 'Usage: dev.sh up|down|status' >&2; exit 2 ;;
esac
