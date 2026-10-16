#!/usr/bin/env bash
# Immutable, root-owned releases; the declarative NixOS unit runs as smr-editions.
set -euo pipefail
APP=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$APP/.." && pwd)
BASE=/opt/smr-editions
mode=${1:-status}

quiet_worker() {
  if sudo test -f /var/lib/smr-editions/jobs.sqlite; then
    sudo touch /var/lib/smr-editions/maintenance
    trap 'sudo rm -f /var/lib/smr-editions/maintenance' EXIT
    n=$(sudo "$BASE/current/app/editions/.venv/bin/python" -c "import sqlite3; c=sqlite3.connect('/var/lib/smr-editions/jobs.sqlite',timeout=20); c.execute('BEGIN IMMEDIATE'); print(c.execute(\"select count(*) from jobs where state in ('extracting','classifying','rewriting','exporting','audio')\").fetchone()[0]); c.commit()")
    [[ "$n" == 0 ]] || { echo 'A private job is processing; wait before switching releases.' >&2; exit 1; }
  fi
}
health() {
  for _ in $(seq 1 30); do
    if systemctl is-active --quiet smr-editions-worker && curl -fsS --max-time 2 http://127.0.0.1:8825/api/editions/health | "$BASE/current/app/editions/.venv/bin/python" -c 'import json,sys; d=json.load(sys.stdin);sys.exit(0 if d.get("ok") and d.get("worker_online") and d.get("worker_seen",0)>=float(sys.argv[1]) else 1)' "${HEALTH_AFTER:-0}" 2>/dev/null; then return 0; fi
    sleep 1
  done
  return 1
}
restart_and_check() {
  sudo systemctl restart smr-editions smr-editions-worker || return 1
  HEALTH_AFTER=$(date +%s.%N)
  health
}
switch_to() {
  sudo ln -sfn "$1" "$BASE/current.new"
  sudo mv -Tf "$BASE/current.new" "$BASE/current"
}

case "$mode" in
  prepare)
    cd "$REPO"
    [[ -z "$(git status --porcelain -- app/editions reading_engine glossary/glossary.py glossary/inputs.py app/web/src/programs.ts)" ]] || { echo 'Commit the service/engine changes before preparing a release.' >&2; exit 1; }
    rev=$(git rev-parse HEAD)
    dest="$BASE/releases/$rev"
    if sudo test -f "$dest/RELEASE.json"; then echo "$dest"; exit 0; fi
    node --conditions=functai-source app/tools/check_prompts.ts
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    git archive HEAD app/editions reading_engine glossary/glossary.py glossary/inputs.py | tar -x -C "$tmp"
    "$APP/node_modules/.bin/esbuild" reading_engine/rewrite.ts --bundle --conditions=functai-source --platform=node --format=esm \
      --outfile="$tmp/app/editions/rewrite.mjs" \
      --banner:js='import { createRequire as __smrRequire } from "node:module"; const require = __smrRequire(import.meta.url);'
    python_store=$(nix build nixpkgs#python313 --no-link --print-out-paths)
    sudo mkdir -p "$dest" "$BASE/reference"
    sudo rsync -a --chown=root:root "$tmp/" "$dest/"
    sudo ln -s /opt/smr-editions/reference "$dest/glossary/data"
    sudo nix-store --add-root "$dest/.python-root" --indirect -r "$python_store" >/dev/null
    sudo env UV_CACHE_DIR=/var/cache/smr-editions-uv UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
      "$(command -v uv)" sync --frozen --no-dev --python "$python_store/bin/python3.13" --project "$dest/app/editions"
    sudo chmod -R a+rX "$dest"
    sudo env LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib "$dest/app/editions/.venv/bin/python" -c \
      "import fastapi,pymupdf,playwright,wordfreq,tiktoken; print('runtime imports OK')"
    sudo "$dest/app/editions/.venv/bin/python" -c \
      "import json,hashlib,pathlib; p=pathlib.Path('$dest'); (p/'RELEASE.json').write_text(json.dumps({'commit':'$rev','python':'$python_store','worker_sha256':hashlib.sha256((p/'app/editions/rewrite.mjs').read_bytes()).hexdigest()},indent=2))"
    echo "$dest"
    ;;
  activate)
    rev=${2:-$(git -C "$REPO" rev-parse HEAD)}
    [[ "$rev" =~ ^[a-f0-9]{40}$ ]] || { echo 'Expected a full commit hash.' >&2; exit 2; }
    dest="$BASE/releases/$rev"
    sudo test -f "$dest/RELEASE.json" || { echo 'Prepare this release first.' >&2; exit 1; }
    quiet_worker
    old=$(readlink "$BASE/current" || true)
    if [[ -n "$old" ]]; then sudo ln -sfn "$old" "$BASE/previous"; fi
    switch_to "$dest"
    if ! restart_and_check; then
      if [[ -n "$old" ]]; then switch_to "$old"; restart_and_check; fi
      echo 'New release did not pass health; previous release restored when available.' >&2
      exit 1
    fi
    echo "Reading service active: $rev"
    ;;
  rollback)
    old=$(readlink "$BASE/previous")
    quiet_worker
    switch_to "$old"
    restart_and_check
    echo "Restored $old"
    ;;
  enable)
    health
    sudo touch /var/lib/smr-editions/public-enabled
    echo 'Public creation of reading jobs enabled.'
    ;;
  disable)
    sudo rm -f /var/lib/smr-editions/public-enabled
    echo 'New public jobs disabled. Existing private jobs and downloads remain available.'
    ;;
  status)
    readlink "$BASE/current" || true
    systemctl is-active smr-editions smr-editions-worker || true
    curl -fsS --max-time 3 http://127.0.0.1:8825/api/editions/health || true
    echo
    ;;
  *) echo 'Usage: release.sh prepare|activate [COMMIT]|enable|disable|rollback|status' >&2; exit 2 ;;
esac
