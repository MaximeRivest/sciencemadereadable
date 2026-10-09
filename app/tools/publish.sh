#!/usr/bin/env bash
# Put what you see on the dev site on sciencemadereadable.com, checked before and after.
#
#   app/tools/publish.sh            the committed, pushed state of app/ -> GitHub Pages
#   app/tools/publish.sh rollback   the previous published version back (one revert on gh-pages)
#   app/tools/publish.sh queue      restart the queue server (app/server.py changes), only while no rewrite runs
#
# Steps: refuse uncommitted or unpushed changes under app/ (the live site is always a commit you can name);
# build; walk through the dev site in a browser (dev.sh check); export; one commit on gh-pages naming the
# source commit; push; wait until GitHub Pages serves it; walk through the live site. If the live check
# fails it tells you; `publish.sh rollback` puts the previous version back in about a minute.
set -euo pipefail
APP="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$APP/.." && pwd)"
SITE="${SITE:-$HOME/Projects/sciencemadereadable-site}"
API="https://api.sciencemadereadable.com"
LIVE="https://sciencemadereadable.com/"

wait_live() {
  local v; v=$(grep -o 'app.js?v=[a-z0-9]*' "$SITE/index.html")
  echo "waiting for GitHub Pages to serve $v"
  for _ in $(seq 1 40); do curl -s "$LIVE" | grep -q "$v" && { echo "live"; return 0; }; sleep 15; done
  echo "not live after 10 minutes (GitHub Pages is slow sometimes); check $LIVE" >&2; return 1
}

case "${1:-}" in
  rollback)
    git -C "$SITE" pull -q --ff-only
    echo "reverting: $(git -C "$SITE" log -1 --format='%h %s')"
    git -C "$SITE" revert --no-edit HEAD >/dev/null
    git -C "$SITE" push -q
    wait_live
    exit 0
    ;;
  queue)
    s=$(curl -s 127.0.0.1:8795/api/status)
    echo "$s" | grep -q '"queued": 0, "running": 0' || { echo "rewrites are running or waiting ($s); try again later" >&2; exit 1; }
    systemctl --user restart smr-server
    for _ in $(seq 1 30); do curl -sf -o /dev/null "$API/api/status" && break; sleep 1; done
    curl -s "$API/api/status"; echo
    exit 0
    ;;
  "") ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac

cd "$REPO"
if [ -n "$(git status --porcelain -- app/web app/tools app/server.py)" ]; then
  git status --short -- app/web app/tools app/server.py
  echo "commit these first: the live site is always a commit you can name" >&2; exit 1
fi
git fetch -q origin
[ "$(git rev-parse HEAD)" = "$(git rev-parse '@{u}')" ] || { echo "push first (git push)" >&2; exit 1; }
SRC=$(git rev-parse --short HEAD)

(cd "$APP" && node tools/build.mjs)
if [ "${SKIP_DEV_CHECK:-}" != 1 ]; then
  curl -sf -o /dev/null http://127.0.0.1:8940/ || { echo "the dev site isn't running (app/tools/dev.sh up), or set SKIP_DEV_CHECK=1" >&2; exit 1; }
  "$APP/tools/dev.sh" check || { echo "the dev site fails its checks: not publishing" >&2; exit 1; }
fi

.venv/bin/python app/tools/export_site.py "$API" sciencemadereadable.com
git -C "$SITE" pull -q --ff-only
rsync -a --delete --exclude .git app/site/ "$SITE"/
git -C "$SITE" add -A
if git -C "$SITE" diff --cached --quiet; then echo "nothing changed on the site"; exit 0; fi
git -C "$SITE" diff --cached --stat | tail -15
git -C "$SITE" commit -q -m "site from $SRC: $(git log -1 --format=%s | cut -c1-80)"
git -C "$SITE" push -q
wait_live
"$APP/tools/dev.sh" check "$LIVE" || { echo "LIVE CHECK FAILED. Undo with: app/tools/publish.sh rollback" >&2; exit 1; }
echo "published $SRC"
