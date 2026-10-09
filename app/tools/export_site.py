"""Build the static site for GitHub Pages: app/site/ (copy it to the Pages repository).

    .venv/bin/python app/tools/export_site.py https://QUEUE-ADDRESS [custom.domain]

The page, its script and style, config.js pointing at the public queue, and the saved example
rewrites (so the examples still open when the home GPU and queue are offline)."""
import json
import shutil
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
SITE = DEMO / "site"
api = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else ""

shutil.rmtree(SITE, ignore_errors=True)
(SITE / "examples").mkdir(parents=True)
for f in ("index.html", "style.css", "app.js", "theme.js"):
    shutil.copy(DEMO / "web" / f, SITE / f)
shutil.copytree(DEMO / "web" / "map", SITE / "map")   # the map of science (images, field labels)
domain = sys.argv[2] if len(sys.argv) > 2 else ""
# support links: the Stripe ones only when they are live (test links take no real payment)
support = json.loads((DEMO / "support.json").read_text()) if (DEMO / "support.json").exists() else {}
stripe = support.get("stripe", {}) if support.get("stripe_mode") == "live" else {}
config = {"api": api, "support": {"github": "https://github.com/sponsors/MaximeRivest", "stripe": stripe}}
(SITE / "config.js").write_text(f"window.SRL_CONFIG = {json.dumps(config)};\n")
# Each version's files under their own address (?v=fingerprint): a browser that kept yesterday's
# script for a few minutes (GitHub Pages lets it) never runs it with today's page.
import hashlib
page = (SITE / "index.html").read_text()
for f in ("config.js", "app.js", "style.css", "theme.js"):
    v = hashlib.sha256((SITE / f).read_bytes()).hexdigest()[:10]
    assert page.count(f'"{f}"') == 1, f
    page = page.replace(f'"{f}"', f'"{f}?v={v}"')
(SITE / "index.html").write_text(page)
if domain:
    (SITE / "CNAME").write_text(domain + "\n")
(SITE / ".nojekyll").write_text("")
index = []
for meta_file in sorted((DEMO / "rewrites").glob("*/paper.json")):
    meta = json.loads(meta_file.read_text())
    folder = meta_file.parent
    rewrites = {f.stem: json.loads(f.read_text()) for f in folder.glob("*.json") if f.stem != "paper"}
    if not rewrites:
        continue
    # every saved rewrite goes with the site: the library reads fine while the home machine is off
    (SITE / "examples" / f"{folder.name}.json").write_text(json.dumps(rewrites, ensure_ascii=False))
    if not meta.get("example"):
        continue
    index.append({**meta, "plain_title": (rewrites.get("our-9b") or {}).get("parts", {}).get("title")})
(SITE / "examples" / "index.json").write_text(json.dumps(index, ensure_ascii=False))
# the library as of publishing, for when the queue server is offline
try:
    import urllib.request
    lib = urllib.request.urlopen("http://127.0.0.1:8795/api/library", timeout=10).read()
    (SITE / "examples" / "library.json").write_bytes(lib)
except OSError:
    (SITE / "examples" / "library.json").write_text(json.dumps(index, ensure_ascii=False))
print(f"{SITE}: {len(index)} example papers; queue: {api or '(none: examples and your own keys only)'}")
