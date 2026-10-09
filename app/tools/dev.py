"""Development server for sciencemadereadable.com: the working copy of app/web, private, live-reloading.

    .venv/bin/python app/tools/dev.py [--port 8940] [--data prod|dev|public|URL] [--queue prod|public|URL]

Serves app/web as it is on disk (build with `node tools/build.mjs --watch`, which dev.sh starts), with:
  /api/data/*  -> the data API (search, map, walk). prod: 127.0.0.1:8810 (what the public site uses);
                  dev: 127.0.0.1:8811 (search/dev.sh up: a second instance running a branch);
                  public: https://api.sciencemadereadable.com (from a machine other than lambda)
  /api/*       -> the queue server's public door (rewrites, library, support): prod 127.0.0.1:8799
  config.js    -> {"api": ""}: everything goes through this server, same origin
  every page   -> a DEV badge naming the backends, and a reload as soon as app.js / style.css / index.html
                  change on disk (polls /__dev/version; the page's policy allows only our own scripts, so the
                  reloader is a file served from here, never inline)
Nothing is cached. Bound to 127.0.0.1; dev.sh puts it on the tailnet only (tailscale serve), never public.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
BACKENDS = {"data": {"prod": "http://127.0.0.1:8810", "dev": "http://127.0.0.1:8811",
                     "public": "https://api.sciencemadereadable.com/api/data"},
            "queue": {"prod": "http://127.0.0.1:8799", "public": "https://api.sciencemadereadable.com"}}
WATCH = ["app.js", "style.css", "index.html", "theme.js", "map/regions.json"]

RELOAD_JS = """// dev only (app/tools/dev.py): a badge, and a reload when the files change on disk
(function () {
  var b = document.createElement("div");
  b.textContent = "DEV · data: %(data)s · queue: %(queue)s";
  b.style.cssText = "position:fixed;z-index:99;left:8px;top:8px;background:#b3402a;color:#fff;font:600 11px/1.6 system-ui;" +
    "padding:1px 8px;border-radius:999px;opacity:.85;pointer-events:none";
  document.addEventListener("DOMContentLoaded", function () { document.body.appendChild(b); });
  var seen = null;
  setInterval(function () {
    fetch("/__dev/version", { cache: "no-store" }).then(function (r) { return r.text(); }).then(function (v) {
      if (seen !== null && v !== seen) location.reload();
      seen = v;
    }).catch(function () {});
  }, 800);
})();
"""


def make_handler(data: str, queue: str, names: tuple[str, str]):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(WEB), **k)

        def log_message(self, fmt, *args):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def send_body(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def proxy(self, url: str, method: str = "GET", body: bytes | None = None):
            req = urllib.request.Request(url, data=body, method=method)
            for h in ("Content-Type", "Range"):
                if self.headers.get(h):
                    req.add_header(h, self.headers[h])
            try:
                with urllib.request.urlopen(req, timeout=75) as r:
                    code, out, ctype = r.status, r.read(), r.headers.get("Content-Type", "application/json")
            except urllib.error.HTTPError as e:
                code, out, ctype = e.code, e.read(), e.headers.get("Content-Type", "application/json")
            except OSError as e:
                code, out, ctype = 502, json.dumps({"error": f"dev proxy: {url.split('?')[0]} unreachable ({e})"}).encode(), "application/json"
            self.send_body(code, out, ctype)

        def route(self, method: str):
            p = self.path
            if p.startswith("/api/data/"):
                return self.proxy(data + p[len("/api/data"):], method)
            if p.startswith("/api/"):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
                return self.proxy(queue + p, method, body)
            return None

        def do_POST(self):
            if self.route("POST") is None:
                self.send_body(404, b'{"error":"not found"}', "application/json")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path.startswith("/api/"):
                return self.route("GET")
            if path == "/config.js":
                return self.send_body(200, b'window.SRL_CONFIG = {"api": ""};\n', "text/javascript")
            if path == "/__dev/reload.js":
                js = RELOAD_JS % {"data": names[0], "queue": names[1]}
                return self.send_body(200, js.encode(), "text/javascript")
            if path == "/__dev/version":
                v = ":".join(str((WEB / f).stat().st_mtime_ns) if (WEB / f).exists() else "-" for f in WATCH)
                return self.send_body(200, v.encode(), "text/plain")
            if path in ("/", "/index.html"):
                page = (WEB / "index.html").read_text()
                page = page.replace("</head>", '<script src="/__dev/reload.js"></script>\n</head>', 1)
                return self.send_body(200, page.encode(), "text/html; charset=utf-8")
            return super().do_GET()

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8940)
    ap.add_argument("--data", default="prod")
    ap.add_argument("--queue", default="prod")
    a = ap.parse_args()
    data = BACKENDS["data"].get(a.data, a.data).rstrip("/")
    queue = BACKENDS["queue"].get(a.queue, a.queue).rstrip("/")
    print(f"dev site on http://127.0.0.1:{a.port}/  data -> {data}  queue -> {queue}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(data, queue, (a.data, a.queue))).serve_forever()


if __name__ == "__main__":
    main()
