"""The demo's paper service and web server.

    .venv/bin/python app/server.py [--port 8795] [--host 127.0.0.1]

GET /api/paper?doi=...   the paper as our models read it: sections (split exactly as the training
                         corpus was), the reference glossary for each piece, the reply limits the
                         benchmark used. Used by the worker (on this machine). Only CC BY papers.
GET /api/rewrites?doi=   the saved rewrites of a paper, by model (examples, and our models' jobs).
GET /api/examples        the papers with saved rewrites ready to read (no key needed).
GET /api/search?q=...&mode=auto|semantic|keyword&year_from=YYYY
                         search over the papers we can open (our index on this machine: CC BY
                         research articles with PMC full text): by meaning, or by exact words when the
                         query uses quotes, AND / OR / NOT, brackets or word*. Answers
                         {"hits": [...], "mode": "semantic"|"keyword", "matches": n}; the page
                         falls back to Europe PMC when this fails. At most 30 searches a minute per
                         address. The words searched are not stored.
GET /api/check?ids=PMC1,PMC2,...   which search results can be opened: "ok", or why not ("licence",
                         "layout", "no full text"), by the same rules as opening the paper. Kept forever
                         in app/checks.jsonl. Answers as soon as one is known, with all known then;
                         the page asks again for the rest ("?": couldn't be checked).
GET /api/status          is the home GPU worker online, how long is the queue.
GET /api/data/...        the data API (search/srl_search, release in DATA.md), read-only, as is: search,
                         similar/REF, works/REF, map/place, map/regions, map/counts, map/tiles/RELEASE/..., walk.
                         Per-address limits; tiles are immutable (cached a year).
GET /api/now             what is being rewritten right now (titles, progress) and the latest finished.
GET /api/support         what the GPU costs a day, support of the last 24 h (Stripe, GitHub Sponsors), the
                         Stripe links, today's sponsor.
GET /api/library         every paper with a saved rewrite (titles, journal, year, which models).
POST /api/event          a usage count from the page (no cookies; see stats.py).
GET /stats               the usage dashboard: only for people on the tailnet (Tailscale says who).
POST /api/jobs           {doi, model}: ask our model (running on a home GPU) to rewrite a paper.
GET /api/jobs/ID         that job: queued (and where in the line), running (the text so far), done.
POST /api/worker/next    (worker, with its token) take the next job.
POST /api/worker/jobs/ID (worker) progress or the result; a finished rewrite is saved for everyone.
everything else          the web app (app/web/).

No API key ever reaches this server: the app calls Anthropic and OpenAI from the browser
with the reader's own key. Papers are cached in app/cache/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait as wait_all
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in ("paper_corpus", "rewrite_benchmark", "glossary"):
    sys.path.insert(0, str(ROOT / p))
import glossary as G                                     # noqa: E402  (offline Wikipedia copy when present)
import tiktoken                                          # noqa: E402
from harvest import license_ok, split_sections           # noqa: E402  (the corpus's own rules)
from inputs import glossary_lines                        # noqa: E402

WEB = HERE / "web"
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
STORE = HERE / "rewrites"
STORE.mkdir(exist_ok=True)
JOBS: dict[str, dict] = {}
QUEUE: list[str] = []
JOBS_LOCK = threading.Lock()
WORKERS: dict[str, dict] = {}   # each worker by name: {seen, models (the ones answering now), parallel}


def online_workers() -> list[dict]:
    return [w for w in WORKERS.values() if time.time() - w["seen"] < 30]


def models_online() -> list[str]:
    return sorted({m for w in online_workers() for m in w["models"]})


def worker_online() -> bool:
    return bool(online_workers())
TOKEN_FILE = HERE / "worker_token"
if not TOKEN_FILE.exists():
    TOKEN_FILE.write_text(__import__("secrets").token_urlsafe(32))
    TOKEN_FILE.chmod(0o600)
WORKER_TOKEN = TOKEN_FILE.read_text().strip()
MAX_QUEUE, MAX_PER_ADDRESS = 30, 2
ASKS: dict[str, list[float]] = {}
RECENT_DONE: list[dict] = []   # the last rewrites finished (for "right now")
JOBS_FILE = HERE / "jobs.json"   # the queue, saved every 2 s when it changed: a restart loses no rewrite


def save_jobs_forever():
    last = None
    while True:
        time.sleep(2)
        with JOBS_LOCK:
            state = json.dumps({"jobs": JOBS, "queue": QUEUE, "recent": RECENT_DONE}, ensure_ascii=False)
        if state != last:
            tmp = JOBS_FILE.with_suffix(".tmp")
            tmp.write_text(state)
            tmp.replace(JOBS_FILE)
            last = state


if JOBS_FILE.exists():
    try:
        _saved = json.loads(JOBS_FILE.read_text())
        JOBS.update(_saved["jobs"])
        QUEUE.extend(j for j in _saved["queue"] if j in JOBS)
        RECENT_DONE.extend(_saved.get("recent", []))
        for _j in JOBS.values():   # a running job carries on: its worker is still writing it
            _j["updated"] = time.time()
    except (ValueError, KeyError):
        pass   # address → times it asked for a rewrite (last minute)
import stats   # noqa: E402  (app/stats.py)
ENC = tiktoken.get_encoding("o200k_base")
OPENING = ["title", "abstract", "introduction_first", "conclusion"]
OTHER = ["introduction_rest", "methods", "results", "discussion"]
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
GLOSSARY_LOCK = threading.Lock()


class Refused(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "sciencemadereadable (https://sciencemadereadable.com)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def normalize_doi(raw: str) -> str:
    doi = urllib.parse.unquote(raw.strip())
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I).strip().rstrip(".")
    if not re.match(r"^10\.\d{4,9}/\S+$", doi):
        raise Refused(400, "That doesn't look like a DOI. A DOI starts with 10., like 10.1371/journal.pbio.3002345.")
    return doi.lower()


def paper(doi: str) -> dict:
    cached = CACHE / (hashlib.sha256(doi.encode()).hexdigest()[:24] + ".json")
    if cached.exists():
        return json.loads(cached.read_text())
    q = urllib.parse.urlencode({"query": f'DOI:"{doi}"', "format": "json", "resultType": "core", "pageSize": 5})
    hits = json.loads(fetch(f"{EPMC}/search?{q}"))["resultList"]["result"]
    hit = next((h for h in hits if (h.get("doi") or "").lower() == doi), None)
    if not hit:
        raise Refused(404, "Europe PMC doesn't know this DOI. The demo works with open-access papers in PubMed Central.")
    if not hit.get("pmcid") or hit.get("isOpenAccess") != "Y":
        raise Refused(422, "This paper's full text isn't openly available in PubMed Central, so it can't be rewritten here.")
    raw = fetch(f"{EPMC}/{hit['pmcid']}/fullTextXML")
    root = ET.fromstring(raw)
    if not license_ok(root):
        raise Refused(451, f"This paper ({hit.get('license') or 'licence unknown'}) isn't under a CC BY licence. "
                           "A full rewrite republishes the paper, which only CC BY clearly allows, so the demo refuses it.")
    sections = split_sections(root)
    if not sections:
        raise Refused(422, "This isn't laid out as a standard research paper (introduction, methods, results), "
                           "which is what the models were trained on.")
    sections = {k: v for k, v in sections.items() if k != "original_section_order"}
    whole = "\n\n".join(f"## {s}\n\n{sections[s]}" for s in OPENING + OTHER if sections.get(s))
    saved = G.OUT / f"{hit['pmcid']}.json"
    if saved.exists():    # a corpus paper: the very glossary its training or test data used
        entries = json.loads(saved.read_text())
    else:                 # a new paper: built the same way (offline copy: 99% identical explanations)
        with GLOSSARY_LOCK:   # the glossary module keeps one database connection
            entries = G.glossary({"paper_id": hit["pmcid"], **sections})
    out = {
        "doi": doi, "pmcid": hit["pmcid"], "title": hit.get("title"), "journal": ((hit.get("journalInfo") or {}).get("journal") or {}).get("title") or hit.get("journalTitle"),
        "year": hit.get("pubYear"), "authors": hit.get("authorString"), "license": hit.get("license") or "cc by",
        "url": f"https://doi.org/{doi}", "sections": sections,
        # what each student call reads (glossary/inputs.py): the whole paper for the opening,
        # the section itself for each section
        "glossary": {"opening": glossary_lines(entries, whole),
                     **{s: glossary_lines(entries, sections[s]) for s in OTHER if sections.get(s)}},
        "glossary_terms": len(entries),
        # the students' reply limits, as in training/eval_student.py
        "max_tokens": {"opening": 6000, **{s: min(16000, max(1500, 2 * len(ENC.encode(sections[s]))))
                                           for s in OTHER if sections.get(s)}},
        "fetched": time.strftime("%Y-%m-%d"),
    }
    cached.write_text(json.dumps(out, ensure_ascii=False))
    return out


# ---------------------------------------------------------------- can it be opened? (search results)
# The page opens a paper itself (Europe PMC XML, web/src/jats.ts). These are the same rules, in the
# corpus's Python (tools/check_jats.ts: identical on 500/500 papers), run here so a phone doesn't
# download every result's full text (~200 kB each) just to grey some out. A verdict never changes.
CHECKS_FILE = HERE / "checks.jsonl"
CHECKS: dict[str, str] = {}
CHECKS_LOCK = threading.Lock()
CHECKING: dict[str, Future] = {}           # pmcid → the fetch under way (two readers, one fetch)
CHECK_POOL = ThreadPoolExecutor(8)          # at most 8 full texts fetched from Europe PMC at once
CHECK_ASKS: dict[str, list[float]] = {}     # address → times of its uncached checks (last minute)
PMCID = re.compile(r"^PMC\d{1,10}$")
if CHECKS_FILE.exists():
    for _line in CHECKS_FILE.read_text().splitlines():
        try:
            _c = json.loads(_line)
            CHECKS[_c["pmcid"]] = _c["verdict"]
        except (ValueError, KeyError):
            pass


def verdict(pmcid: str) -> str | None:
    """"ok", or why the page would refuse it; None when Europe PMC didn't answer (ask again later)."""
    try:
        raw = fetch(f"{EPMC}/{pmcid}/fullTextXML", timeout=20)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return None
        v = "no full text"
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    else:
        try:
            root = ET.fromstring(raw)
            v = "licence" if not license_ok(root) else "layout" if not split_sections(root) else "ok"
        except ET.ParseError:
            v = "no full text"   # the page couldn't read it either
    with CHECKS_LOCK:
        CHECKS[pmcid] = v
        with CHECKS_FILE.open("a") as f:
            f.write(json.dumps({"pmcid": pmcid, "verdict": v}) + "\n")
    return v


def check_many(ids: list[str], who: str, seconds: float = 2.5) -> dict[str, str]:
    """The verdicts known now; if none is, waits (up to `seconds`) for the first to come, and returns every
    one ready by then. The page asks again for the rest, so each verdict reaches it as soon as it is known.
    "?": it couldn't be checked (Europe PMC didn't answer, or this address asked for too many)."""
    out, waiting = {}, {}
    now = time.time()
    with CHECKS_LOCK:
        recent = [t for t in CHECK_ASKS.get(who, []) if now - t < 60]
        for p in ids:
            if p in CHECKS:
                out[p] = CHECKS[p]
            elif p in CHECKING:
                waiting[p] = CHECKING[p]
            elif len(recent) < 90:   # a few searches a minute; past that, results show unchecked
                recent.append(now)
                fut = CHECKING[p] = CHECK_POOL.submit(verdict, p)
                fut.add_done_callback(lambda _f, p=p: CHECKING.pop(p, None))
                waiting[p] = fut
            else:
                out[p] = "?"
        CHECK_ASKS[who] = recent
    if waiting and not out:
        wait_all(waiting.values(), timeout=seconds, return_when=FIRST_COMPLETED)
    for p, fut in waiting.items():
        if fut.done():
            out[p] = (fut.exception() is None and fut.result()) or "?"
    return out


def store_folder(doi: str) -> Path:
    return STORE / hashlib.sha256(doi.lower().encode()).hexdigest()[:24]


def plain_title(t):
    """Europe PMC titles carry markup (<i>, <sup>, &amp;): plain text, no final period."""
    if not t:
        return t
    import html as _html
    return _html.unescape(re.sub(r"<[^>]+>", "", t)).strip().rstrip(".")


def paper_meta(doi: str) -> dict:
    """Title, journal and year of a paper we have seen (from the paper cache)."""
    f = CACHE / (hashlib.sha256(doi.lower().encode()).hexdigest()[:24] + ".json")
    try:
        d = json.loads(f.read_text())
        return {"title": d.get("title"), "journal": d.get("journal"), "year": d.get("year")}
    except (OSError, ValueError):
        return {}


def progress_of(j: dict) -> float:
    """How far a job is: text written against ~1.2 times the original's length (as the page estimates)."""
    sections = (j.get("paper") or {}).get("sections") or {}
    total = sum(max(200, len(v) * 1.2) for v in sections.values() if v) or 1
    done = set(j.get("done") or [])
    written = sum(max(200, len(v) * 1.2) if k in done else min(len(j["parts"].get(k) or ""), len(v) * 1.2 * 0.97)
                  for k, v in sections.items() if v)
    return round(min(0.99, written / total), 3)


LIBRARY = {"at": 0.0, "items": []}


# ---------------------------------------------------------------- support
# support.json (not published; example: support.example.json): what the GPU costs a day, the Stripe links
# (made by tools/stripe_setup.py), sponsors of the day by date, support counted by hand.
# Read every 5 minutes, only totals are shown:
#   Stripe, with the read-only key in stripe_read_key (Checkout Sessions and Subscriptions: read).
#     Supporters' names and emails stay here. A day's sponsor goes to sponsors.json, waiting for approval
#     (tools/sponsor_day.py): nothing a stranger types is shown on the site before that.
#   GitHub Sponsors, with this machine's GitHub login (gh); the token never leaves it. The logins of
#     sponsors who chose to be public are shown, as GitHub shows them.
SUPPORT_FILE = HERE / "support.json"
STRIPE_KEY_FILE = HERE / "stripe_read_key"
SPONSORS_FILE = HERE / "sponsors.json"
SUPPORTERS = {"github": None, "stripe": None}   # each: {day_count, day_dollars, monthly, recent?}
GITHUB_QUERY = """{ viewer {
  sponsorsActivities(first: 100, period: DAY, actions: [NEW_SPONSORSHIP]) { nodes { sponsorsTier { monthlyPriceInDollars } } }
  sponsorshipsAsMaintainer(first: 6, includePrivate: false, orderBy: {field: CREATED_AT, direction: DESC}) {
    nodes { sponsorEntity { ... on User { login } ... on Organization { login } } } }
  monthly: sponsorshipsAsMaintainer(first: 100, includePrivate: true, activeOnly: true) { nodes { isOneTimePayment } } } }"""


def read_github():
    import subprocess
    out = subprocess.run(["gh", "api", "graphql", "-f", f"query={GITHUB_QUERY}"], capture_output=True, text=True, timeout=60)
    v = json.loads(out.stdout)["data"]["viewer"]
    day = [n["sponsorsTier"]["monthlyPriceInDollars"] for n in v["sponsorsActivities"]["nodes"] if n.get("sponsorsTier")]
    return {"day_count": len(day), "day_dollars": sum(day),
            "monthly": sum(1 for n in v["monthly"]["nodes"] if not n["isOneTimePayment"]),
            "recent": [n["sponsorEntity"]["login"] for n in v["sponsorshipsAsMaintainer"]["nodes"] if n.get("sponsorEntity")]}


def stripe_get(key: str, path: str, params: dict) -> list[dict]:
    items, params = [], {"limit": 100, **params}
    while True:
        req = urllib.request.Request(f"https://api.stripe.com/v1/{path}?{urllib.parse.urlencode(params)}",
                                     headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.loads(r.read())
        items += page["data"]
        if not page.get("has_more") or len(items) > 2000:
            return items
        params["starting_after"] = page["data"][-1]["id"]


def read_stripe():
    if not STRIPE_KEY_FILE.exists():
        return None
    key = STRIPE_KEY_FILE.read_text().strip()
    paid = [x for x in stripe_get(key, "checkout/sessions", {"status": "complete", "created[gte]": int(time.time()) - 86400})
            if x.get("payment_status") in ("paid", "no_payment_required") and x.get("currency") == "usd"]
    try:
        monthly = len(stripe_get(key, "subscriptions", {"status": "active"}))
    except urllib.error.HTTPError:   # the key may not read subscriptions
        monthly = 0
    # a day's sponsor: kept for approval, never shown before it
    fields = lambda x: {f["key"]: (f.get("text") or {}).get("value") for f in x.get("custom_fields") or []}
    new = {x["id"]: {**fields(x), "dollars": x["amount_total"] / 100, "paid": x["created"], "status": "waiting"}
           for x in paid if fields(x).get("name")}
    if new:
        with SPONSORS_LOCK:
            known = json.loads(SPONSORS_FILE.read_text()) if SPONSORS_FILE.exists() else {}
            added = {k: v for k, v in new.items() if k not in known}
            if added:
                SPONSORS_FILE.write_text(json.dumps({**known, **added}, indent=1, ensure_ascii=False))
                print(f"day sponsor waiting for approval: {', '.join(v['name'] for v in added.values())} "
                      f"(app/tools/sponsor_day.py)", flush=True)
    return {"day_count": len(paid), "day_dollars": round(sum(x["amount_total"] for x in paid) / 100, 2), "monthly": monthly}


SPONSORS_LOCK = threading.Lock()


def read_supporters_forever():
    while True:
        for name, read in (("github", read_github), ("stripe", read_stripe)):
            try:
                SUPPORTERS[name] = read()
            except Exception as e:   # no gh, no network, a wrong key: the page shows the costs without the totals
                print(f"support from {name} not read: {e}", flush=True)
        time.sleep(300)


def support_state(private: bool = False) -> dict:
    try:
        cfg = json.loads(SUPPORT_FILE.read_text())
    except (OSError, ValueError):
        cfg = {}
    today = time.strftime("%Y-%m-%d")
    sponsor = (cfg.get("sponsors_by_day") or {}).get(today)
    by_hand = [g for g in cfg.get("support_by_hand", []) if time.time() - g.get("at", 0) < 86400]
    stripe = cfg.get("stripe") or {}
    if cfg.get("stripe_mode") != "live" and not private:
        stripe = {}   # test links: only on the private port (lambda and the tailnet), never through the public door
    out = {"daily_cost": cfg.get("daily_cost"), "gpu": cfg.get("gpu"), "stripe": stripe,
           "sponsor_of_the_day": sponsor}
    seen = [v for v in SUPPORTERS.values() if v]
    if seen:
        out.update(day_count=sum(v["day_count"] for v in seen) + len(by_hand),
                   day_dollars=round(sum(v["day_dollars"] for v in seen) + sum(g.get("dollars", 0) for g in by_hand), 2),
                   monthly_supporters=sum(v["monthly"] for v in seen),
                   recent=(SUPPORTERS["github"] or {}).get("recent", []))
    return out


def library() -> list[dict]:
    """Every paper with a saved rewrite, newest first (rebuilt at most every 30 s)."""
    if time.time() - LIBRARY["at"] < 30:
        return LIBRARY["items"]
    items = []
    for meta_file in STORE.glob("*/paper.json"):
        try:
            meta = json.loads(meta_file.read_text())
        except ValueError:
            continue
        models = {f.stem: f for f in meta_file.parent.glob("*.json") if f.stem != "paper"}
        if not models:
            continue
        doi = meta["doi"]
        info = {**paper_meta(doi), **{k: v for k, v in meta.items() if v}}
        plain = None
        for m in ("our-9b", "opus", "sonnet", *models):
            if m in models:
                try:
                    plain = json.loads(models[m].read_text())["parts"].get("title")
                except (ValueError, KeyError):
                    pass
                if plain:
                    break
        items.append({"doi": doi, "title": plain_title(info.get("title")) or doi, "plain_title": plain, "journal": info.get("journal"),
                      "year": info.get("year"), "models": sorted(models), "example": bool(meta.get("example")),
                      "added": max(f.stat().st_mtime for f in models.values())})
    items.sort(key=lambda x: -x["added"])
    LIBRARY.update(at=time.time(), items=items)
    return items


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(WEB), **k)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def send_json(self, status: int, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        if self.path.startswith("/api/") and not self.path.startswith("/api/worker/"):
            # the page may live elsewhere (GitHub Pages); nothing here uses cookies or logins
            self.send_header("Access-Control-Allow-Origin", "*")
        if not self.path.startswith("/api/"):
            # the page holds API keys: run only this site's own code, send only to the providers,
            # this server, or a model server the reader typed in
            self.send_header("Content-Security-Policy", "default-src \'self\'; script-src \'self\'; style-src \'self\'; img-src \'self\' data: https://pmc-oa-opendata.s3.amazonaws.com; connect-src \'self\' https:; base-uri \'none\'; form-action \'none\'; object-src \'none\'; frame-ancestors 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")   # re-check on each visit: page and script stay in step
        super().end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/paper":
            try:
                doi = normalize_doi(urllib.parse.parse_qs(u.query).get("doi", [""])[0])
                self.send_json(200, paper(doi))
            except Refused as r:
                self.send_json(r.status, {"error": str(r)})
            except (urllib.error.URLError, TimeoutError) as e:
                self.send_json(502, {"error": f"Europe PMC didn't answer ({type(e).__name__}). Try again in a minute."})
            except ET.ParseError:
                self.send_json(502, {"error": "Europe PMC returned a full text that couldn't be read."})
            return
        if u.path == "/api/examples":
            out = []
            for f in sorted(STORE.glob("*/paper.json")):
                meta = json.loads(f.read_text())
                if meta.get("example"):
                    plain = f.parent / "our-9b.json"
                    out.append({**meta, "plain_title": json.loads(plain.read_text())["parts"].get("title") if plain.exists() else None,
                                "models": sorted(x.stem for x in f.parent.glob("*.json") if x.stem != "paper")})
            self.send_json(200, out)
            return
        if u.path == "/api/rewrites":
            try:
                doi = normalize_doi(urllib.parse.parse_qs(u.query).get("doi", [""])[0])
            except Refused as r:
                self.send_json(r.status, {"error": str(r)})
                return
            f = store_folder(doi)
            self.send_json(200, {x.stem: json.loads(x.read_text()) for x in f.glob("*.json") if x.stem != "paper"}
                           if f.exists() else {})
            return
        if u.path == "/stats":
            # tailnet people only: Tailscale adds who is asking; requests from the internet (Funnel) don't carry it
            if not self.headers.get("Tailscale-User-Login") or self.headers.get("Tailscale-Funnel-Request"):
                self.send_response(404)
                self.end_headers()
                return
            body = stats.page(CACHE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/api/now":
            with JOBS_LOCK:
                live = [{"doi": j["doi"], "model": j["model"], "status": j["status"],
                         "title": plain_title((j.get("paper") or {}).get("title")), "plain_title": j["parts"].get("title"),
                         "progress": progress_of(j), "since": round(time.time() - j["created"])}
                        for j in sorted(JOBS.values(), key=lambda j: j["created"]) if j["status"] in ("running", "queued")]
                recent = list(RECENT_DONE)
            if len(recent) < 4:   # after a restart: the newest papers in the library
                seen = {r["doi"] for r in recent}
                recent += [{"doi": x["doi"], "title": x["title"], "plain_title": x["plain_title"]}
                           for x in library() if x["doi"] not in seen and not x["example"]][:4 - len(recent)]
            return self.send_json(200, {"live": live, "recent": recent,
                                        "worker_online": worker_online()})
        if u.path == "/api/library":
            return self.send_json(200, library())
        if u.path.startswith("/api/data/"):
            return self.data_proxy(u)
        if u.path == "/api/search":
            args = urllib.parse.parse_qs(u.query)
            q = (args.get("q") or [""])[0].strip()
            mode = (args.get("mode") or ["auto"])[0]
            year = (args.get("year_from") or [""])[0]
            if not 2 <= len(q) <= 300 or mode not in ("auto", "semantic", "keyword") or (year and not (year.isdigit() and 1800 <= int(year) <= 2100)):
                return self.send_json(400, {"error": "send q= (2 to 300 characters), optional mode=auto|semantic|keyword, year_from=YYYY"})
            now, who_ = time.time(), self.client()
            with SEARCHES_LOCK:
                recent = [t for t in SEARCHES.get(who_, []) if now - t < 60]
                SEARCHES[who_] = recent + [now]
                if len(SEARCHES) > 10000:   # forget idle addresses
                    for k in [k for k, v in SEARCHES.items() if now - v[-1] > 60]:
                        del SEARCHES[k]
            if len(recent) >= 30:
                return self.send_json(429, {"error": "Too many searches at once. Wait a minute."})
            try:
                return self.send_json(200, our_search(q, mode, int(year) if year else None))
            except Exception as e:   # noqa: BLE001  the page falls back to Europe PMC
                return self.send_json(502, {"error": f"search unavailable ({type(e).__name__})"})
        if u.path == "/api/check":
            raw = (urllib.parse.parse_qs(u.query).get("ids") or [""])[0]
            ids = list(dict.fromkeys(p.strip().upper() for p in raw.split(",") if p.strip()))[:25]
            if not ids or not all(PMCID.match(p) for p in ids):
                return self.send_json(400, {"error": "send ids=PMC…,PMC… (at most 25)"})
            return self.send_json(200, check_many(ids, self.client()))
        if u.path == "/api/support":
            return self.send_json(200, support_state(private=not isinstance(self, PublicHandler)))
        if u.path == "/api/status":
            with JOBS_LOCK:
                running = sum(1 for j in JOBS.values() if j["status"] == "running")
            self.send_json(200, {"worker_online": worker_online(), "models": models_online(),
                                 "queued": len(QUEUE), "running": running})
            return
        if u.path.startswith("/api/jobs/"):
            return self.job_view(u.path.rsplit("/", 1)[-1])
        if u.path == "/api/jobs":   # ?doi=&model=: a rewrite of this paper already in line or running
            q = urllib.parse.parse_qs(u.query)
            doi, model = (q.get("doi") or [""])[0].lower(), (q.get("model") or [""])[0]
            with JOBS_LOCK:
                j = next((j for j in JOBS.values() if j["doi"] == doi and j["model"] == model
                          and j["status"] in ("queued", "running")), None)
            return self.send_json(200, {"id": j["id"]} if j else {})
        super().do_GET()

    def body_json(self, limit=4_000_000):
        return json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), limit)) or b"{}")

    def data_proxy(self, u):
        """Read-only pass-through to the data API, for the routes in DATA_ROUTES only."""
        route = DATA_ROUTES.match(u.path)
        if not route or len(u.query) > 2000 or ".." in urllib.parse.unquote(u.path):
            return self.send_json(404, {"error": "not found"})
        tiles = route.group(1).startswith("map/tiles/")
        now, who_ = time.time(), ("t:" if tiles else "d:") + self.client()
        with SEARCHES_LOCK:
            recent = [t for t in SEARCHES.get(who_, []) if now - t < 60]
            SEARCHES[who_] = recent + [now]
        if len(recent) >= (1200 if tiles else 60):
            return self.send_json(429, {"error": "Too many requests. Wait a minute."})
        req = urllib.request.Request(SEARCH_API + u.path[len("/api/data"):] + (("?" + u.query) if u.query else ""))
        if tiles and self.headers.get("Range"):
            req.add_header("Range", self.headers["Range"])
        try:
            with urllib.request.urlopen(req, timeout=60 if route.group(1) == "walk" else 15) as r:
                status, body, ctype = r.status, r.read(), r.headers.get("Content-Type", "application/json")
                crange = r.headers.get("Content-Range")
                cenc = r.headers.get("Content-Encoding")   # map tiles: stored gzipped, sent as is
        except urllib.error.HTTPError as e:
            status, body, ctype, crange, cenc = e.code, e.read(), e.headers.get("Content-Type", "application/json"), None, None
        except OSError as e:
            return self.send_json(502, {"error": f"data service unavailable ({type(e).__name__})"})
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if crange:
            self.send_header("Content-Range", crange)
        if cenc == "gzip" and tiles:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if tiles and status < 300 else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def client(self) -> str:
        # behind tailscale serve / a tunnel, the reader's address is in X-Forwarded-For
        return (self.headers.get("X-Forwarded-For") or self.client_address[0]).split(",")[0].strip()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        # Chrome asks before a public page (sciencemadereadable.com) calls a private-network address,
        # which is what this machine's name resolves to on devices running Tailscale.
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/jobs":
            return self.new_job()
        if u.path == "/api/event":
            try:
                stats.record(self.body_json(4000), self.client(), self.headers.get("User-Agent", ""))
            except Exception:   # noqa: BLE001 - a bad count is dropped, never an error for the reader
                pass
            return self.send_json(204, {})
        if u.path.startswith("/api/worker/"):
            if self.headers.get("Authorization") != f"Bearer {WORKER_TOKEN}":
                return self.send_json(401, {"error": "worker token needed"})
            if u.path == "/api/worker/next":
                return self.next_job()
            return self.job_update(u.path.rsplit("/", 1)[-1])
        self.send_json(404, {"error": "not found"})

    def new_job(self):
        try:
            d = self.body_json(10_000)
            doi, model = normalize_doi(d["doi"]), str(d["model"])
        except Refused as r:
            return self.send_json(r.status, {"error": str(r)})
        except Exception:   # noqa: BLE001
            return self.send_json(400, {"error": "send {doi, model}"})
        if model not in models_online():
            return self.send_json(409, {"error": f"{model} isn't running on our GPU right now."})
        if (store_folder(doi) / f"{model}.json").exists():
            return self.send_json(200, {"saved": True})
        now, who_ = time.time(), self.client()
        with JOBS_LOCK:   # preparing a paper costs a fetch and a glossary: at most 10 asks a minute per address
            recent = [t for t in ASKS.get(who_, []) if now - t < 60]
            ASKS[who_] = recent + [now]
        if len(recent) >= 10:
            return self.send_json(429, {"error": "That's a lot of papers at once. Wait a minute and try again."})
        try:   # prepared here, on the home machine (glossary from the offline copy), and sent with the job
            prepared = paper(doi)
        except Refused as r:
            return self.send_json(r.status, {"error": str(r)})
        except Exception:   # noqa: BLE001
            return self.send_json(502, {"error": "We couldn't get this paper from Europe PMC. Try again in a minute."})
        who = self.client()
        with JOBS_LOCK:
            for j in JOBS.values():
                if j["doi"] == doi and j["model"] == model and j["status"] in ("queued", "running"):
                    return self.send_json(200, {"id": j["id"]})
            mine = sum(1 for j in JOBS.values() if j["who"] == who and j["status"] in ("queued", "running"))
            if mine >= MAX_PER_ADDRESS:
                return self.send_json(429, {"error": "You already have papers in the line. Wait for them to finish."})
            if len(QUEUE) >= MAX_QUEUE:
                return self.send_json(503, {"error": "The line is full right now. Try again later."})
            jid = __import__("secrets").token_urlsafe(9)
            JOBS[jid] = {"id": jid, "doi": doi, "model": model, "who": who, "status": "queued", "parts": {}, "paper": prepared,
                         "done": [], "created": time.time(), "updated": time.time()}
            QUEUE.append(jid)
        stats.record({"t": "job", "doi": doi, "model": model}, who, self.headers.get("User-Agent", ""))
        self.send_json(200, {"id": jid})

    def job_view(self, jid: str):
        with JOBS_LOCK:
            j = JOBS.get(jid)
            if not j:
                return self.send_json(404, {"error": "no such job (the server may have restarted)"})
            ahead = QUEUE.index(jid) if jid in QUEUE else 0
            running = sum(1 for x in JOBS.values() if x["status"] == "running")
            out = {k: j.get(k) for k in ("id", "doi", "model", "status", "parts", "done", "error", "seconds")}
        out["ahead"] = ahead
        out["running_others"] = running - (1 if j["status"] == "running" else 0)
        out["parallel"] = sum(w["parallel"] for w in online_workers() if j["model"] in w["models"]) or 1
        out["worker_online"] = worker_online()
        self.send_json(200, out)

    def next_job(self):
        d = self.body_json(10_000)
        name = str(d.get("name") or "worker")[:60]
        models = [str(m) for m in d.get("models") or []]
        WORKERS[name] = {"seen": time.time(), "models": models, "parallel": int(d.get("parallel") or 1)}
        if d.get("busy"):   # a heartbeat from a worker with no free slot: no job handed out
            return self.send_json(200, {})
        with JOBS_LOCK:
            now = time.time()
            for j in JOBS.values():   # a job whose worker went quiet goes back in line once
                if j["status"] == "running" and now - j["updated"] > 600:
                    j.update(status="queued", parts={}, done=[], updated=now)
                    QUEUE.insert(0, j["id"])
            for jid in [x for x, j in JOBS.items() if j["status"] in ("done", "failed") and now - j["updated"] > 3600]:
                del JOBS[jid]
            for jid in list(QUEUE):
                j = JOBS[jid]
                if j["model"] in models:   # only a job this worker's models can write
                    QUEUE.remove(jid)
                    j["worker"] = name
                    j.update(status="running", started=now, updated=now)
                    return self.send_json(200, {k: j[k] for k in ("id", "doi", "model", "paper")})
        self.send_json(200, {})

    def job_update(self, jid: str):
        d = self.body_json()
        with JOBS_LOCK:
            j = JOBS.get(jid)
            if not j:
                return self.send_json(404, {"error": "no such job"})
            j.update(parts=d.get("parts") or j["parts"], done=d.get("done") or j["done"], updated=time.time())
            if j.get("worker") in WORKERS:   # a worker busy with a paper is online too
                WORKERS[j["worker"]]["seen"] = time.time()
            if d.get("status") in ("done", "failed"):
                j.update(status=d["status"], error=d.get("error"), seconds=round(time.time() - j["started"]))
                stats.record({"t": f"job_{d['status']}", "doi": j["doi"], "model": j["model"], "s": j["seconds"],
                              "wait": round(j["started"] - j["created"])}, j["who"], "")
        if d.get("status") == "done":
            f = store_folder(j["doi"])
            f.mkdir(exist_ok=True)
            if not (f / "paper.json").exists() or "title" not in json.loads((f / "paper.json").read_text()):
                pp = j.get("paper") or {}
                (f / "paper.json").write_text(json.dumps({"doi": j["doi"], "title": pp.get("title"),
                                                          "journal": pp.get("journal"), "year": pp.get("year")},
                                                         ensure_ascii=False))
            LIBRARY["at"] = 0.0
            with JOBS_LOCK:
                RECENT_DONE.insert(0, {"doi": j["doi"], "model": j["model"], "seconds": j["seconds"], "at": time.time(),
                                       "title": plain_title((j.get("paper") or {}).get("title")), "plain_title": j["parts"].get("title")})
                del RECENT_DONE[12:]
            (f / f"{j['model']}.json").write_text(json.dumps({
                "doi": j["doi"], "model": j["model"], "parts": j["parts"], "source": "demo",
                "created": time.strftime("%Y-%m-%d"), "seconds": j["seconds"]}, ensure_ascii=False))
        self.send_json(200, {"ok": True})


SEARCH_API = os.environ.get("SRL_SEARCH_API", "http://127.0.0.1:8810")
DATA_ROUTES = re.compile(r"^/api/data/(search|walk|map/place|map/regions|map/counts|similar/[\w./:%-]{2,300}|works/[\w./:%-]{2,300}"
                         r"|map/tiles/[\w-]{1,40}/(?:index\.json|\d{1,2}/\d{1,6}/\d{1,6}\.(?:pts|ids)"
                         r"|v2/(?:map|labels)\.json|v2/[dpi]/\d{1,2}/\d{1,6}/\d{1,6}\.bin))$")
SEARCHES: dict[str, list[float]] = {}
SEARCHES_LOCK = threading.Lock()


def our_search(q: str, mode: str = "auto", year_from: int | None = None) -> dict:
    """The scholarsreadinglist search service, collection 'smr', mapped to the page's Hit shape."""
    params = {"q": q, "k": 15, "collection": "smr", "mode": mode, "abstracts": "false"}
    if year_from:
        params["year_from"] = year_from
    url = f"{SEARCH_API}/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=6) as r:
        d = json.load(r)
    hits = []
    for x in d["results"]:
        names = [a for a in (x.get("authors") or "").split(" | ") if a]
        authors = ", ".join(names[:6]) + (", et al." if (x.get("authors_count") or 0) > 6 else "")
        hits.append({"pmcid": x["pmcid"], "doi": x.get("doi"), "title": (x.get("title") or "").strip().rstrip("."),
                     "authors": authors, "journal": x.get("venue"), "year": str(x["year"]) if x.get("year") else None})
    return {"hits": [h for h in hits if h["pmcid"]], "mode": d.get("mode", "semantic"), "matches": d.get("matches")}


PUBLIC = {("GET", "/api/search"), ("GET", "/api/check"), ("GET", "/api/support"), ("GET", "/api/now"), ("GET", "/api/library"), ("GET", "/api/status"), ("GET", "/api/rewrites"), ("GET", "/api/examples"), ("GET", "/api/jobs"),
          ("POST", "/api/jobs"), ("POST", "/api/event"), ("POST", "/api/worker/next")}


class PublicHandler(Handler):
    """The door to the internet (Tailscale Funnel): the queue, saved rewrites, examples and counts.
    Not the paper service, not the dashboard, not the files (the page itself is on GitHub Pages)."""

    def allowed(self, method: str) -> bool:
        path = urllib.parse.urlparse(self.path).path
        return ((method, path) in PUBLIC or (method == "GET" and path.startswith(("/api/jobs/", "/api/data/")))
                or (method == "POST" and path.startswith("/api/worker/jobs/")))

    def do_GET(self):
        if not self.allowed("GET"):
            return self.send_json(404, {"error": "not found"})
        super().do_GET()

    def do_POST(self):
        if not self.allowed("POST"):
            return self.send_json(404, {"error": "not found"})
        super().do_POST()

    def do_HEAD(self):
        self.send_json(404, {"error": "not found"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8795)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--public-port", type=int, default=8799, help="the public door (0: none)")
    a = ap.parse_args()
    threading.Thread(target=save_jobs_forever, daemon=True).start()
    threading.Thread(target=read_supporters_forever, daemon=True).start()
    if a.public_port:
        public = ThreadingHTTPServer(("127.0.0.1", a.public_port), PublicHandler)
        threading.Thread(target=public.serve_forever, daemon=True).start()
        print(f"public door on http://127.0.0.1:{a.public_port} (queue, saved rewrites, examples, counts)", flush=True)
    print(f"demo on http://{a.host}:{a.port}  (glossary: {'offline copy' if G.OFFLINE else 'online Wikipedia'})", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
