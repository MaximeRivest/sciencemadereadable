"""The numbers of the live admin dashboard (server.py serves /admin on the tailnet only; admin/ holds its page).

Everything comes from the usage files stats.py writes (stats/events-*.jsonl for the site, stats/api-*.jsonl for
programs and AI apps), read incrementally: each refresh reads only the lines added since the last one.
Before Oct 17, 2026 the site did not record search words, topics or people pages, and api-*.jsonl holds only
the MCP tool calls and connections recovered from the server's log (rows with "log": 1, no words, no visitor).

Also used by tools/day_usage.py and tools/plot_usage.py (page_of, without_site_checks).
"""
from __future__ import annotations

import base64
import collections
import datetime as dt
import json
import queue
import statistics
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, quote

HERE = Path(__file__).resolve().parent
DIR = HERE / "stats"
SITE = "https://sciencemadereadable.com/"
QUEUE_EVENTS = {"job", "job_done", "job_failed"}          # from the queue, not a reader
ACTIONS = {"search", "topics", "pick", "more", "open", "read", "make", "copy", "support", "give", "author",
           "institution", "field_pay"}
RANGES = {"today", "yesterday", "7d", "30d"}
KEEP_DAYS = 65   # in memory: the longest range (30 days) and the one before it


# ------------------------------------------------------------------------------------------- reading the files
class Log:
    """The rows of stats/<prefix>-YYYY-MM.jsonl, read once and then only what was appended."""

    def __init__(self, prefix: str):
        self.prefix, self.rows, self.seen = prefix, [], {}   # seen: file name → bytes read
        self.lock = threading.Lock()

    def read(self) -> list[dict]:
        with self.lock:
            files = sorted(DIR.glob(f"{self.prefix}-????-??.jsonl"))
            if any(f.stat().st_size < self.seen.get(f.name, 0) for f in files) or \
                    set(self.seen) - {f.name for f in files}:
                self.rows, self.seen = [], {}          # a file was rewritten or deleted: start over
            for f in files:
                at = self.seen.get(f.name, 0)
                if f.stat().st_size == at:
                    continue
                with f.open("rb") as fh:
                    fh.seek(at)
                    chunk = fh.read()
                end = chunk.rfind(b"\n") + 1                 # a line being written stays for next time
                for line in chunk[:end].splitlines():
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(r.get("at"), (int, float)):
                        self.rows.append(r)
                self.seen[f.name] = at + end
            old = time.time() - KEEP_DAYS * 86400
            if self.rows and self.rows[0]["at"] < old:     # memory holds what the dashboard can show
                self.rows = [r for r in self.rows if r["at"] >= old]
            return self.rows


SITE_LOG, API_LOG = Log("events"), Log("api")


def without_site_checks(events: list[dict]) -> list[dict]:
    """The site check (tools/check_site.py, run at every publish) browses like a very busy visitor: a 1280-px
    window (counted as 1300) and 40+ pages in a day. Since Oct 17, 2026 it sends Global Privacy Control and is
    not counted at all; before, its visitor-days are recognised by that pattern and left out."""
    views, wide = collections.Counter(), set()
    for e in events:
        if e.get("t") == "view":
            key = (dt.date.fromtimestamp(e["at"]), e["v"])
            views[key] += 1
            if e.get("w") == 1300:
                wide.add(key)
    checks = {k for k in wide if views[k] >= 40}
    if not checks:
        return events
    return [e for e in events if (dt.date.fromtimestamp(e["at"]), e.get("v")) not in checks]


# ------------------------------------------------------------------------------------------- naming things
def page_of(e: dict) -> tuple[str, str]:
    """A page shown, as (what kind of page, what it was about): the address tells a search from a topic map."""
    view, u = e.get("view", "?"), parse_qs(e.get("u", ""))
    g = lambda k: (u.get(k) or [""])[0]   # noqa: E731
    if view == "results":
        if g("topics"):
            return ("topic map (whole field)" if g("scope") == "field" else "topic map"), g("topics")
        for k, name in (("author", "author page"), ("institution", "institution page"), ("people", "people search")):
            if g(k):
                return name, g(k)
        return "search results", g("q")
    if view == "explore":
        if g("work"):
            return "citation walk on the map", g("work")
        about = g("topics") or g("author") or g("institution")
        return ("a topic or person on the map", about) if about else ("map of science", "")
    names = {"api": "API / agents page", "work": "a paper's page", "home": "home", "library": "library of rewrites",
             "reader": "reading a rewrite", "fields": "field maps", "reading": "PDF reading desk",
             "account": "account", "support": "support page"}
    return names.get(view, view), g("work") or g("paper") or g("doi")


def site_link(e: dict) -> str:
    if e.get("t") == "search" and e.get("q"):
        return SITE + "?q=" + quote(e["q"])
    if e.get("t") == "topics" and e.get("q"):
        return SITE + "?topics=" + quote(e["q"]) + ("&scope=field" if e.get("field") else "")
    if e.get("t") in ("author", "institution") and e.get("id"):
        return SITE + f"?{e['t']}=" + quote(e["id"])
    if e.get("t") in ("open", "read", "make") and e.get("doi"):
        return SITE + "?work=" + quote(e["doi"])
    if e.get("u"):
        return SITE + "?" + e["u"]
    return ""


def device(w) -> str:
    return "" if not w else "phone" if w < 700 else "tablet" if w < 1100 else "computer"


def site_sentence(e: dict) -> tuple[str, str]:
    """What a site event says, in words, and a mark: 'bad' for something that went wrong, 'good' for a success."""
    t = e["t"]
    if t == "view":
        kind, about = page_of(e)
        ref = f" (arrived from {e['ref']})" if e.get("ref") else ""
        return f"{kind}{': ' + about if about else ''}{ref}", ""
    if t == "hide":
        busy = f", while {e['busy']} was still under way" if e.get("busy") else ""
        return f"left or hid the page after {fmt_s(e.get('s', 0))}{busy}", "bad" if busy else ""
    if t == "search":
        if "q" not in e:
            return f"searched ({e.get('n', 0):.0f} found)", ""
        n = int(e.get("n", 0))
        return f"searched “{e['q']}”: {n} found" + (f" in {e['ms'] / 1000:.1f} s" if e.get("ms") else ""), \
            "bad" if not n else ""
    if t == "topics":
        what = f"topic map “{e.get('q', '?')}”" + (" (whole field)" if e.get("field") else "")
        if e.get("pending"):
            return what + ": being made, waiting", ""
        return what + (f": {e.get('groups', 0):.0f} groups" if e.get("groups") else ""), ""
    if t == "pick":
        return f"picked box {e.get('box', '?')} in the map", ""
    if t == "more":
        return f"read further in the “{e.get('list', '?')}” list (from {e.get('n', 0):.0f})", ""
    if t in ("author", "institution"):
        return f"{t} page: {e.get('name') or e.get('id') or '?'}" + \
            (f" ({e['works']:,.0f} studies)" if e.get("works") else ""), ""
    if t == "open":
        if e.get("err"):
            return f"couldn't open a paper ({e['err']})", "bad"
        return f"opened “{e.get('title') or (title_of(e['doi']) if e.get('doi') else '?')}”", ""
    if t == "read":
        return f"read a saved rewrite ({e.get('doi', '?')})", "good"
    if t == "make":
        return f"asked for a new rewrite ({e.get('doi', '?')})", "good"
    if t in ("done", "job_done"):
        return f"a rewrite was finished in {fmt_s(e.get('s', 0))}", "good"
    if t in ("fail", "job_failed"):
        return "a rewrite failed", "bad"
    if t == "job":
        return "a rewrite joined the queue", ""
    if t == "error":
        return f"was shown an error ({e.get('where', '?')}{', ' + str(int(e['status'])) if e.get('status') else ''})", "bad"
    if t == "copy":
        return f"copied the agent setup ({e.get('src', '?')})", "good"
    if t == "support":
        return "opened the support window", ""
    if t == "give":
        return f"clicked to support ({e.get('src', '')})", "good"
    if t in ("supported", "thanks_shown"):
        return "came back after supporting", "good"
    if t == "field_pay":
        return "clicked to pay for a whole-field map", "good"
    if t == "field_paid":
        return "paid for a whole-field map", "good"
    return t, ""


def api_sentence(r: dict) -> str:
    what = {"mcp": "MCP", "api": "API", "ask": "Ask"}.get(r.get("k"), r.get("k", "?"))
    q = f" “{r['q']}”" if r.get("q") else ""
    app = f" · {r['app']}" if r.get("app") else ""
    return f"{what} {r.get('r', '?')}{q}{app}" + (" (refused: rate limit)" if r.get("limited") else "")


def fmt_s(s) -> str:
    s = int(s or 0)
    return f"{s} s" if s < 90 else f"{s // 60} min" if s < 5400 else f"{s / 3600:.1f} h"


# ------------------------------------------------------------------------------------------- paper titles
SEARCH_API = "http://127.0.0.1:8810"   # set by server.py
_titles: dict[str, str | None] = {}     # DOI or W-number → title ("" when unknown); None: being asked
_wanted: queue.Queue = queue.Queue()


def _title_worker():
    while True:
        ref = _wanted.get()
        try:
            url = f"{SEARCH_API}/works/{quote(ref, safe='/')}"
            with urllib.request.urlopen(url, timeout=5) as r:
                _titles[ref] = json.load(r).get("title") or ""
        except Exception:   # noqa: BLE001
            _titles[ref] = ""


threading.Thread(target=_title_worker, daemon=True).start()


def title_of(ref: str) -> str:
    """A paper's title for events recorded before the page sent it (asked in the background, kept)."""
    if ref not in _titles and len(_titles) < 50000:
        _titles[ref] = None
        _wanted.put(ref)
    return _titles.get(ref) or ref


# ------------------------------------------------------------------------------------------- money and accounts
_stripe = {"at": 0.0, "rows": None, "busy": False}


def _stripe_refresh():
    key_file = HERE / "stripe_read_key"
    try:
        auth = base64.b64encode((key_file.read_text().strip() + ":").encode()).decode()
        since = int(time.time()) - 32 * 86400
        rows, seen = [], set()
        for ep in ("checkout/sessions", "payment_intents"):
            req = urllib.request.Request(f"https://api.stripe.com/v1/{ep}?limit=100&created[gte]={since}",
                                         headers={"Authorization": "Basic " + auth})
            for x in json.load(urllib.request.urlopen(req, timeout=15))["data"]:
                paid = x.get("payment_status") == "paid" if ep.startswith("checkout") else x.get("status") == "succeeded"
                pi = x.get("payment_intent") if ep.startswith("checkout") else x["id"]
                if pi and pi in seen:
                    continue
                if pi:
                    seen.add(pi)
                rows.append({"at": x["created"], "paid": paid, "cents": x.get("amount_total") or x.get("amount") or 0,
                             "started": ep.startswith("checkout")})
        _stripe["rows"] = rows
    except Exception:   # noqa: BLE001  no key, no network: the dashboard says Stripe can't be read
        _stripe["rows"] = _stripe["rows"] if _stripe["rows"] is not None else []
        _stripe["error"] = True
    finally:
        _stripe["at"], _stripe["busy"] = time.time(), False


def stripe_rows():
    """Payments of the last 32 days, refreshed in the background every 10 minutes (None until the first read)."""
    if time.time() - _stripe["at"] > 600 and not _stripe["busy"]:
        _stripe["busy"] = True
        threading.Thread(target=_stripe_refresh, daemon=True).start()
    return _stripe["rows"]


def accounts_summary(start: float, end: float) -> dict | None:
    try:
        import accounts as A
        iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat()   # noqa: E731
        return {
            "total": A.q1("SELECT count(*) n FROM users")["n"],
            "new": A.q1("SELECT count(*) n FROM users WHERE created >= ? AND created < ?", (iso(start), iso(end)))["n"],
            "paid": A.q1("SELECT count(*) n FROM subscriptions WHERE status IN ('active', 'trialing')")["n"],
            "granted": A.q1("SELECT count(*) n FROM users WHERE grant_until >= ?", (dt.date.today().isoformat(),))["n"],
            "key_requests": A.q1("SELECT coalesce(sum(requests), 0) n FROM usage WHERE day >= ? AND day <= ?",
                                 (dt.date.fromtimestamp(start).isoformat(),
                                  dt.date.fromtimestamp(end - 1).isoformat()))["n"],
        }
    except Exception:   # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------- the dashboard
def period(rng: str, now: float) -> tuple[float, float, float, float, str]:
    """(start, end, previous start, previous end, how the previous one is called). Today is compared with
    yesterday up to the same time of day; the others with the period just before."""
    midnight = dt.datetime.combine(dt.date.fromtimestamp(now), dt.time()).timestamp()
    day = 86400
    if rng == "today":
        return midnight, now, midnight - day, now - day, "yesterday at this time"
    if rng == "yesterday":
        return midnight - day, midnight, midnight - 2 * day, midnight - day, "the day before"
    n = 7 if rng == "7d" else 30
    return midnight - (n - 1) * day, now, midnight - (2 * n - 1) * day, now - n * day, f"the {n} days before"


def visits_of(events: list[dict]) -> dict[tuple, list[dict]]:
    out = collections.defaultdict(list)
    for e in events:
        if e["t"] not in QUEUE_EVENTS:
            out[(dt.date.fromtimestamp(e["at"]).isoformat(), e["v"])].append(e)
    for evs in out.values():
        evs.sort(key=lambda e: e["at"])
    return out


def counts(events: list[dict], api: list[dict]) -> dict:
    vis = visits_of(events)
    reads = {(e["v"], e.get("doi"), e.get("model")) for e in events if e["t"] == "read"}   # re-sent every 30 s
    x = sum(1 for evs in vis.values() if any(e.get("ref") == "t.co" for e in evs))
    calls = [r for r in api if r.get("r") != "connect"]
    return {
        "visitors": len(vis), "from_x": x,
        "searches": sum(1 for e in events if e["t"] == "search"),
        "empty": sum(1 for e in events if e["t"] == "search" and not e.get("n")),
        "topics": sum(1 for e in events if e["t"] == "topics" and not e.get("pending")),
        "people": sum(1 for e in events if e["t"] in ("author", "institution")),
        "opened": sum(1 for e in events if e["t"] == "open" and not e.get("err")),
        "reads": len(reads),
        "made": sum(1 for e in events if e["t"] in ("job_done", "done")),
        "agent_calls": len(calls),
        "mcp_calls": sum(1 for r in calls if r["k"] == "mcp"),
        "api_calls": sum(1 for r in calls if r["k"] == "api" and r["r"] != "ask"),
        "ask": sum(1 for r in calls if r["r"] == "ask"),
        "mcp_connects": sum(1 for r in api if r.get("r") == "connect"),
        "agents": len({r["v"] for r in calls if r.get("v")}),
        "support": sum(1 for e in events if e["t"] == "support"),
        "give": sum(1 for e in events if e["t"] in ("give", "field_pay")),
        "copies": sum(1 for e in events if e["t"] == "copy"),
        "errors": sum(1 for e in events if e["t"] == "error"),
    }


def per_visitor(rows, n=20, keep_key=False):
    """[(text, visitors, link)] most asked first: the same visitor asking again counts once.
    keep_key: a fourth column, the text itself (to recognise rows still named by an id)."""
    who, links = collections.defaultdict(set), {}
    for v, text, link in rows:
        if text:
            who[text].add(v)
            links.setdefault(text, link)
    out = sorted(((t, len(vs), links[t]) for t, vs in who.items()), key=lambda r: (-r[1], r[0].lower()))
    return [[*r, r[0]] if keep_key else list(r) for r in out[:n]]


def top(counter: collections.Counter, n=12):
    return [[k, c, ""] for k, c in counter.most_common(n)]


def data(rng: str, health: dict) -> dict:
    t0 = time.time()
    now = time.time()
    rng = rng if rng in RANGES else "today"
    start, end, pstart, pend, pname = period(rng, now)
    site = without_site_checks([e for e in SITE_LOG.read() if e["at"] >= pstart - 86400])
    api = [r for r in API_LOG.read() if r["at"] >= pstart]
    ev = [e for e in site if start <= e["at"] < end]
    prev = [e for e in site if pstart <= e["at"] < pend]
    ap = [r for r in api if start <= r["at"] < end]
    app_prev = [r for r in api if pstart <= r["at"] < pend]

    # charts: hours for one day, days otherwise
    hourly = rng in ("today", "yesterday")
    if hourly:
        labels = [f"{h:02d}" for h in range(24)]
        slot = lambda t: dt.datetime.fromtimestamp(t).hour   # noqa: E731
    else:
        first = dt.date.fromtimestamp(start)
        days = [first + dt.timedelta(d) for d in range((dt.date.fromtimestamp(end - 1) - first).days + 1)]
        labels = [d.strftime("%a %-d") for d in days]
        slot = lambda t: (dt.date.fromtimestamp(t) - first).days   # noqa: E731
    n = len(labels)
    series = {k: [0] * n for k in ("x", "other", "searches", "opened", "topics", "people", "reads",
                                   "mcp", "api", "ask")}
    vis = visits_of(ev)
    seen_slot = set()
    for (day, v), evs in vis.items():
        x = any(e.get("ref") == "t.co" for e in evs)
        for e in evs:
            s = slot(e["at"])
            if (s, day, v) not in seen_slot:
                seen_slot.add((s, day, v))
                series["x" if x else "other"][s] += 1
    seen_read = set()
    for e in ev:
        s, t = slot(e["at"]), e["t"]
        if t == "search":
            series["searches"][s] += 1
        elif t == "open" and not e.get("err"):
            series["opened"][s] += 1
        elif t == "topics" and not e.get("pending"):
            series["topics"][s] += 1
        elif t in ("author", "institution"):
            series["people"][s] += 1
        elif t == "read" and (e["v"], e.get("doi"), e.get("model")) not in seen_read:
            seen_read.add((e["v"], e.get("doi"), e.get("model")))
            series["reads"][s] += 1
    for r in ap:
        if r.get("r") == "connect":
            continue
        k = "ask" if r.get("r") == "ask" else r["k"]
        if k in series:
            series[k][slot(r["at"])] += 1

    # what people looked for
    searches = [e for e in ev if e["t"] == "search" and e.get("q")]
    looked = {
        "searches": per_visitor((e["v"], e["q"].strip().lower(), site_link(e)) for e in searches),
        "empty": per_visitor((e["v"], e["q"].strip().lower(), site_link(e)) for e in searches if not e.get("n")),
        "topics": per_visitor((e["v"], e["q"].strip().lower() + (" (whole field)" if e.get("field") else ""),
                               site_link(e)) for e in ev if e["t"] == "topics" and e.get("q") and not e.get("pending")),
        "people": per_visitor(
            [(e["v"], e.get("name") or e.get("id"), site_link(e)) for e in ev
             if e["t"] in ("author", "institution") and (e.get("name") or e.get("id"))] +
            [(e["v"], page_of(e)[1], site_link(e)) for e in ev if e["t"] == "view"
             and page_of(e)[0] in ("author page", "institution page") and not any(
                x["t"] in ("author", "institution") for x in vis[(dt.date.fromtimestamp(e["at"]).isoformat(), e["v"])])]),
        "papers": [[r[0] if r[0] != r[3] else title_of(r[3]), r[1], r[2]] for r in per_visitor(
            ((e["v"], e.get("title") or e["doi"], site_link(e)) for e in ev
             if e["t"] == "open" and not e.get("err") and e.get("doi")), 15, keep_key=True)],
        "cant_open": top(collections.Counter(e["err"] for e in ev if e["t"] == "open" and e.get("err"))),
    }
    ms = sorted(e["ms"] for e in ev if e["t"] == "search" and e.get("ms"))
    tms = sorted(e["ms"] for e in ev if e["t"] == "topics" and e.get("ms") and not e.get("cached"))
    q = lambda xs, p: round(xs[min(len(xs) - 1, int(p * len(xs)))] / 1000, 1) if xs else None   # noqa: E731

    # how visits went
    entries, exits, paths, after_search = (collections.Counter() for _ in range(4))
    refs, devices, left_busy = collections.Counter(), collections.Counter(), collections.Counter()
    lengths, bounced, visit_list = [], 0, []
    for (day, v), evs in vis.items():
        steps = []
        for e in evs:
            if e["t"] == "view":
                name = page_of(e)[0]
                if not steps or steps[-1] != name:
                    steps.append(name)
        first_view = next((e for e in evs if e["t"] == "view"), {})
        if first_view.get("ref"):
            refs[first_view["ref"]] += 1
        if device(first_view.get("w")):
            devices[device(first_view.get("w"))] += 1
        for e in evs:
            if e["t"] == "hide" and e.get("busy"):
                left_busy[e["busy"]] += 1
        if steps:
            entries[steps[0]] += 1
            exits[steps[-1]] += 1
            paths[" → ".join(steps[:5]) + (" → …" if len(steps) > 5 else "")] += 1
            for a, b in zip(steps, steps[1:]):
                if a == "search results":
                    after_search[b] += 1
        did = any(e["t"] in ACTIONS for e in evs)
        if len(steps) <= 1 and not did:
            bounced += 1
        if len(evs) > 1:
            lengths.append(evs[-1]["at"] - evs[0]["at"])
        visit_list.append((evs[-1]["at"], day, v, evs, steps, first_view))
    visit_list.sort(key=lambda r: -r[0])
    visits = [{
        "key": f"{day}/{v}", "v": v, "start": evs[0]["at"], "last": last, "pages": len(steps),
        "length": evs[-1]["at"] - evs[0]["at"], "ref": fv.get("ref", ""), "device": device(fv.get("w")),
        "summary": " → ".join(steps[:4]) + (" → …" if len(steps) > 4 else ""),
        "did": sorted({e["t"] for e in evs if e["t"] in ACTIONS}),
        "steps": [[e["at"], *site_sentence(e), site_link(e)] for e in evs[-120:]],
    } for last, day, v, evs, steps, fv in visit_list[:80]]

    # programs and AI apps
    calls = [r for r in ap if r.get("r") != "connect"]
    agents = {
        "tools": top(collections.Counter(("MCP " if r["k"] == "mcp" else "API ") + r["r"] if r["r"] != "ask" else "Ask"
                                         for r in calls), 15),
        "apps": top(collections.Counter(r.get("app") or "(unnamed)" for r in ap if r.get("r") == "connect"), 15),
        "programs": top(collections.Counter(r.get("app") or "(unnamed)" for r in calls if r["k"] != "mcp"), 10),
        "queries": per_visitor((r["v"] or "log", f"{r['r']}: {r['q'].strip().lower()}", "") for r in calls
                               if r.get("q") and r["r"] not in ("works", "get_papers", "similar", "find_similar_papers")),
        "limited": sum(1 for r in ap if r.get("limited")),
        "busiest": top(collections.Counter(r["v"] for r in calls if r.get("v")), 5),
    }

    # right now (whatever the range)
    live_site = [e for e in site if e["at"] >= now - 3600]
    active = []
    for (day, v), evs in visits_of([e for e in live_site if e["at"] >= now - 1800]).items():
        if evs[-1]["at"] < now - 300 or evs[-1]["t"] == "hide" and evs[-1]["at"] < now - 60:
            continue
        cur = next((e for e in reversed(evs) if e["t"] == "view"), None)
        active.append({"v": v, "key": f"{day}/{v}", "since": evs[0]["at"], "last": evs[-1]["at"],
                       "page": " · ".join(x for x in page_of(cur) if x) if cur else "?",
                       "link": site_link(cur) if cur else "", "pages": sum(1 for e in evs if e["t"] == "view"),
                       "doing": site_sentence(evs[-1])[0]})
    active.sort(key=lambda a: -a["last"])
    feed = [[e["at"], "site", e["v"], *site_sentence(e), site_link(e)]
            for e in live_site[-150:] if e["t"] not in ("hide",) or e.get("busy")]
    live_api = [r for r in api if r["at"] >= now - 3600 and r.get("r") != "connect"]
    feed += [[r["at"], "agent", r.get("v", ""), api_sentence(r), "bad" if r.get("limited") else "", ""]
             for r in live_api[-100:]]
    feed.sort(key=lambda f: -f[0])

    pay = stripe_rows()
    money = None if pay is None else {
        "paid": sum(1 for p in pay if p["paid"] and start <= p["at"] < end),
        "dollars": sum(p["cents"] for p in pay if p["paid"] and start <= p["at"] < end) / 100,
        "started": sum(1 for p in pay if p["started"] and start <= p["at"] < end),
        "error": bool(_stripe.get("error")),
    }

    return {
        "range": rng, "now": now, "start": start, "end": end, "previous": pname, "hourly": hourly,
        "first_detail": 1792266000,   # Oct 17, 2026, ~15:40: searches, topics, people, flows recorded from then
        "now_counts": counts(ev, ap), "prev_counts": counts(prev, app_prev),
        "chart": {"labels": labels, **series},
        "looked": looked,
        "speed": {"search_median": q(ms, .5), "search_p90": q(ms, .9), "map_median": q(tms, .5), "map_p90": q(tms, .9)},
        "flow": {"visits": len(vis), "bounced": bounced,
                 "median_min": round(statistics.median(lengths) / 60, 1) if lengths else None,
                 "entries": top(entries), "exits": top(exits), "paths": top(paths, 15), "after_search": top(after_search),
                 "left_busy": top(left_busy), "errors": top(collections.Counter(
                     e.get("where", "?") + (f" ({int(e['status'])})" if e.get("status") else "")
                     for e in ev if e["t"] == "error")),
                 "refs": top(refs), "devices": top(devices),
                 "copies": top(collections.Counter(e.get("src", "?") for e in ev if e["t"] == "copy"))},
        "agents": agents,
        "live": {"active": active, "feed": feed[:120],
                 "agents_10min": len([r for r in live_api if r["at"] >= now - 600]),
                 "agent_apps_10min": top(collections.Counter(r.get("app") or "(unnamed)" for r in live_api
                                                             if r["at"] >= now - 600), 6)},
        "visits": visits,
        "money": money, "accounts": accounts_summary(start, end),
        "health": health,
        "took_ms": round((time.time() - t0) * 1000),
    }
