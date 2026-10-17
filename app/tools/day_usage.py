"""One day of sciencemadereadable.com usage, as a page of charts.

    python3 app/tools/day_usage.py [YYYY-MM-DD]      (default: today, server time)

Writes app/stats/day-YYYY-MM-DD.html (app/stats/ is private and not committed).

Three sources:
  * app/stats/events-*.jsonl: what people did on the website (counted by app/stats.py): pages, searches,
    topic maps, people pages, where they gave up. Searches and topics are kept from Oct 17, 2026 on.
  * app/stats/api-*.jsonl: what programs and AI apps asked the API and the MCP server (from Oct 17, 2026).
  * the smr-server journal: requests to the Ask address, API calls and MCP methods per app, for every day.
    The journal has no address, so there "requests" are not "people".
"""
from __future__ import annotations

import collections
import datetime as dt
import html as H
import json
import statistics
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs

STATS = Path(__file__).resolve().parents[1] / "stats"
DAY = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date.today()
OUT = STATS / f"day-{DAY}.html"
HOURS = list(range(24))

# 1. Website events of the day ------------------------------------------------------------------
events = []
for f in sorted(STATS.glob("events-*.jsonl")):
    for line in f.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
def without_site_checks(events: list[dict]) -> list[dict]:
    """The site check (tools/check_site.py, run at every publish) browses like a very busy visitor: a 1280-px
    window (counted as 1300) and 40+ pages in a day. Since Oct 17, 2026 it sends Global Privacy Control and is
    not counted at all; before, its visitor-days are recognised by that pattern and left out here."""
    views, wide = collections.Counter(), set()
    for e in events:
        key = (dt.date.fromtimestamp(e["at"]), e["v"])
        if e["t"] == "view":
            views[key] += 1
            if e.get("w") == 1300:
                wide.add(key)
    checks = {k for k in wide if views[k] >= 40}
    return [e for e in events if (dt.date.fromtimestamp(e["at"]), e["v"]) not in checks]


events = without_site_checks(events)
for e in events:
    e["when"] = dt.datetime.fromtimestamp(e["at"])
today = [e for e in events if e["when"].date() == DAY]
if not today:
    sys.exit(f"no events on {DAY}")
last_event = max(e["when"] for e in today)

from_x = {e["v"] for e in today if e["t"] == "view" and e.get("ref") == "t.co"}
people = collections.defaultdict(set)         # hour → visitor codes
searches = collections.Counter()              # hour → searches
opened = collections.Counter()                # hour → paper pages opened
reads = collections.Counter()                 # hour → rewrites read (once per visitor and paper)
made = []                                     # finished rewrites: (time, seconds)
seen_read = set()
page_people = collections.defaultdict(set)    # page → visitor codes
did = collections.defaultdict(set)            # step → visitor codes
width = {}
copies, supports = collections.Counter(), 0
search_kind = collections.Counter()
for e in today:
    h, t, v = e["when"].hour, e["t"], e["v"]
    if t not in ("job", "job_done", "job_failed"):   # those come from the queue, not a reader
        people[h].add(v)
    if t == "view":
        page_people[e.get("view", "?")].add(v)
        if "w" in e:
            width[v] = max(width.get(v, 0), e["w"])
    elif t == "search":
        searches[h] += 1
        search_kind[e.get("src", "?")] += 1
        did["searched"].add(v)
    elif t == "open" and not e.get("err"):
        opened[h] += 1
        did["opened"].add(v)
    elif t == "read":
        did["read"].add(v)
        if (v, e.get("doi"), e.get("model")) not in seen_read:   # the page re-sends "read" every 30 s
            seen_read.add((v, e.get("doi"), e.get("model")))
            reads[h] += 1
    elif t == "make":
        did["made"].add(v)
    elif t == "job_done":
        made.append((e["when"].strftime("%H:%M"), e.get("s", 0)))
    elif t == "copy":
        copies[e.get("src", "?")] += 1
    elif t == "support":
        supports += 1

visitors = set().union(*people.values())

# a few heavy users (often our own tests) can make most of the clicks
by_person = collections.Counter(e["v"] for e in today if e["t"] == "search")
top4 = sum(n for _, n in by_person.most_common(4))
first_page = {}
for e in today:
    if e["t"] == "view":
        first_page.setdefault(e["v"], e.get("view", "?"))
landed = collections.Counter(first_page.values())
devices = collections.Counter("phone" if w < 700 else "tablet" if w < 1100 else "computer" for w in width.values())

# earlier days, for scale
daily = collections.defaultdict(set)
daily_x = collections.defaultdict(set)
x_codes = {(e["when"].date(), e["v"]) for e in events if e["t"] == "view" and e.get("ref") == "t.co"}
for e in events:
    if e["t"] not in ("job", "job_done", "job_failed") and e["when"].date() <= DAY:
        d = e["when"].date()
        (daily_x if (d, e["v"]) in x_codes else daily)[d].add(e["v"])
days = sorted(set(daily) | set(daily_x))

# 1b. What people looked for, and how their visits went (events with q and u: from Oct 17, 2026) ---------------
def page_of(e) -> str:
    """A page shown, named by what it is: the address says if 'results' was a search, a topic map or a person."""
    view, u = e.get("view", "?"), parse_qs(e.get("u", ""))
    if view == "results":
        if "topics" in u:
            return "topic map (whole field)" if u.get("scope") == ["field"] else "topic map"
        for k, name in (("author", "author page"), ("institution", "institution page"), ("people", "people search")):
            if k in u:
                return name
        return "search results"
    if view == "explore":
        if "work" in u:
            return "citation walk on the map"
        if "topics" in u or "author" in u or "institution" in u:
            return "a topic or person on the map"
        return "map of science"
    return {"api": "API / agents page", "work": "a paper's page", "home": "home", "library": "library of rewrites",
            "reader": "reading a rewrite", "fields": "field maps", "reading": "PDF reading desk", "account": "account",
            "support": "support page"}.get(view, view)


ACTIONS = {"search", "topics", "pick", "more", "open", "read", "make", "copy", "support", "give", "author", "institution"}
visits = collections.defaultdict(list)
for e in today:
    if e["t"] not in ("job", "job_done", "job_failed"):
        visits[e["v"]].append(e)
paths, lengths, bounced, exits = collections.Counter(), [], 0, collections.Counter()
after_search = collections.Counter()
for v, evs in visits.items():
    steps = []
    for e in evs:
        if e["t"] == "view":
            name = page_of(e)
            if not steps or steps[-1] != name:
                steps.append(name)
    if not steps:
        continue
    exits[steps[-1]] += 1
    paths[" → ".join(steps[:4]) + (" → …" if len(steps) > 4 else "")] += 1
    for a, b in zip(steps, steps[1:]):
        if a == "search results":
            after_search[b] += 1
    hides = [e.get("s", 0) for e in evs if e["t"] == "hide"]
    span = evs[-1]["at"] - evs[0]["at"] + (hides[-1] if evs[-1]["t"] != "hide" and hides else 0)
    lengths.append(max(span, max(hides, default=0)))
    if len(steps) == 1 and not any(e["t"] in ACTIONS for e in evs) and max(hides, default=0) < 15:
        bounced += 1

def per_visitor(rows):
    """(text, visitors) most asked first: one visitor asking the same thing again counts once."""
    seen = collections.defaultdict(set)
    for v, text in rows:
        seen[text].add(v)
    return sorted(((t, len(vs)) for t, vs in seen.items()), key=lambda kv: (-kv[1], kv[0]))

site_searches = [e for e in today if e["t"] == "search" and e.get("q")]
top_searches = per_visitor((e["v"], e["q"].strip().lower()) for e in site_searches)
empty_searches = per_visitor((e["v"], e["q"].strip().lower()) for e in site_searches if not e.get("n"))
search_ms = sorted(e["ms"] for e in today if e["t"] == "search" and e.get("ms"))
topic_ev = [e for e in today if e["t"] == "topics" and e.get("q")]
top_topics = per_visitor((e["v"], e["q"].strip().lower() + (" (whole field)" if e.get("field") else ""))
                         for e in topic_ev if not e.get("pending"))
waited = {(e["v"], e["q"]) for e in topic_ev if e.get("pending")}
got = {(e["v"], e["q"]) for e in topic_ev if not e.get("pending")}
gave_up_field = len(waited - got)
people_pages = per_visitor((e["v"], f"{k}={parse_qs(e.get('u', '')).get(k, ['?'])[0]}")
                           for e in today if e["t"] == "view" and e.get("view") == "results"
                           for k in ("author", "institution") if k in parse_qs(e.get("u", "")))
picks = sum(1 for e in today if e["t"] == "pick")
more_lists = collections.Counter(e.get("list", "?") for e in today if e["t"] == "more")
errors = collections.Counter(e.get("where", "?") + (f" ({e['status']:.0f})" if e.get("status") else "")
                             for e in today if e["t"] == "error")
left_busy = collections.Counter(e["busy"] for e in today if e["t"] == "hide" and e.get("busy"))
page_people2 = collections.defaultdict(set)
for e in today:
    if e["t"] == "view":
        page_people2[page_of(e)].add(e["v"])

# 2b. Programs and AI apps: what they asked (api-*.jsonl, from Oct 17, 2026) ------------------------------------
api_rows = []
for f in sorted(STATS.glob("api-*.jsonl")):
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if dt.datetime.fromtimestamp(r["at"]).date() == DAY:
            api_rows.append(r)
calls = [r for r in api_rows if r["r"] != "connect"]
agent_routes = collections.Counter(f"{'MCP' if r['k'] == 'mcp' else 'API'} {r['r']}" for r in calls)
agent_queries = per_visitor((r["v"], f"{r['r']}: {r['q'].strip().lower()}") for r in calls
                            if r.get("q") and r["r"] in ("search", "search_papers", "topics", "map_topic",
                                                         "read_topic_group", "find_people", "authors", "institutions"))
agent_clients = len({r["v"] for r in calls})
agent_limited = sum(1 for r in api_rows if r.get("limited"))

# 2. Ask address and search API, from the server journal -----------------------------------------
log = subprocess.run(["journalctl", "--user", "-u", "smr-server", "--no-pager", "-o", "cat",
                      "--since", str(DAY), "--until", str(DAY + dt.timedelta(days=1))],
                     capture_output=True, text=True).stdout.splitlines()
REQ = re.compile(r'^(\d\d):(\d\d):\d\d "GET (\S+) HTTP/[\d.]+" (\d+)')
ask_q, ask_help, ask_junk = collections.Counter(), collections.Counter(), 0
ask_texts = set()
api_search, api_refused, docs = collections.Counter(), 0, 0
help_minute = collections.Counter()
for line in log:
    m = REQ.match(line)
    if not m:
        continue
    h, path, status = int(m.group(1)), m.group(3), int(m.group(4))
    if path.startswith("/api/ask/"):
        words = path[len("/api/ask/"):].split("?")[0]
        if not words:
            ask_help[h] += 1
            help_minute[f"{m.group(1)}:{m.group(2)}"] += 1
        elif status == 200 and words != "robots.txt":
            ask_q[h] += 1
            ask_texts.add(words.lower())
        elif status == 404:
            ask_junk += 1
    elif path.startswith("/api/data/search"):
        if status == 429:
            api_refused += 1
        else:
            api_search[h] += 1
    elif re.match(r"/api/data/(llms\.txt|docs|openapi\.json)", path):
        docs += 1

mcp_tools, mcp_apps, mcp_pings = collections.Counter(), collections.Counter(), 0
for line in log:
    if line.startswith("mcp "):
        parts = line.split(" ", 3)
        method, tool, app = parts[1], parts[2] if len(parts) > 2 else "", (parts[3] if len(parts) > 3 else "").strip()
        if method == "tools/call":
            mcp_tools[tool] += 1
        elif method == "initialize":
            mcp_apps[app or "(unnamed)"] += 1
        elif method == "ping":
            mcp_pings += 1

readers = collections.Counter()
for line in log:
    if line.startswith("robots-reader"):
        ua = line.split(" ", 2)[-1]
        name = re.search(r"(Googlebot|Twitterbot|ClaudeBot|Claude-User|GPTBot|ChatGPT-User|OAI-SearchBot|PerplexityBot|bingbot)", ua)
        readers[name.group(1) if name else ("test" if "test" in ua else "a browser")] += 1
bursts = [(k, n) for k, n in help_minute.most_common() if n >= 100]

# Stripe: checkout pages opened and payments made that day, read-only key (None if Stripe can't be read)
def stripe_day():
    key_file = STATS.parent / "stripe_read_key"
    if not key_file.exists():
        return None
    start = int(dt.datetime.combine(DAY, dt.time()).timestamp())
    q = f"limit=100&created[gte]={start}&created[lt]={start + 86400}"
    out = {}
    for name, ep in (("checkouts", "checkout/sessions"), ("intents", "payment_intents")):
        r = subprocess.run(["curl", "-s", "-g", "-u", key_file.read_text().strip() + ":",
                            f"https://api.stripe.com/v1/{ep}?{q}"], capture_output=True, text=True)
        try:
            out[name] = json.loads(r.stdout)["data"]
        except (ValueError, KeyError):
            return None
    paid = [c for c in out["checkouts"] if c.get("payment_status") == "paid"]
    paid += [p for p in out["intents"] if p.get("status") == "succeeded"
             and not any(c.get("payment_intent") == p["id"] for c in paid)]
    return {"started": len(out["checkouts"]), "paid": len(paid),
            "dollars": sum((x.get("amount_total") or x.get("amount") or 0) for x in paid) / 100}


stripe = stripe_day()
gave_click = sum(1 for e in today if e["t"] == "give")

# 3. The page -----------------------------------------------------------------------------------
data = {
    "day": DAY.strftime("%A %B %-d, %Y"), "until": last_event.strftime("%H:%M"),
    "hours": [f"{h:02d}" for h in HOURS],
    "visitors_x": [len(people[h] & from_x) for h in HOURS],
    "visitors_other": [len(people[h] - from_x) for h in HOURS],
    "searches": [searches[h] for h in HOURS], "opened": [opened[h] for h in HOURS], "reads": [reads[h] for h in HOURS],
    "ask": [ask_q[h] for h in HOURS], "api": [api_search[h] for h in HOURS],
    "funnel": [len(visitors), len(did["searched"]), len(did["opened"]), len(did["read"]), len(did["made"])],
    "pages": sorted(((k, len(v)) for k, v in page_people.items()), key=lambda kv: -kv[1]),
    "devices": [devices["phone"], devices["tablet"], devices["computer"]],
    "days": [d.strftime("%b %-d") for d in days],
    "days_x": [len(daily_x[d]) for d in days], "days_other": [len(daily[d]) for d in days],
    "made": made,
}
total = {
    "visitors": len(visitors), "from_x": len(from_x & visitors), "searches": sum(searches.values()),
    "semantic": search_kind["semantic"], "keyword": search_kind["keyword"],
    "opened": sum(opened.values()), "reads": sum(reads.values()), "made": len(made),
    "made_s": round(sum(s for _, s in made) / len(made)) if made else 0,
    "ask": sum(ask_q.values()), "ask_unique": len(ask_texts), "ask_junk": ask_junk,
    "ask_help": sum(ask_help.values()), "api": sum(api_search.values()), "api_refused": api_refused, "docs": docs,
    "copies": sum(copies.values()), "copy_kinds": dict(copies), "supports": supports,
    "bursts": bursts, "readers": readers.most_common(),
    "topic_maps": sum(n for _, n in top_topics), "people_pages": sum(n for _, n in people_pages),
    "bounced": bounced, "visit_minutes_median": round(statistics.median(lengths) / 60, 1) if lengths else 0,
    "gave_up_field_map": gave_up_field, "left_busy": dict(left_busy), "errors": dict(errors),
    "mcp_calls": sum(mcp_tools.values()), "mcp_tools": dict(mcp_tools), "mcp_connects": sum(mcp_apps.values()),
    "mcp_pings": mcp_pings, "agent_clients": agent_clients,
}

PAGE_NAMES = {"results": "search results", "api": "API / agents page", "work": "a paper's page", "home": "home",
              "explore": "map of science", "library": "library of rewrites", "reader": "reading a rewrite"}
data["pages"] = sorted(([k, len(v)] for k, v in page_people2.items()), key=lambda kv: -kv[1])

pct = lambda a, b: f"{round(100 * a / b)}%" if b else "–"
def table(head, rows, empty="None yet."):
    if not rows:
        return f'<p class="none">{empty}</p>'
    return ("<table><tr>" + "".join(f"<th>{H.escape(h)}</th>" for h in head) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{H.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows) + "</table>")

def ms(q):
    return f"{search_ms[min(len(search_ms) - 1, int(q * len(search_ms)))] / 1000:.1f} s" if search_ms else "–"

NEW = "Recorded from Oct 17, 2026."
looked = f"""
 <div class="card"><h2>What people searched</h2><p>Visitors per search (the same person asking again counts once). Typical wait {ms(.5)}, slowest tenth over {ms(.9)}.</p>{table(["search", "visitors"], top_searches[:25], NEW)}</div>
 <div class="card"><h2>Searches that found nothing</h2><p>What people wanted and didn't get.</p>{table(["search", "visitors"], empty_searches[:15], "None.")}</div>
 <div class="card"><h2>Topic maps opened</h2><p>{picks} boxes picked inside maps; reading lists read further: {', '.join(f'{k} {n}' for k, n in more_lists.most_common()) or 'none'}.
   {gave_up_field} left a whole-field map before it was ready.</p>{table(["topic", "visitors"], top_topics[:20], NEW)}</div>
 <div class="card"><h2>People and organisations looked at</h2><p>OpenAlex ids (A: author, I: institution).</p>{table(["page", "visitors"], people_pages[:15], NEW)}</div>
 <div class="card wide"><h2>How visits went</h2><p>{len(lengths)} visits; typical length {total['visit_minutes_median']} min; {bounced} left from their first page within 15 s without doing anything ({pct(bounced, len(lengths))}).
   Left while something was still under way: {', '.join(f'{k} {n}' for k, n in left_busy.most_common()) or 'none'}.
   Errors shown: {', '.join(f'{k} {n}' for k, n in errors.most_common()) or 'none'}.</p>
   <div class="grid">{table(["the first pages of a visit", "visitors"], paths.most_common(12))}
   <div>{table(["last page before leaving", "visitors"], exits.most_common(8))}
   {table(["after search results, they went to", "times"], after_search.most_common(8), "No step after a search yet.")}</div></div></div>
 <div class="card wide"><h2>AI apps and programs</h2><p>MCP: {total['mcp_calls']:,} tool calls ({', '.join(f'{k} {n}' for k, n in mcp_tools.most_common()) or 'none'});
   {total['mcp_connects']:,} connections by app; {mcp_pings:,} keep-alive pings (not use). Programs on the JSON API: {total['api']:,} searches.
   {agent_clients} different addresses made calls; {agent_limited} requests were over the rate limit.</p>
   <div class="grid">{table(["what agents asked", "addresses"], agent_queries[:25], NEW)}
   <div>{table(["tool or route", "calls"], agent_routes.most_common(12), NEW)}
   {table(["app connecting (MCP)", "connections"], mcp_apps.most_common(12), "None.")}</div></div></div>
"""

bursts_text = ", ".join(f"{n:,} in the minute of {k}" for k, n in total["bursts"]) or "none"
readers_text = ", ".join(f"{k} ({n})" for k, n in total["readers"]) or "nobody"
made_text = ", ".join(f"{t} ({s:.0f} s)" for t, s in made) or "none"

html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScienceMadeReadable · {data['day']}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a65; --card:#fff; --line:#e7e4dc; --a:#d9822b; --b:#4a6fa5; --c:#5a9e6f; --d:#a05a9e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#141412; --fg:#ecebe6; --muted:#9a988f; --card:#1d1d1a; --line:#2e2d29; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:1100px; margin:auto; padding:28px 20px 60px; }}
h1 {{ font-size:26px; margin:0 0 2px; }} h2 {{ font-size:17px; margin:0 0 4px; }}
.sub {{ color:var(--muted); margin:0 0 22px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:18px; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px; }}
.tile b {{ display:block; font-size:28px; line-height:1.1; }} .tile span {{ color:var(--muted); font-size:13px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
.card p {{ color:var(--muted); font-size:13px; margin:0 0 10px; }} .wide {{ grid-column:1/-1; }}
canvas {{ max-height:280px; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin-top:6px; }} th {{ text-align:left; color:var(--muted); font-weight:500; }}
td, th {{ padding:3px 6px; border-bottom:1px solid var(--line); vertical-align:top; }} td:last-child, th:last-child {{ text-align:right; white-space:nowrap; }}
.none {{ font-style:italic; }} ul {{ margin:6px 0 0; padding-left:18px; color:var(--muted); font-size:13px; }}
</style>
<main>
<h1>ScienceMadeReadable · {data['day']}</h1>
<p class="sub">Eastern time, midnight to {data['until']}. Website numbers count people (one private code per visitor per day);
Ask, API and MCP numbers count requests.</p>

<div class="tiles">
 <div class="tile"><b>{total['visitors']}</b><span>visitors · {pct(total['from_x'], total['visitors'])} came from X</span></div>
 <div class="tile"><b>{total['searches']}</b><span>searches on the site ({total['semantic']} by meaning, {total['keyword']} exact words)</span></div>
 <div class="tile"><b>{total['opened']}</b><span>paper pages opened</span></div>
 <div class="tile"><b>{total['reads']}</b><span>plain-language rewrites read</span></div>
 <div class="tile"><b>{total['made']}</b><span>new rewrites written (avg {total['made_s']} s each)</span></div>
 <div class="tile"><b>{total['topic_maps']}</b><span>topic maps opened · {total['people_pages']} author and institution pages</span></div>
 <div class="tile"><b>{total['mcp_calls']:,}</b><span>MCP tool calls by AI apps ({total['mcp_connects']:,} connections)</span></div>
 <div class="tile"><b>{total['ask']}</b><span>questions to ask.sciencemadereadable.com ({total['ask_unique']} different)</span></div>
 <div class="tile"><b>{total['api']:,}</b><span>search API calls (site + agents + scripts)</span></div>
 <div class="tile"><b>{total['supports']}</b><span>opened the “support” panel · {gave_click} clicked a payment button</span></div>
 <div class="tile"><b>{'?' if stripe is None else f"${stripe['dollars']:,.0f}"}</b><span>{'Stripe could not be read' if stripe is None else f"donated ({stripe['paid']} payments; {stripe['started']} payment pages opened)"}</span></div>
</div>

<div class="grid">
 <div class="card wide"><h2>Visitors per hour</h2><p>People seen on the website each hour. Orange: their first page came from a link on X.</p><canvas id="vis"></canvas></div>
 <div class="card wide"><h2>What they did, per hour</h2><p>Searches, paper pages opened, and rewrites read (a rewrite counts once per person, even if the tab stays open).</p><canvas id="act"></canvas></div>
 <div class="card"><h2>How far people went</h2><p>Of everyone who came, how many searched, opened a paper, read a rewrite, asked for a new one.</p><canvas id="fun"></canvas></div>
 <div class="card"><h2>Which pages they saw</h2><p>People who saw each page at least once (before Oct 17, topic maps and people pages count as search results).</p><canvas id="pag"></canvas></div>
 <div class="card"><h2>Phone or computer</h2><p>From screen width: under 700 px phone, under 1100 px tablet.</p><canvas id="dev"></canvas></div>
 <div class="card"><h2>Agents and scripts</h2><p>Requests per hour: questions typed at the Ask address, and all calls to the search API (the site's own searches go through it too).</p><canvas id="bot"></canvas></div>
{looked}
 <div class="card wide"><h2>Today next to the days before</h2><p>Visitors per day since launch. A person who returns on another day counts again.</p><canvas id="days"></canvas></div>
 <div class="card wide"><h2>Notes</h2><ul>
  <li>Agent setup copied from the API page: {total['copies']} times ({', '.join(f'{k}: {n}' for k, n in total['copy_kinds'].items()) or 'none'}). API guide read {total['docs']} times.</li>
  <li>New rewrites finished at: {made_text}.</li>
  <li>The Ask help page (the address with no question) was loaded {total['ask_help']:,} times; bursts: {bursts_text}. That is a program, not people, so it is left out of the charts.</li>
  <li>{total['ask_junk']} Ask requests were robots probing for files (wp-login, feed.xml…) and got “not a question”. {total['api_refused']} API calls were refused for going over 30 a minute.</li>
  <li>Who read the Ask robots.txt: {readers_text}.</li>
  <li>Where people landed: {', '.join(f'{PAGE_NAMES.get(k, k)} {n}' for k, n in landed.most_common(4))}.</li>
  <li>The 4 busiest visitors made {top4} of the {sum(by_person.values())} searches ({pct(top4, sum(by_person.values()))}).
      Your own visits and tests are mixed in: they cannot be told apart from other people's.</li>
 </ul></div>
</div>
</main>
<script>
const D = {json.dumps(data)};
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
Chart.defaults.color = css('--muted'); Chart.defaults.borderColor = css('--line'); Chart.defaults.font.family = 'system-ui';
const A = css('--a'), B = css('--b'), C = css('--c'), P = css('--d');
const stack = {{ x: {{ stacked: true }}, y: {{ stacked: true, beginAtZero: true }} }};
new Chart(vis, {{ type: 'bar', data: {{ labels: D.hours, datasets: [
  {{ label: 'came from X', data: D.visitors_x, backgroundColor: A }},
  {{ label: 'other', data: D.visitors_other, backgroundColor: B }} ] }}, options: {{ scales: stack }} }});
new Chart(act, {{ type: 'line', data: {{ labels: D.hours, datasets: [
  {{ label: 'searches', data: D.searches, borderColor: B, backgroundColor: B, tension: .3 }},
  {{ label: 'paper pages opened', data: D.opened, borderColor: A, backgroundColor: A, tension: .3 }},
  {{ label: 'rewrites read', data: D.reads, borderColor: C, backgroundColor: C, tension: .3 }} ] }},
  options: {{ scales: {{ y: {{ beginAtZero: true }} }} }} }});
new Chart(fun, {{ type: 'bar', data: {{ labels: ['came', 'searched', 'opened a paper', 'read a rewrite', 'asked for a rewrite'],
  datasets: [{{ data: D.funnel, backgroundColor: [B, B, A, C, P] }}] }},
  options: {{ indexAxis: 'y', plugins: {{ legend: {{ display: false }} }} }} }});
new Chart(pag, {{ type: 'bar', data: {{ labels: D.pages.map(p => p[0]), datasets: [{{ data: D.pages.map(p => p[1]), backgroundColor: B }}] }},
  options: {{ indexAxis: 'y', plugins: {{ legend: {{ display: false }} }} }} }});
new Chart(dev, {{ type: 'doughnut', data: {{ labels: ['phone', 'tablet', 'computer'], datasets: [{{ data: D.devices, backgroundColor: [A, C, B], borderWidth: 0 }}] }},
  options: {{ plugins: {{ legend: {{ position: 'right' }} }} }} }});
new Chart(bot, {{ type: 'bar', data: {{ labels: D.hours, datasets: [
  {{ label: 'Ask questions', data: D.ask, backgroundColor: P }},
  {{ label: 'search API calls', data: D.api, backgroundColor: B }} ] }}, options: {{ scales: {{ y: {{ beginAtZero: true }} }} }} }});
new Chart(days, {{ type: 'bar', data: {{ labels: D.days, datasets: [
  {{ label: 'came from X', data: D.days_x, backgroundColor: A }},
  {{ label: 'other', data: D.days_other, backgroundColor: B }} ] }}, options: {{ scales: stack }} }});
</script></html>"""
OUT.write_text(html)
print(OUT)
print(json.dumps(total, indent=1, default=str))
