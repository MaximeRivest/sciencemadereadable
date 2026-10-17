"""Plot sciencemadereadable.com usage through time, hour by hour, from app/stats/events-*.jsonl.

    uv run -q --no-project --with matplotlib app/tools/plot_usage.py [out.png]

Writes app/stats/usage.png by default (app/stats/ is private and not committed).

Two things to know when reading the chart:
  * A visitor code changes every midnight, so "visitors" means people seen that hour; the same person
    on two days counts twice in the running total ("visitor-days").
  * The site's own automatic check (run at every publish) is left out (without_site_checks).
  * Topic maps and author / institution pages are counted from Oct 17, 2026 (before, the server dropped them).
  * The page re-sends "read" every 30 s while a saved rewrite stays open (the GPU-status refresh
    calls showChosen). Raw "read" counts are therefore open-tab time, not reads. Here a read is counted
    once per visitor, paper and model.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

STATS = Path(__file__).resolve().parents[1] / "stats"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else STATS / "usage.png"

# 1. Load every event --------------------------------------------------------------------------
events = []
for f in sorted(STATS.glob("events-*.jsonl")):
    for line in f.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
events.sort(key=lambda e: e["at"])


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
if not events:
    sys.exit("no events recorded yet")


def hour(e):
    return dt.datetime.fromtimestamp(e["at"]).replace(minute=0, second=0, microsecond=0)


start, end = hour(events[0]), hour(events[-1])
hours = [start + dt.timedelta(hours=i) for i in range(int((end - start).total_seconds() // 3600) + 1)]

# 2. Count per hour ----------------------------------------------------------------------------
visitors = collections.defaultdict(set)      # hour → visitor codes seen
from_x = set()                               # visitor codes whose first page came via t.co (X/Twitter)
for e in events:
    if e["t"] == "view" and e.get("ref") == "t.co":
        from_x.add(e["v"])
count = {k: collections.Counter() for k in ("view", "search", "open", "read", "made", "topics", "people")}
seen_read = set()
for e in events:
    h, t = hour(e), e["t"]
    visitors[h].add(e["v"])
    if t in ("view", "search"):
        count[t][h] += 1
    elif t == "open" and not e.get("err"):
        count["open"][h] += 1
    elif t == "read":
        key = (e["v"], e.get("doi"), e.get("model"))
        if key not in seen_read:
            seen_read.add(key)
            count["read"][h] += 1
    elif t in ("job_done", "done"):
        count["made"][h] += 1
    elif t == "topics" and not e.get("pending"):   # counted from Oct 17, 2026
        count["topics"][h] += 1
    elif t in ("author", "institution"):
        count["people"][h] += 1

vx = [len(visitors[h] & from_x) for h in hours]
vo = [len(visitors[h] - from_x) for h in hours]
day_codes = {(dt.datetime.fromtimestamp(e["at"]).date(), e["v"]) for e in events}

# 3. Draw --------------------------------------------------------------------------------------
fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True,
                       gridspec_kw={"height_ratios": [3, 2, 2, 2.4]})
w = 1 / 24 * 0.85
ax[0].bar(hours, vo, w, label="direct / other", color="#4c78a8")
ax[0].bar(hours, vx, w, bottom=vo, label="arrived from X (t.co)", color="#f58518")
ax[0].set_ylabel("visitors\nper hour")
ax[0].legend(loc="upper right", frameon=False)
ax[0].set_title(f"sciencemadereadable.com usage, {start:%b %d %H:%M} – {dt.datetime.fromtimestamp(events[-1]['at']):%b %d %H:%M} "
                f"(server time) · {len(day_codes)} visitor-days", loc="left")

for k, lab, c in [("view", "pages shown", "#4c78a8"), ("search", "searches", "#54a24b")]:
    ax[1].plot(hours, [count[k][h] for h in hours], label=lab, color=c, marker=".")
ax[1].set_ylabel("per hour")
ax[1].legend(loc="upper right", frameon=False)

for k, lab, c in [("open", "papers opened", "#b279a2"), ("topics", "topic maps opened", "#9d755d"),
                  ("people", "author / institution pages", "#bab0ac"), ("read", "saved rewrites read", "#e45756"),
                  ("made", "new rewrites made", "#72b7b2")]:
    ax[2].plot(hours, [count[k][h] for h in hours], label=lab, color=c, marker=".")
ax[2].set_ylabel("per hour")
ax[2].legend(loc="upper right", frameon=False)


def running(xs):
    out, s = [], 0
    for x in xs:
        s += x
        out.append(s)
    return out


new_visitor_days = collections.Counter()
for d, v in day_codes:
    first = min(e["at"] for e in events if e["v"] == v and dt.datetime.fromtimestamp(e["at"]).date() == d)
    new_visitor_days[hour({"at": first})] += 1
for series, lab, c in [(new_visitor_days, "visitor-days", "#f58518"), (count["search"], "searches", "#54a24b"),
                       (count["open"], "papers opened", "#b279a2"), (count["topics"], "topic maps opened", "#9d755d"),
                       (count["people"], "author / institution pages", "#bab0ac"),
                       (count["read"], "saved rewrites read", "#e45756"),
                       (count["made"], "new rewrites made", "#72b7b2")]:
    ys = running([series[h] for h in hours])
    ax[3].step(hours, ys, where="post", label=f"{lab} ({ys[-1]})", color=c)
ax[3].set_ylabel("running total")
ax[3].legend(loc="upper left", frameon=False)

for a in ax:
    a.grid(axis="y", alpha=0.3)
    a.spines[["top", "right"]].set_visible(False)
    for h in hours:
        if h.hour == 0:
            a.axvline(h, color="grey", lw=0.6, ls=":")
if end - start > dt.timedelta(days=3):     # long spans: one label per day, a small tick at noon
    ax[3].xaxis.set_major_locator(mdates.DayLocator())
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%b %d"))
    ax[3].xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
else:
    ax[3].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M"))
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(OUT)
