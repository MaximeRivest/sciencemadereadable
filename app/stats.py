"""Usage of the site and of the API, kept on our own server: no cookies, no third party.

A visitor is a code made each day from their address and browser, with a random key that changes
every day and is then thrown away: visits can be followed within a day (what people look for, where
they go next, where they give up), nobody can be followed across days, and the address itself is
never stored. Browsers that send Do Not Track or Global Privacy Control send nothing (the page checks).
Files older than KEEP_MONTHS are deleted.

Site events (app/stats/events-YYYY-MM.jsonl, one JSON per line):
  view     a page shown (view: home, results, work, reader, explore, library, fields, api, support, reading,
           account; u: the page's address, only the parameters in track.ts's list: the search, the topic,
           the author, the paper...; ref: the site the visitor came from, first view only)
  hide     the page left or put in the background (view, s: seconds since the page was shown, busy: what was
           still loading or being written then)
  search   q: the words, by, n: papers found, ok: how many of them can be opened, ms: how long it took
  topics   a topic map shown (q, group, field, groups, cached, pending: still being made), pick: a box picked
           in it (box), more: a reading list read further (list, n: from)
  author / institution   a person's or organisation's page (works, groups)
  open     a paper opened (doi), or err: why it couldn't be (licence, layout, missing...)
  read     a saved rewrite shown (doi, model, src: benchmark/demo/browser)
  make     "make it readable" pressed (doi, model)
  done / fail   a rewrite finished in the page (doi, model, s: seconds)
  error    something the reader was told didn't work (where, status)
  job, job_done, job_failed   recorded by the queue itself (s: writing time, wait: time in line)
  support, give, thanks_shown, supported, field_pay, field_paid, copy   the support and API pages

API events (app/stats/api-YYYY-MM.jsonl), from server.py, for programs and AI apps (not the site's own calls):
  k: api (the JSON API) or mcp; r: the route or the MCP method/tool; q: its main argument (the words, the topic,
  the ids); app: the MCP app or the program's name; v: the daily code of the address; key: 1 when an API key
  was used (never which); limited: 1 when refused for the rate limit.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import secrets
import threading
import time
from pathlib import Path

DIR = Path(__file__).resolve().parent / "stats"
DIR.mkdir(exist_ok=True)
LOCK = threading.Lock()
TYPES = {"view", "hide", "search", "topics", "pick", "more", "author", "institution", "open", "read", "make", "done",
         "fail", "error", "job", "job_done", "job_failed", "support", "give", "thanks_shown", "supported", "copy",
         "field_pay", "field_paid"}
# text fields: their longest length; 0: a number
FIELDS = {"view": 12, "u": 400, "ref": 80, "q": 300, "by": 12, "doi": 120, "model": 20, "err": 40, "src": 12,
          "where": 20, "busy": 20, "box": 20, "list": 20, "group": 12, "field": 4, "title": 200, "name": 120, "id": 60,
          "n": 0, "s": 0, "wait": 0, "w": 0, "ok": 0, "ms": 0, "status": 0, "groups": 0, "works": 0, "cached": 0,
          "sub": 0, "pending": 0, "eta_h": 0, "amount": 0}
assert not FIELDS.keys() & {"at", "t", "v"}, "a field would overwrite the time, type or visitor"
KEEP_MONTHS = 25   # the longest the French data regulator (CNIL) accepts for audience measurement
RECENT = collections.defaultdict(collections.deque)   # address → times of its last events (rate limit)


def _salt(day: str) -> str:
    """Today's random key, kept on disk only for today (a restart keeps today's counts whole)."""
    f = DIR / f"salt-{day}"
    if not f.exists():
        for old in DIR.glob("salt-*"):
            old.unlink()
        f.write_text(secrets.token_hex(16))
        f.chmod(0o600)
        _forget_old(dt.date.fromisoformat(day))
    return f.read_text()


def _forget_old(today: dt.date):
    """Monthly files older than KEEP_MONTHS go (checked once a day, when the key changes)."""
    first = today.year * 12 + today.month - 1 - KEEP_MONTHS
    for f in DIR.glob("*-????-??.jsonl"):
        y, m = f.stem[-7:].split("-")
        if y.isdigit() and m.isdigit() and int(y) * 12 + int(m) - 1 < first:
            f.unlink()


def visitor(address: str, agent: str) -> str:
    return hashlib.sha256(f"{_salt(dt.date.today().isoformat())}|{address}|{agent}".encode()).hexdigest()[:10]


def _write(name: str, out: dict):
    with LOCK, (DIR / f"{name}-{dt.date.today().isoformat()[:7]}.jsonl").open("a") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


def record_api(kind: str, route: str, q: str, app: str, address: str, keyed: bool = False, limited: bool = False):
    """One request from a program or an AI app (server.py): what it asked for, never who it is."""
    out = {"at": round(time.time()), "k": kind, "r": route[:40], "v": visitor(address, "")}
    if q:
        out["q"] = q[:300]
    if app:
        out["app"] = app[:60]
    if keyed:
        out["key"] = 1
    if limited:
        out["limited"] = 1
    try:
        _write("api", out)
    except OSError:
        pass


def record(e: dict, address: str, agent: str):
    t = e.get("t")
    if t not in TYPES:
        return
    now = time.time()
    q = RECENT[address]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= 120:          # at most 120 counts a minute from one address
        return
    q.append(now)
    out = {"at": round(now), "t": t, "v": visitor(address, agent)}
    for k, n in FIELDS.items():
        v = e.get(k)
        if v is None:
            continue
        if n == 0:   # a number
            if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) < 1e7:
                out[k] = round(float(v), 1)
        else:
            out[k] = str(v)[:n]
    _write("events", out)
