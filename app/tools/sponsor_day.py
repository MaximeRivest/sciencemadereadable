"""Day sponsors: see who paid to sponsor a day, and put their name on the site.

    .venv/bin/python app/tools/sponsor_day.py                       # who is waiting, and the days booked
    .venv/bin/python app/tools/sponsor_day.py approve cs_... [DAY]  # show them on that day (default: the day
                                                                    # they asked for, else the next free day)
    .venv/bin/python app/tools/sponsor_day.py approve cs_... DAY --name "Shorter Name" --url https://...
    .venv/bin/python app/tools/sponsor_day.py refuse cs_...         # not shown (refund them in Stripe)
    .venv/bin/python app/tools/sponsor_day.py unbook DAY

The queue writes app/sponsors.json when a "sponsor a day" payment arrives (Stripe). Nothing reaches the
site before it is approved here: the name goes into support.json's sponsors_by_day, read on every request.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
SPONSORS = APP / "sponsors.json"
SUPPORT = APP / "support.json"


def load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def save(p: Path, d: dict):
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")


def option(args: list[str], name: str) -> str | None:
    if name in args:
        i = args.index(name)
        value = args[i + 1]
        del args[i:i + 2]
        return value
    return None


def main(args: list[str]):
    sponsors, support = load(SPONSORS), load(SUPPORT)
    booked = support.setdefault("sponsors_by_day", {})
    if not args:
        waiting = {k: v for k, v in sponsors.items() if v["status"] == "waiting"}
        print(f"{len(waiting)} waiting for approval")
        for k, v in waiting.items():
            paid = dt.datetime.fromtimestamp(v["paid"]).strftime("%Y-%m-%d %H:%M")
            print(f"  {k}  ${v['dollars']:.0f}  paid {paid}  name: {v.get('name')!r}  link: {v.get('website')!r}  "
                  f"day: {v.get('day')!r}")
        print("days booked:", ", ".join(f"{d} {b['name']}" for d, b in sorted(booked.items())) or "none")
        return
    verb, sid = args[0], args[1] if len(args) > 1 else None
    if verb == "unbook":
        booked.pop(sid, None)
        save(SUPPORT, support)
        print(f"{sid} is free again")
        return
    if sid not in sponsors:
        sys.exit(f"no sponsor {sid}; run without arguments to see them")
    s = sponsors[sid]
    if verb == "refuse":
        s["status"] = "refused"
        save(SPONSORS, sponsors)
        print("refused; refund them in the Stripe dashboard (Payments)")
        return
    if verb != "approve":
        sys.exit(__doc__)
    name, url = option(args, "--name") or s.get("name"), option(args, "--url") or s.get("website")
    day = args[2] if len(args) > 2 else None
    if not day and re.fullmatch(r"\d{4}-\d{2}-\d{2}", (s.get("day") or "").strip()):
        day = s["day"].strip()
    if not day:
        d = dt.date.today()
        while d.isoformat() in booked:
            d += dt.timedelta(days=1)
        day = d.isoformat()
    dt.date.fromisoformat(day)
    if day in booked:
        sys.exit(f"{day} is already booked by {booked[day]['name']}; choose another day")
    if url and not url.startswith(("https://", "http://")):
        url = "https://" + url
    booked[day] = {"name": name.strip()[:60], **({"url": url.strip()} if url else {})}
    s["status"], s["shown_on"] = "approved", day
    save(SUPPORT, support)
    save(SPONSORS, sponsors)
    print(f"{name} is shown on {day}: \"Today's rewrites are sponsored by {name}\"")


main(sys.argv[1:])
