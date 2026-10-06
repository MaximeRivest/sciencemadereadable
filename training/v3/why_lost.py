"""
Why did v3 lose the blind head-to-head?  (analysis of the 50-paper pilot)

The pilot kept only the winner of each blind comparison. This asks the same judge again on a
sample of parts, in both orders, and keeps its reasons. Output: pilot50/why_lost.json.

    .venv/bin/python training/v3/why_lost.py
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor

import pilot50 as P

random.seed(7)
parts = []
for f in sorted(P.OUT.glob("PMC*.json")):
    d = json.loads(f.read_text())
    for c in d["conversations"]:
        parts.append((d["paper_id"], c))

old_wins = [p for p in parts if p[1]["measure"]["blind"] == "old"]
new_wins = [p for p in parts if p[1]["measure"]["blind"] == "new"]
sample = random.sample(old_wins, 30) + random.sample(new_wins, 10)


def why(item):
    pid, c = item
    a = P.ask(P.COMPARE, "blind", original=c["original"], version_a=c["old_answer"], version_b=c["filled"], reader="a curious 14-year-old")
    b = P.ask(P.COMPARE, "blind", original=c["original"], version_a=c["filled"], version_b=c["old_answer"], reader="a curious 14-year-old")
    name = lambda w, first: {"A": first, "B": "new" if first == "old" else "old"}.get(w, "tie")
    return {"paper": pid, "section": c["section"], "pilot_result": c["measure"]["blind"],
            "markers": c["measure"]["markers"],
            "old_first": {"winner": name(a.winner, "old"), "reasons": a.reasons},
            "new_first": {"winner": name(b.winner, "new"), "reasons": b.reasons}}


with ThreadPoolExecutor(8) as pool:
    out = list(pool.map(why, sample))
(P.OUT / "why_lost.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("done", len(out))
