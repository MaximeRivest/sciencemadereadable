"""
Judge a finished try again, with the current judge (prompts.compare), without rewriting anything
================================================================================================

Used once so far: 4b was judged by a judge that did not see the paper's figure captions and
tables; this asks the corrected judge on the same texts. The first verdicts are kept in each part
as "judge_before" and the new ones replace "judge".

    .venv/bin/python training/notes/bundle/rejudge.py 4b
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pilot as P
import render

TRY = sys.argv[1]
P.OUT = P.HERE / "out" / TRY

for f in sorted(P.OUT.glob("PMC*.json")):
    d = json.loads(f.read_text())
    parts = {p["section"]: p for p in d["parts"]}
    names = list(parts)
    figures = "\n\n".join(render._figure(x) for x in d["verbatim"]["figures"]) or "(none)"
    jobs = [(n, pair) for n in names for pair in parts[n]["judge"]]

    def one(job):
        n, pair = job
        a, b = pair.split(" vs ")
        before = lambda x: "" if n == "opening" else parts["opening"]["texts"][x]
        rest = "\n\n".join(parts[m]["original"] for m in names if m != n)
        r = P.judge(parts[n]["original"], rest, figures, before(a), parts[n]["texts"][a], before(b), parts[n]["texts"][b])
        r["winner"] = {"first": a, "second": b}.get(r["winner"], r["winner"])
        return r

    with ThreadPoolExecutor(10) as pool:
        verdicts = list(pool.map(one, jobs))
    for p in d["parts"]:
        p.setdefault("judge_before", p["judge"])
        p["judge"] = {}
    for (n, pair), r in zip(jobs, verdicts):
        parts[n]["judge"][pair] = r
    f.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    print("rejudged", d["paper_id"], flush=True)
P.report()
