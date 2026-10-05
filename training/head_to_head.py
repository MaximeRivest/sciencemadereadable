"""Blind head-to-head between every pair of writers, on the benchmark's 3 test papers (15 parts).

    .venv/bin/python training/head_to_head.py            # run (cached calls are reused)
    .venv/bin/python training/head_to_head.py --report   # tables only
    add --judge astra to use GPT-6 Astra (low reasoning) as the judge instead of Opus

Same comparison as training/v3/polish.py: Opus (low reasoning) sees the original part
and two rewrites labelled only A and B, and says which a curious 14-year-old would find more
pleasant and easier to follow, given that it must stay faithful (or "tie"). Every pair is asked
in both orders; a win counts only when it holds both ways, otherwise "no clear winner".
Rewrites are the scored candidates (model_baselines/rewrites/NAME), one part = one
benchmark unit (opening, introduction_rest, methods, results, discussion).

Writes speed/../head_to_head/{calls.jsonl, results.json} and prints the tables. A Bradley-Terry
strength (from wins; a no-clear-winner counts half to each) puts the 6 on one scale.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))
sys.path.insert(0, str(HERE / "v3"))
import eval_v3                                       # noqa: E402
from polish import COMPARE, compare, unmark          # noqa: E402  (the same judge instructions and settings)
import recipes                                       # noqa: E402  (each writer's settings)

WRITERS = {"Opus": "opus", "Sonnet": "v8-sonnet", "Astra": "v8-astra", "Sol": "v8-sol", "Terra 5.6": "v8-terra",
           "Luna": "v8-luna",
           "Our 9B": "student-9b-glossary-final-greedy", "Our 4B": "student-4b-glossary-final-greedy",
           "Our 0.8B": "student-0.8b-glossary-final-greedy"}
READER = "a curious 14-year-old"
OUT = HERE / "head_to_head"
OUT.mkdir(exist_ok=True)
JUDGES = {"opus": COMPARE,   # Opus, reasoning low (as in polish.py)
          "astra": compare.using(**recipes.WRITER_SETTINGS["astra"])}   # GPT-6 Astra, reasoning low
JUDGE = "opus"
CALLS = OUT / "calls.jsonl"   # set per judge in main


def set_judge(name):
    global JUDGE, CALLS
    JUDGE = name
    CALLS = OUT / ("calls.jsonl" if name == "opus" else f"calls-{name}.jsonl")


def parts():
    """{(paper, unit): {"original": ..., writer: rewrite}} for the 3 test papers."""
    table = {}
    for label, name in WRITERS.items():
        rewrites = eval_v3.load_rewrites(name)
        for paper in eval_v3.test_papers()[:3]:
            for unit, original, new, _ in eval_v3.units(paper, rewrites.get(paper["paper_id"], {})):
                row = table.setdefault((paper["paper_id"], unit), {"original": original})
                row[label] = unmark(new)
    return table


def key(original, a, b):
    return hashlib.sha256(json.dumps([READER, original, a, b]).encode()).hexdigest()[:24]


def load_cache():
    cache = {}
    if CALLS.exists():
        for line in CALLS.read_text().splitlines():
            d = json.loads(line)
            cache[d["key"]] = d
    return cache


def run(table):
    cache, lock = load_cache(), threading.Lock()
    jobs = []
    for (paper, unit), row in table.items():
        for x, y in itertools.combinations(WRITERS, 2):
            for first, second in [(x, y), (y, x)]:
                k = key(row["original"], row[first], row[second])
                if k not in cache:
                    jobs.append((k, paper, unit, first, second, row))
    print(f"{len(jobs)} comparisons to ask ({len(cache)} already done)", flush=True)

    def ask(job):
        k, paper, unit, first, second, row = job
        for attempt in range(3):
            try:
                p = JUDGES[JUDGE](original=row["original"], version_a=row[first], version_b=row[second], reader=READER)
                rec = {"key": k, "paper": paper, "unit": unit, "a": first, "b": second,
                       "winner": p.winner, "reasons": p.reasons}
                break
            except Exception as error:   # noqa: BLE001 - not cached, so retried on the next run
                rec = None
                err = f"{type(error).__name__}: {str(error)[:150]}"
        if rec is None:
            print("  failed:", paper, unit, first, "vs", second, err, flush=True)
            return
        with lock:
            with CALLS.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache[k] = rec
    done = 0
    with ThreadPoolExecutor(12) as pool:
        for _ in pool.map(ask, jobs):
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)


def outcomes(table):
    """One outcome per (part, pair): the winner's label, or None (no clear winner)."""
    cache = load_cache()
    res = []
    for (paper, unit), row in table.items():
        for x, y in itertools.combinations(WRITERS, 2):
            d1 = cache.get(key(row["original"], row[x], row[y]))      # x shown as A
            d2 = cache.get(key(row["original"], row[y], row[x]))      # y shown as A
            if not d1 or not d2:
                continue
            w1 = {"A": x, "B": y, "tie": None}[d1["winner"]]
            w2 = {"A": y, "B": x, "tie": None}[d2["winner"]]
            res.append({"paper": paper, "unit": unit, "x": x, "y": y, "winner": w1 if w1 == w2 else None,
                        "orders": [d1["winner"], d2["winner"]], "first_shown_won": [d1["winner"] == "A", d2["winner"] == "A"]})
    return res


def bradley_terry(res, iters=2000):
    """Strengths from pairwise results (no-clear-winner = half a win each), on an Elo-like scale."""
    names = list(WRITERS)
    wins = {(a, b): 0.0 for a in names for b in names}
    for r in res:
        if r["winner"]:
            loser = r["y"] if r["winner"] == r["x"] else r["x"]
            wins[(r["winner"], loser)] += 1
        else:
            wins[(r["x"], r["y"])] += 0.5
            wins[(r["y"], r["x"])] += 0.5
    p = {n: 1.0 for n in names}
    for _ in range(iters):
        new = {}
        for i in names:
            w = sum(wins[(i, j)] for j in names if j != i)
            d = sum((wins[(i, j)] + wins[(j, i)]) / (p[i] + p[j]) for j in names if j != i)
            new[i] = max(w, 1e-3) / d if d else p[i]
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        p = {n: v / g for n, v in new.items()}
    return {n: 400 * math.log10(v) for n, v in p.items()}


def report(table):
    res = outcomes(table)
    names = list(WRITERS)
    print(f"\n{len(res)} part-by-pair results ({len(table)} parts x {len(names) * (len(names) - 1) // 2} pairs)")
    pos = [w for r in res for w in r["first_shown_won"]]
    print(f"position check: the version shown first won {sum(pos)} of {len(pos)} single judgments")
    print("\nrow vs column: wins-losses (no clear winner)")
    print(f"{'':11}" + "".join(f"{n:>11}" for n in names))
    grid = {}
    for a in names:
        line = f"{a:11}"
        for b in names:
            if a == b:
                line += f"{'—':>11}"
                continue
            rs = [r for r in res if {r["x"], r["y"]} == {a, b}]
            w, l = sum(r["winner"] == a for r in rs), sum(r["winner"] == b for r in rs)
            t = len(rs) - w - l
            grid[f"{a}|{b}"] = {"wins": w, "losses": l, "no_clear_winner": t}
            line += f"{f'{w}-{l} ({t})':>11}"
        print(line)
    bt = bradley_terry(res)
    print("\nstrength (Bradley-Terry, Elo-like scale, mean 0):")
    for n in sorted(names, key=lambda n: -bt[n]):
        w = sum(r["winner"] == n for r in res)
        l = sum(r["winner"] not in (None, n) and n in (r["x"], r["y"]) for r in res)
        print(f"  {n:11} {bt[n]:+6.0f}   overall {w} wins, {l} losses")
    name = "results.json" if JUDGE == "opus" else f"results-{JUDGE}.json"
    (OUT / name).write_text(json.dumps({"judge": JUDGE, "writers": WRITERS, "reader": READER, "grid": grid,
                                                  "strength": bt, "outcomes": res}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--judge", choices=sorted(JUDGES), default="opus")
    a = ap.parse_args()
    set_judge(a.judge)
    print("judge:", a.judge)
    t = parts()
    missing = [(k, w) for k, row in t.items() for w in WRITERS if not row.get(w, "").strip()]
    if missing:
        print("empty rewrites:", missing[:5])
    if not a.report:
        run(t)
    report(t)
