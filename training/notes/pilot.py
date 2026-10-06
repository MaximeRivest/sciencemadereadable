"""
Writing from notes · the pilot
==============================

Question: can a writer that knows nothing about a field write a paper well for a curious
14-year-old when it gets the knowledge as notes? If yes, a small model only needs writing skill
and everyday knowledge, not the science. Here the writer is Opus: this run measures the ceiling
and checks the method before any small model is trained on it. (README.md explains the idea.)

For each paper, each part (the opening, then the 4 sections):

  1. Opus, as the scientist, turns the part into notes: facts, plan, definitions
  2. a checker compares notes with the paper: what did the notes lose, change or add?
     If anything: Opus corrects the notes, and the checker looks again (from round 2)
  3. Opus, as the writer, writes from the notes alone, twice:
       level 1 (notes with a paragraph plan and importance)  and  level 3 (shuffled facts only)
  4. a checker traces each text back to the notes: facts kept, changed, missing; additions
  5. the fair blind judge compares, against the old Opus answer (written from the paper):
       level 1 vs old, level 3 vs old, and v3 vs old (to see what the fair judge does to v3)

The 5 papers come from the v3 pilot (training/v3/pilot50), so the old and v3 answers already
exist and the comparison is like for like.

How to run
----------
    .venv/bin/python training/notes/pilot.py --papers 1     # one paper first
    .venv/bin/python training/notes/pilot.py                # all 5 (resumes: done papers are skipped)
    .venv/bin/python training/notes/viewer.py               # -> training/notes/out/index.html

Cost: about 70 Opus calls per paper (5 notes, 5 checks, up to 5 fixes and 5 second checks,
10 texts, 10 traces, 30 judge calls at low effort).

Each round of changes to the prompts writes to its own folder, out/round<N>/, so rounds can be
compared; README.md says what each round changed and what it showed.
"""
from __future__ import annotations

import json
import random
import re
import statistics as st
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lm15

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))
sys.path.insert(0, str(ROOT / "paper_corpus"))
import prompts as PR                                # noqa: E402
import render                                       # noqa: E402
import readability as RD                            # noqa: E402
from eval_v3 import numbers                         # noqa: E402
from usage import claude_usage                      # noqa: E402

# --------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------

V3 = ROOT / "training/v3/pilot50"                   # where the papers, old and v3 answers come from
ROUND = 3                                           # bump when prompts.py changes; results go to out/round<N>
OUT = HERE / "out" / f"round{ROUND}"
N_PAPERS = 5
SEED = 2026
STOP_AT_WEEKLY_PERCENT = 85
LEVELS = ("level1", "level3")
READER_SHORT = "a curious 14-year-old"

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------

CALLS = Counter()
_lock = threading.Lock()


def ask(fn, kind: str, **inputs):
    """One model call, counted; a busy or rate-limited server is waited out and asked again."""
    for attempt in range(8):
        with _lock:
            CALLS[kind] += 1
        try:
            return fn(**inputs)
        except lm15.RateLimitError:
            time.sleep(300)
        except Exception as error:                                         # noqa: BLE001
            busy = "Overloaded" in str(error) or "529" in str(error) or type(error).__name__ in ("ServerError", "TransportError")
            if not busy or attempt == 7:
                raise
            time.sleep(30 * (attempt + 1))
    raise RuntimeError("the model kept failing")


def choose_papers(n: int) -> list[dict]:
    """n papers of the v3 pilot, each from a different subfield, in a fixed random order."""
    records = [json.loads(f.read_text()) for f in sorted(V3.glob("PMC*.json"))]
    random.Random(SEED).shuffle(records)
    chosen, fields = [], set()
    for r in records:
        if r["subfield"] not in fields and len(r["conversations"]) == 5:
            chosen.append(r)
            fields.add(r["subfield"])
    return chosen[:n]


def glossary_for(record: dict, text: str) -> str:
    """The paper's reference entries (offline Wikipedia) for terms that appear in this part."""
    out = []
    for t in record["terms"]:
        if t.get("kind") == "content" and t.get("reference") and re.search(rf"\b{re.escape(t['term'])}\b", text):
            out.append(f"- {t['term']}" + (f" ({t['stands_for']})" if t.get("stands_for") else "")
                       + f": {t['reference'][:400]}")
    return "\n".join(out + ["", "Statistics sheet:", PR.STATISTICS_SHEET])


def judge(original, before_a, a, before_b, b) -> dict:
    """The fair blind judge, asked in both orders. A win must hold both ways."""
    one = ask(PR.COMPARE, "judge", original=original, read_before_a=before_a, version_a=a,
              read_before_b=before_b, version_b=b, reader=READER_SHORT)
    two = ask(PR.COMPARE, "judge", original=original, read_before_a=before_b, version_a=b,
              read_before_b=before_a, version_b=a, reader=READER_SHORT)
    first = {"A": "first", "B": "second"}.get(one.winner, "tie")
    second = {"A": "second", "B": "first"}.get(two.winner, "tie")
    return {"winner": first if first == second and first != "tie" else "no clear winner",
            "reasons_in_order": one.reasons, "reasons_swapped": two.reasons}


def numbers_kept(original: str, text: str) -> float:
    o = numbers(original)
    return round(len(o & numbers(text)) / len(o), 3) if o else 1.0


# --------------------------------------------------------------------------------------------
# One paper
# --------------------------------------------------------------------------------------------

def do_paper(record: dict) -> None:
    pid = record["paper_id"]
    convs = record["conversations"]                    # opening first, then the sections
    whole = "\n\n".join(c["original"] for c in convs)
    parts = [{"section": c["section"], "original": c["original"], "old": c["old_answer"], "v3": c["filled"],
              "glossary": glossary_for(record, c["original"])} for c in convs]

    with ThreadPoolExecutor(len(parts) * 2) as pool:
        # 1. notes (all parts at once: they do not depend on each other)
        def notes_of(p):
            n = ask(PR.MAKE_NOTES, "notes", original=p["original"], rest_of_paper=whole,
                    glossary=p["glossary"], reader=PR.READER)
            return n.model_dump()
        for p, n in zip(parts, pool.map(notes_of, parts)):
            p["notes"] = n
            p["inputs"] = {"level1": render.level1(n), "level3": render.level3(n, seed=f"{pid}{p['section']}")}

        # 2. check the notes against the paper; if anything is wrong, fix them and check again
        def checked(p):
            first = [x.model_dump() for x in ask(PR.CHECK_NOTES, "check notes", original=p["original"],
                                                 notes=render.full(p["notes"]), rest_of_paper=whole).problems]
            if not first:
                return p["notes"], first, []
            problems = "\n".join(f"- {x['kind']} ({x['severity']}): {x['detail']} [{x['quote']}]" for x in first)
            fixed = ask(PR.FIX_NOTES, "fix notes", original=p["original"], notes=json.dumps(p["notes"], ensure_ascii=False),
                        problems=problems, rest_of_paper=whole).model_dump()
            second = [x.model_dump() for x in ask(PR.CHECK_NOTES, "check notes", original=p["original"],
                                                  notes=render.full(fixed), rest_of_paper=whole).problems]
            return fixed, first, second
        for p, (notes, first, second) in zip(parts, pool.map(checked, parts)):
            p["notes_before_fix"], p["notes"] = p["notes"], notes
            p["notes_check_before_fix"], p["notes_check"] = first, second
            p["inputs"] = {"level1": render.level1(notes), "level3": render.level3(notes, seed=f"{pid}{p['section']}")}

        # 3. write, at both levels: the opening first, then the 4 sections with it as already read
        def write(p, level, already_read):
            return ask(PR.WRITE, "write", notes=p["inputs"][level], already_read=already_read,
                       brief=PR.BRIEF, level=PR.LEVEL_1 if level == "level1" else PR.LEVEL_3,
                       reader=PR.READER).text
        for p in parts:
            p["texts"] = {}
        openings = dict(zip(LEVELS, pool.map(lambda lv: write(parts[0], lv, ""), LEVELS)))
        parts[0]["texts"].update(openings)
        jobs = [(p, lv) for p in parts[1:] for lv in LEVELS]
        for (p, lv), text in zip(jobs, pool.map(lambda j: write(j[0], j[1], openings[j[1]]), jobs)):
            p["texts"][lv] = text

        # 4. trace each text back to its notes
        jobs = [(p, lv) for p in parts for lv in LEVELS]
        traced = pool.map(lambda j: ask(PR.TRACE, "trace", notes=render.full(j[0]["notes"]),
                                        text=j[0]["texts"][j[1]], reader=PR.READER), jobs)
        for p in parts:
            p["trace"] = {}
        for (p, lv), t in zip(jobs, traced):
            p["trace"][lv] = t.model_dump()

        # 5. the fair blind judge: each version is read after its own opening
        def before(version, p):
            if p is parts[0]:
                return ""
            return parts[0]["texts"][version] if version in LEVELS else parts[0][version]
        def text_of(version, p):
            return p["texts"][version] if version in LEVELS else p[version]
        pairs = [("level1", "old"), ("level3", "old"), ("v3", "old")]
        jobs = [(p, a, b) for p in parts for a, b in pairs]
        judged = pool.map(lambda j: judge(j[0]["original"], before(j[1], j[0]), text_of(j[1], j[0]),
                                          before(j[2], j[0]), text_of(j[2], j[0])), jobs)
        for p in parts:
            p["judge"] = {}
        for (p, a, b), r in zip(jobs, judged):
            r["winner"] = {"first": a, "second": b}.get(r["winner"], r["winner"])
            p["judge"][f"{a} vs {b}"] = r

    # mechanical measures, same for every version
    for p in parts:
        versions = {"old": p["old"], "v3": p["v3"], **p["texts"]}
        p["numbers_kept"] = {v: numbers_kept(p["original"], t) for v, t in versions.items()}
        p["readability"] = {v: RD.measure(t) for v, t in versions.items()}

    out = {"paper_id": pid, "title": record["title"], "subfield": record["subfield"],
           "renamed": bool(record["renamed"]), "parts": parts}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{pid}.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    tmp.replace(OUT / f"{pid}.json")


# --------------------------------------------------------------------------------------------
# The report (over every finished paper)
# --------------------------------------------------------------------------------------------

def report() -> None:
    parts = [p for f in sorted(OUT.glob("PMC*.json")) for p in json.loads(f.read_text())["parts"]]
    if not parts:
        return
    s = {"parts": len(parts), "judge": {}, "notes_check": {}, "trace": {}, "numbers_kept": {}, "readability": {}}
    for pair in parts[0]["judge"]:
        s["judge"][pair] = dict(Counter(p["judge"][pair]["winner"] for p in parts))
    s["notes_check_before_fix"] = dict(Counter(f"{x['kind']} ({x['severity']})" for p in parts
                                               for x in p.get("notes_check_before_fix", p["notes_check"])))
    s["notes_check"] = dict(Counter(f"{x['kind']} ({x['severity']})" for p in parts for x in p["notes_check"]))
    s["facts_in_notes"] = sum(len(p["notes"]["facts"]) for p in parts)
    for lv in LEVELS:
        statuses = Counter(x["status"] for p in parts for x in p["trace"][lv]["facts"])
        additions = Counter(x["kind"] for p in parts for x in p["trace"][lv]["additions"])
        s["trace"][lv] = {**statuses, **{f"addition: {k}": v for k, v in additions.items()}}
    for v in ("old", "v3", *LEVELS):
        s["numbers_kept"][v] = round(st.mean(p["numbers_kept"][v] for p in parts), 3)
        r = RD.measure("\n\n".join(p["texts"][v] if v in LEVELS else p[v] for p in parts))
        s["readability"][v] = {k: round(r[k], 1) for k in ("words", "sent_mean", "sent_over_30", "para_mean", "fk_grade")}
    s["calls_this_run"] = dict(CALLS)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def main():
    n = int(sys.argv[sys.argv.index("--papers") + 1]) if "--papers" in sys.argv else N_PAPERS
    todo = [r for r in choose_papers(N_PAPERS)[:n] if not (OUT / f"{r['paper_id']}.json").exists()]
    print(f"{n} papers chosen, {len(todo)} to do", flush=True)
    for r in todo:                                     # one paper at a time: each already runs ~10 calls at once
        weekly = (claude_usage() or {}).get("seven_day") or 0
        if weekly >= STOP_AT_WEEKLY_PERCENT:
            print(f"stop: weekly allowance at {weekly}%", flush=True)
            break
        try:
            do_paper(r)
            print(f"done {r['paper_id']} ({r['subfield']}) · calls so far {sum(CALLS.values())}", flush=True)
        except Exception as error:                                         # noqa: BLE001
            print(f"FAILED {r['paper_id']}: {type(error).__name__}: {str(error)[:300]}", flush=True)
    report()


if __name__ == "__main__":
    main()
