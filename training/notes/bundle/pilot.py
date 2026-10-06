"""
Writing from a dense bundle · round 4
=====================================

Same question as rounds 1-3 (../README.md): can a writer that never sees the paper write it
well for a curious 14-year-old, given the knowledge? Round 4 changes what it is given: one dense,
structured bundle per paper instead of sentence-like notes per part.

For each paper:
  1. copy title, abstract, figure captions and tables word for word (sources.py)
  2. Opus, as the scientist: the paper's things and glossary (1 call)
  3. Opus, as the scientist: each part as dense facts and a paragraph plan (5 calls)
  4. a checker compares each part of the bundle with the paper; Opus fixes; checked again
  5. Opus, as the writer, writes each part from the whole bundle (opening first)
  6. code checks the form (placeholders, citations, prose only); trace each part's text back
     to its facts; fair judge (sees the whole paper; from 4c, prose only):
       4a-4b: bundle vs old answer, bundle vs round 3 level 1, round 3 level 1 vs old answer
       4c:    bundle vs old answer, bundle vs 4b

Same 5 papers as round 3, so all three can be compared on the same parts with the same judge.

    .venv/bin/python training/notes/bundle/pilot.py --papers 1
    .venv/bin/python training/notes/bundle/pilot.py
    .venv/bin/python training/notes/bundle/viewer.py     # -> training/notes/bundle/out/<try>/index.html

Cost: about 50-60 Opus calls per paper (1 front, 5 parts, 5-15 checks and fixes, 5 texts,
5 traces, 30 judge calls at low effort).
"""
from __future__ import annotations

import json
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))
sys.path.insert(0, str(ROOT / "paper_corpus"))
import prompts as PR                                # noqa: E402  (this folder's prompts.py)
import render                                       # noqa: E402  (this folder's render.py)
from sources import verbatim                        # noqa: E402
import readability as RD                            # noqa: E402
from eval_v3 import numbers                         # noqa: E402
from usage import claude_usage                      # noqa: E402

V3 = ROOT / "training/v3/pilot50"                   # papers, old answers
PREVIOUS = "4b"                                     # the try this one is compared with
TRY = "4c"                                          # bump when prompts.py changes; results go to out/<TRY>
OUT = HERE / "out" / TRY
PAPERS = ["PMC12291832", "PMC12010427", "PMC7616100", "PMC12466960", "PMC11227166"]   # round 3's
STOP_AT_WEEKLY_PERCENT = 85
READER_SHORT = "a curious 14-year-old"

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


def words(text: str) -> int:
    return len(text.split())


def numbers_kept(original: str, text: str) -> float:
    o = numbers(original)
    return round(len(o & numbers(text)) / len(o), 3) if o else 1.0


def opening_body(opening: str) -> str:
    """The opening without its title and abstract (those go into the bundle word for word)."""
    keep = re.findall(r"^## (introduction_first|conclusion)\s*\n(.*?)(?=^## |\Z)", opening, re.S | re.M)
    return "\n\n".join(f"## {label}\n\n{body.strip()}" for label, body in keep if body.strip())


def reference_glossary(record: dict) -> str:
    return "\n".join(f"- {t['term']}" + (f" ({t['stands_for']})" if t.get("stands_for") else "") + f": {t['reference'][:300]}"
                     for t in record["terms"] if t.get("kind") == "content" and t.get("reference"))


PLACEHOLDER = re.compile(r"^\[([FT][A-Z]?\d+)\][ \t]*$", re.M)
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def cited(text: str) -> set[int]:
    """Reference numbers cited in text. The writer writes [3] or [3,5]; the notes sometimes
    write ranges ([66-69]), which count as every number in the range."""
    out = {int(x) for m in CITATION.findall(text) for x in m.split(",")}
    for a, b in re.findall(r"\[(\d+)\s*[-–]\s*(\d+)\]", text):
        out |= set(range(int(a), int(b) + 1))
    return out


def form_checks(text: str, part: dict, expected: list[str], n_refs: int) -> dict:
    """What code can check without a model: every placeholder of this part once, on its own
    line, and no other; citations valid and the same as in the part's facts; prose only."""
    found = PLACEHOLDER.findall(text)
    inline = [m for m in re.findall(r"\[([FT][A-Z]?\d+)\]", text) if m not in found]
    facts_cited = cited(" ".join(f["note"] for f in part["facts"]))
    text_cited = cited(PLACEHOLDER.sub("", text))
    return {"placeholders_missing": [x for x in expected if x not in found],
            "placeholders_extra": [x for x in found if x not in expected],
            "placeholders_twice": sorted({x for x in found if found.count(x) > 1}),
            "placeholders_inline": inline,
            "citations_missing": sorted(facts_cited - text_cited),
            "citations_not_in_facts": sorted(text_cited - facts_cited),
            "citations_invalid": sorted(x for x in text_cited if not 1 <= x <= n_refs),
            "author_year_citations": re.findall(r"\([A-Z][\w-]+(?: et al\.)?,? (?:19|20)\d\d[a-z]?\)", text),
            "lists": len(re.findall(r"^\s*([-*•]|\d+\.)\s", text, re.M)),
            "tables": len(re.findall(r"^\s*\|", text, re.M)),
            "bold": len(re.findall(r"\*\*", text)) // 2}


def for_judge(text: str) -> str:
    """Placeholders as the judge should read them (the app shows the original figure or table)."""
    return PLACEHOLDER.sub(lambda m: f"[{'Table' if m.group(1)[0] == 'T' else 'Figure'} {m.group(1)[1:]} is shown here, "
                                     "as in the original paper]", text)


def prose(text: str) -> str:
    """For readability measures: without placeholders and citation brackets."""
    return re.sub(r"\s?" + CITATION.pattern, "", PLACEHOLDER.sub("", text))


def judge(original, rest, figures, before_a, a, before_b, b) -> dict:
    """The fair judge, both orders; a win must hold both ways."""
    one = ask(PR.COMPARE, "judge", original=original, rest_of_paper=rest, figures_and_tables=figures,
              read_before_a=before_a, version_a=a, read_before_b=before_b, version_b=b, reader=READER_SHORT)
    two = ask(PR.COMPARE, "judge", original=original, rest_of_paper=rest, figures_and_tables=figures,
              read_before_a=before_b, version_a=b, read_before_b=before_a, version_b=a, reader=READER_SHORT)
    first = {"A": "first", "B": "second"}.get(one.winner, "tie")
    second = {"A": "second", "B": "first"}.get(two.winner, "tie")
    return {"winner": first if first == second and first != "tie" else "no clear winner",
            "reasons_in_order": one.reasons, "reasons_swapped": two.reasons}


def do_paper(pid: str) -> None:
    record = json.loads((V3 / f"{pid}.json").read_text())
    previous = {p["section"]: p for p in json.loads((HERE / "out" / PREVIOUS / f"{pid}.json").read_text())["parts"]}
    originals = {c["section"]: c["original"] for c in record["conversations"]}
    olds = {c["section"]: c["old_answer"] for c in record["conversations"]}
    names = [n for n in render.PARTS if n in originals]
    whole = "\n\n".join(originals[n] for n in names)
    v = verbatim(record)
    sheet_text = render.sheets(whole)
    figs = lambda n: "\n\n".join(render._figure(f) for f in v["figures"] if f["part"] == n) or "(none)"
    all_figures = "\n\n".join(render._figure(f) for f in v["figures"]) or "(none)"

    # 1-2. the front: things and glossary
    front = ask(PR.MAKE_FRONT, "front", paper=whole, figures_and_tables="\n\n".join(
        render._figure(f) for f in v["figures"]), reference_glossary=reference_glossary(record) + "\n\nStatistics sheet:\n"
        + PR.STATISTICS_SHEET, reader=PR.READER).model_dump()
    things = "\n".join(f"- {t['name']}: {t['what']}" for t in front["things"])

    with ThreadPoolExecutor(10) as pool:
        # 3. each part: dense facts and plan
        def part_of(n):
            original = opening_body(originals[n]) if n == "opening" else originals[n]
            rest = "\n\n".join(originals[m] for m in names if m != n)
            return ask(PR.MAKE_PART, "part", part_name=n, id_letter=render.LETTER[n], original=original,
                       things=things, figures_and_tables_here=figs(n), references=render._references(v),
                       rest_of_paper=rest).model_dump()
        parts = dict(zip(names, pool.map(part_of, names)))

        # 4. check, fix, check again
        def checked(n):
            rest = "\n\n".join(originals[m] for m in names if m != n)
            first = [x.model_dump() for x in ask(PR.CHECK_PART, "check", original=originals[n],
                     bundle_part=render.part_view(v, front, n, parts[n]), rest_of_paper=rest).problems]
            if not first:
                return parts[n], first, []
            problems = "\n".join(f"- {x['kind']} ({x['severity']}): {x['detail']} [{x['quote']}]" for x in first)
            fixed = ask(PR.FIX_PART, "fix", original=originals[n], part=json.dumps(parts[n], ensure_ascii=False),
                        problems=problems, things=things, rest_of_paper=rest).model_dump()
            second = [x.model_dump() for x in ask(PR.CHECK_PART, "check", original=originals[n],
                      bundle_part=render.part_view(v, front, n, fixed), rest_of_paper=rest).problems]
            return fixed, first, second
        checks = dict(zip(names, pool.map(checked, names)))
        before_fix = {n: parts[n] for n in names}
        parts = {n: checks[n][0] for n in names}
        text_of_bundle = render.bundle(v, front, parts, sheet_text)

        # 5. write: the opening first, then the sections with it as already read
        def write(n, already):
            return ask(PR.WRITE, "write", bundle=text_of_bundle, part_name=n, already_read=already,
                       brief=PR.BRIEF, reader=PR.READER).text
        texts = {"opening": write("opening", "")}
        texts.update(dict(zip(names[1:], pool.map(lambda n: write(n, texts["opening"]), names[1:]))))

        # 6. trace and judge
        traced = dict(zip(names, pool.map(lambda n: ask(PR.TRACE, "trace", items=render.trace_view(v, n, parts[n]),
                                                          text=texts[n], allowed_sources=text_of_bundle,
                                                          reader=PR.READER).model_dump(), names)))
        versions = {"bundle": texts, PREVIOUS: {n: previous[n]["texts"]["bundle"] for n in names}, "old": olds}
        pairs = [("bundle", "old"), ("bundle", PREVIOUS)]
        jobs = [(n, a, b) for n in names for a, b in pairs]

        def judged(job):
            n, a, b = job
            before = lambda x: "" if n == "opening" else versions[x]["opening"]
            rest = "\n\n".join(originals[m] for m in names if m != n)
            r = judge(originals[n], rest, all_figures, for_judge(before(a)), for_judge(versions[a][n]),
                      for_judge(before(b)), for_judge(versions[b][n]))
            r["winner"] = {"first": a, "second": b}.get(r["winner"], r["winner"])
            return r
        verdicts = list(pool.map(judged, jobs))

    out_parts = []
    for n in names:
        out_parts.append({
            "section": n, "original": originals[n], "notes_before_fix": before_fix[n], "notes": parts[n],
            "check_before_fix": checks[n][1], "check": checks[n][2],
            "texts": {k: versions[k][n] for k in versions},
            "form": form_checks(texts[n], parts[n], [f["id"] for f in v["figures"] if f["part"] == n], len(v["references"])),
            "trace": traced[n],
            "judge": {f"{a} vs {b}": r for (m, a, b), r in zip(jobs, verdicts) if m == n},
            "numbers_of_notes_kept": numbers_kept(" ".join(f["note"] for f in parts[n]["facts"]), texts[n]),
            "readability": {k: RD.measure(prose(versions[k][n])) for k in versions},
            "density": {"original_words": words(originals[n]),
                        "part_words": words(render.outline_of(n, parts[n], []))}})
    out = {"paper_id": pid, "title": record["title"], "subfield": record["subfield"], "renamed": bool(record["renamed"]),
           "verbatim": v, "front": front, "bundle": text_of_bundle,
           "density": {"paper_words": words(whole), "bundle_words": words(text_of_bundle),
                       "bundle_words_without_sheets": words(text_of_bundle) - words(sheet_text)},
           "parts": out_parts}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{pid}.json.tmp").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (OUT / f"{pid}.json.tmp").replace(OUT / f"{pid}.json")


def report() -> None:
    papers = [json.loads(f.read_text()) for f in sorted(OUT.glob("PMC*.json"))]
    parts = [p for d in papers for p in d["parts"]]
    if not parts:
        return
    s = {"papers": len(papers), "parts": len(parts), "judge": {}, "trace": {}}
    for pair in parts[0]["judge"]:
        s["judge"][pair] = dict(Counter(p["judge"][pair]["winner"] for p in parts))
    s["density"] = {"paper_words": sum(d["density"]["paper_words"] for d in papers),
                    "bundle_words": sum(d["density"]["bundle_words"] for d in papers),
                    "bundle_words_without_sheets": sum(d["density"]["bundle_words_without_sheets"] for d in papers),
                    "facts": sum(len(p["notes"]["facts"]) for p in parts)}
    s["check_before_fix"] = dict(Counter(f"{x['kind']} ({x['severity']})" for p in parts for x in p["check_before_fix"]))
    s["check"] = dict(Counter(f"{x['kind']} ({x['severity']})" for p in parts for x in p["check"]))
    s["trace"] = {**Counter(x["status"] for p in parts for x in p["trace"]["facts"]),
                  **{f"addition: {k}": v for k, v in Counter(x["kind"] for p in parts for x in p["trace"]["additions"]).items()}}
    s["numbers_of_notes_kept"] = round(st.mean(p["numbers_of_notes_kept"] for p in parts), 3)
    form = Counter()
    for p in parts:
        form.update({k: (len(v) if isinstance(v, list) else v) for k, v in p["form"].items()})
    s["form"] = dict(form)
    s["readability"] = {}
    for k in parts[0]["texts"]:
        r = RD.measure("\n\n".join(prose(p["texts"][k]) for p in parts))
        s["readability"][k] = {x: round(r[x], 1) for x in ("words", "sent_mean", "sent_over_30", "para_mean", "fk_grade")}
    s["calls_this_run"] = dict(CALLS)
    (OUT / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def main():
    n = int(sys.argv[sys.argv.index("--papers") + 1]) if "--papers" in sys.argv else len(PAPERS)
    todo = [p for p in PAPERS[:n] if not (OUT / f"{p}.json").exists()]
    print(f"{n} papers, {len(todo)} to do", flush=True)
    for pid in todo:
        weekly = (claude_usage() or {}).get("seven_day") or 0
        if weekly >= STOP_AT_WEEKLY_PERCENT:
            print(f"stop: weekly allowance at {weekly}%", flush=True)
            break
        try:
            do_paper(pid)
            print(f"done {pid} · calls so far {sum(CALLS.values())}", flush=True)
        except Exception as error:                                         # noqa: BLE001
            print(f"FAILED {pid}: {type(error).__name__}: {str(error)[:300]}", flush=True)
    report()


if __name__ == "__main__":
    main()
