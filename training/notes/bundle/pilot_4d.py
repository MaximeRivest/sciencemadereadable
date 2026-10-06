"""
Tries 4d and 4e: code builds the structure, the model only writes the facts
===========================================================================

4e (the current setting) adds to 4d: worked examples for both scientist steps, and a check of
the things and glossary on their own (check_glossary) before new meanings enter the shared
cache. 4e starts from an empty cache (4d's is kept in out/4d/glossary_cache.json) and is
judged against 4d.

Same 5 papers as 4c. What changes is who does what in making the bundle:

  code   (structure.py)  headings, paragraphs in order, citations as [n] from the XML links,
                         where each placeholder goes (after the paragraph that first mentions it)
  code   (cache.py)      plain meanings already written for an earlier paper are reused
  model                  things + only the new glossary words (1 call per paper)
  model                  each numbered paragraph -> dense facts (1 call per part)
  code   (checks.py)     numbers, citations, hedge words of each paragraph kept in its facts?
  model                  fix, where checks found something

To learn whether the model check can be dropped, 4d runs it on every part as well and records
what it finds that code does not. The writer, the trace, the form checks and the judge are the
same as 4c; the judge compares 4d with 4c (prose only), so we see whether the cheaper bundle is
at least as good.

    .venv/bin/python training/notes/bundle/pilot_4d.py [--papers 1]
    .venv/bin/python training/notes/bundle/viewer.py 4d
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cache                                        # noqa: E402
import checks                                       # noqa: E402
import pilot as P                                   # noqa: E402  helpers shared with 4a-4c: ask, judge, form checks
import prompts as PR                                # noqa: E402
import render                                       # noqa: E402
import structure                                    # noqa: E402
from sources import verbatim                        # noqa: E402

TRY, PREVIOUS = "4e", "4d"
OUT = HERE / "out" / TRY
P.OUT = OUT                                         # P.report() summarises this folder


def build_part(name: str, s_part: dict, notes: dict, placed: dict[int, list[str]], unplaced: list[str]) -> dict:
    """The part in the shape render.py reads: headings (from code), facts (with their heading),
    and the plan: each paragraph in order, split into groups of at most 4 facts, followed by the
    placeholders of the figures and tables it first mentions."""
    heading_of = {p["n"]: p["heading"] for p in s_part["paragraphs"]}
    facts, plan, seen = [], [], set()
    for pn in sorted(notes["paragraphs"], key=lambda x: x["paragraph"]):
        h = heading_of.get(pn["paragraph"], s_part["headings"][-1])
        ids = []
        for f in pn["facts"]:
            fid = f["id"] if f["id"] not in seen else f"{f['id']}b"
            seen.add(fid)
            facts.append({**f, "id": fid, "heading": h})
            ids.append(fid)
        groups = max(1, math.ceil(len(ids) / 4))
        size = math.ceil(len(ids) / groups) if ids else 0
        for g in range(groups):
            items = ids[g * size:(g + 1) * size]
            if items or g == 0:
                plan.append({"heading": h, "purpose": pn["purpose"] + (" (continued)" if g else ""), "items": items})
        plan[-1]["items"] += placed.get(pn["paragraph"], [])
    if unplaced and plan:
        plan[-1]["items"] += unplaced
    return {"headings": s_part["headings"], "facts": facts, "plan": plan}


def do_paper(pid: str, gloss: dict) -> dict:
    record = json.loads((P.V3 / f"{pid}.json").read_text())
    previous = {p["section"]: p for p in json.loads((HERE / "out" / PREVIOUS / f"{pid}.json").read_text())["parts"]}
    originals = {c["section"]: c["original"] for c in record["conversations"]}
    olds = {c["section"]: c["old_answer"] for c in record["conversations"]}
    v = verbatim(record)
    s = structure.paper_structure(record)
    names = [n for n in render.PARTS if n in originals and s["parts"].get(n, {}).get("paragraphs")]
    for f in v["figures"]:                                          # placement by code: first mention
        if f["id"] in s["placement"]:
            f["part"] = s["placement"][f["id"]][0]
    texts_of = {n: structure.part_text(s["parts"][n]) for n in names}
    whole = "\n\n".join(texts_of.values())
    all_figures = "\n\n".join(render._figure(f) for f in v["figures"]) or "(none)"
    figs = lambda n: "\n\n".join(render._figure(f) for f in v["figures"] if f["part"] == n) or "(none)"

    # 1. front: things, new glossary words, which words are central (cache for the rest)
    known = cache.known_in(gloss, v["title"] + " " + v["abstract"] + " " + whole)
    fr = P.ask(PR.MAKE_FRONT_CACHED, "front", paper=whole, figures_and_tables=all_figures,
               already_defined="\n".join(f"- {e['term']}: {e['meaning']}" for e in known) or "(none)",
               reference_glossary=P.reference_glossary(record) + "\n\nStatistics sheet:\n" + PR.STATISTICS_SHEET,
               reader=PR.READER, example=PR.EXAMPLE_FRONT).model_dump()
    new_terms = {d["term"].lower() for d in fr["glossary"]}
    central = {c["term"].lower(): c["picture"] for c in fr["central_known"]}
    glossary = [{"term": e["term"], "meaning": e["meaning"], "central": e["term"].lower() in central,
                 "picture": central.get(e["term"].lower(), "")} for e in known if e["term"].lower() not in new_terms]
    glossary += fr["glossary"]
    things_list = fr["things"]

    # 1b. check things and glossary on their own; corrections apply before anything is cached
    gloss_text = "\n".join(f"- {d['term']}: {d['meaning']}" + (f" (picture: {d['picture']})" if d["picture"] else "")
                           for d in glossary)
    gcheck = [x.model_dump() for x in P.ask(PR.CHECK_GLOSSARY, "check glossary", paper=whole,
              things="\n".join(f"- {t['name']}: {t['what']}" for t in things_list), glossary=gloss_text,
              reader=PR.READER).problems]
    for x in gcheck:
        key = x["entry"].strip().lower()
        if x["field"] == "thing":
            for t in things_list:
                if t["name"].lower() == key:
                    t["what"] = x["correction"]
            things_list = [t for t in things_list if t["what"]]
        else:
            for d in glossary:
                if d["term"].lower() == key:
                    d["meaning" if x["field"] == "meaning" else "picture"] = x["correction"]
            glossary = [d for d in glossary if d["meaning"]]
    checked_new = [d for d in glossary if d["term"].lower() in new_terms]
    front = {"things": things_list, "glossary": glossary}
    things = "\n".join(f"- {t['name']}: {t['what']}" for t in front["things"])
    reused, added = len(glossary) - len(checked_new), cache.add(gloss, checked_new, pid)

    with ThreadPoolExecutor(10) as pool:
        # 2. each paragraph -> dense facts
        def notes_of(n):
            rest = "\n\n".join(texts_of[m] for m in names if m != n)
            return P.ask(PR.MAKE_PARAGRAPH_NOTES, "notes", part_name=n, id_letter=render.LETTER[n], original=texts_of[n],
                         things=things, figures_and_tables_here=figs(n), rest_of_paper=rest,
                         example=PR.EXAMPLE_PARAGRAPHS).model_dump()
        notes = dict(zip(names, pool.map(notes_of, names)))

        def assembled(n, nt):
            placed = {}
            for fid, (part, para) in s["placement"].items():
                if part == n:
                    placed.setdefault(para, []).append(fid)
            unplaced = [f["id"] for f in v["figures"] if f["part"] == n and f["id"] not in s["placement"]]
            return build_part(n, s["parts"][n], nt, placed, unplaced)

        # 3. checks: code everywhere; the model check everywhere too (4d only, to compare)
        def checked(n):
            rest = "\n\n".join(texts_of[m] for m in names if m != n)
            code = checks.part_problems(s["parts"][n], notes[n])
            model = [x.model_dump() for x in P.ask(PR.CHECK_PART, "check", original=texts_of[n],
                     bundle_part=render.part_view(v, front, n, assembled(n, notes[n])), rest_of_paper=rest).problems]
            if not code and not model:
                return notes[n], code, model, []
            problems = "\n".join([f"- ¶{x['paragraph']}: {x['kind']}: {x['detail']}" for x in code]
                                 + [f"- {x['kind']} ({x['severity']}): {x['detail']} [{x['quote']}]" for x in model])
            fixed = P.ask(PR.FIX_PARAGRAPH_NOTES, "fix", original=texts_of[n], notes=json.dumps(notes[n], ensure_ascii=False),
                          problems=problems, things=things, rest_of_paper=rest).model_dump()
            return fixed, code, model, checks.part_problems(s["parts"][n], fixed)
        results = dict(zip(names, pool.map(checked, names)))
        before_fix = {n: assembled(n, notes[n]) for n in names}
        parts = {n: assembled(n, results[n][0]) for n in names}
        text_of_bundle = render.bundle(v, front, parts, render.sheets(whole))

        # 4. write, trace, judge (as in 4c)
        def write(n, already):
            return P.ask(PR.WRITE, "write", bundle=text_of_bundle, part_name=n, already_read=already,
                         brief=PR.BRIEF, reader=PR.READER).text
        texts = {names[0]: write(names[0], "")}
        texts.update(dict(zip(names[1:], pool.map(lambda n: write(n, texts[names[0]]), names[1:]))))
        traced = dict(zip(names, pool.map(lambda n: P.ask(PR.TRACE, "trace", items=render.trace_view(v, n, parts[n]),
                                                            text=texts[n], allowed_sources=text_of_bundle,
                                                            reader=PR.READER).model_dump(), names)))
        versions = {"bundle": texts, PREVIOUS: {n: previous[n]["texts"]["bundle"] for n in names}, "old": olds}

        def judged(n):
            before = lambda x: "" if n == names[0] else versions[x][names[0]]
            rest = "\n\n".join(originals[m] for m in names if m != n)
            r = P.judge(originals[n], rest, all_figures, P.for_judge(before("bundle")), P.for_judge(texts[n]),
                        P.for_judge(before(PREVIOUS)), P.for_judge(versions[PREVIOUS][n]))
            r["winner"] = {"first": "bundle", "second": PREVIOUS}.get(r["winner"], r["winner"])
            return r
        verdicts = dict(zip(names, pool.map(judged, names)))

    out_parts = []
    for n in names:
        fixed, code, model, code_after = results[n]
        out_parts.append({
            "section": n, "original": originals[n], "paper_text": texts_of[n],
            "notes_before_fix": before_fix[n], "notes": parts[n],
            "code_check": code, "code_check_after_fix": code_after,
            "check_before_fix": model, "check": [],                       # the model check ran once, before the fix
            "texts": {k: versions[k][n] for k in versions},
            "form": P.form_checks(texts[n], parts[n], [f["id"] for f in v["figures"] if f["part"] == n], len(v["references"])),
            "trace": traced[n],
            "judge": {f"bundle vs {PREVIOUS}": verdicts[n]},
            "numbers_of_notes_kept": P.numbers_kept(" ".join(f["note"] for f in parts[n]["facts"]), texts[n]),
            "readability": {k: P.RD.measure(P.prose(versions[k][n])) for k in versions},
            "density": {"original_words": P.words(texts_of[n]), "part_words": P.words(render.outline_of(n, parts[n], []))}})
    out = {"paper_id": pid, "title": record["title"], "subfield": record["subfield"], "renamed": bool(record["renamed"]),
           "verbatim": v, "front": front, "bundle": text_of_bundle, "structure": s,
           "glossary_cache": {"reused": reused, "new": len(fr["glossary"]), "added_to_cache": added},
           "glossary_check": gcheck,
           "density": {"paper_words": P.words(whole), "bundle_words": P.words(text_of_bundle),
                       "bundle_words_without_sheets": P.words(text_of_bundle) - P.words(render.sheets(whole))},
           "parts": out_parts}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{pid}.json.tmp").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (OUT / f"{pid}.json.tmp").replace(OUT / f"{pid}.json")
    return out


def report_4d() -> None:
    """The shared report, plus what 4d is about: code checks vs model check, cache reuse, cost."""
    P.report()
    papers = [json.loads(f.read_text()) for f in sorted(OUT.glob("PMC*.json"))]
    parts = [p for d in papers for p in d["parts"]]
    code_flag = [bool(p["code_check"]) for p in parts]
    model_major = [any(x["severity"] == "major" for x in p["check_before_fix"]) for p in parts]
    s = json.loads((OUT / "summary.json").read_text())
    s["code_vs_model_check"] = {
        "parts": len(parts),
        "code found something": sum(code_flag),
        "model found something": sum(bool(p["check_before_fix"]) for p in parts),
        "model found a major problem": sum(model_major),
        "major problems in parts code did not flag": sum(m and not c for m, c in zip(model_major, code_flag)),
        "code problems by kind": dict(Counter(x["kind"] for p in parts for x in p["code_check"])),
        "code problems left after the fix": sum(len(p["code_check_after_fix"]) for p in parts),
        "model problems by kind": dict(Counter(f"{x['kind']} ({x['severity']})" for p in parts for x in p["check_before_fix"]))}
    s["glossary_cache"] = {k: sum(d["glossary_cache"][k] for d in papers) for k in ("reused", "new", "added_to_cache")}
    s["glossary_check"] = dict(Counter(f"{x['field']}: {'removed' if not x['correction'] else 'corrected'}"
                                       for d in papers for x in d.get("glossary_check", [])))
    n = len(papers)
    s["calls_per_paper_in_production"] = {
        "front": 1, "notes": round(len(parts) / n, 1), "fix (where code flags)": round(sum(code_flag) / n, 1),
        "write": round(len(parts) / n, 1), "trace": round(len(parts) / n, 1),
        "check glossary": 1,
        "total": round(2 + 3 * len(parts) / n + sum(code_flag) / n, 1),
        "4c, for comparison": "about 30 (front 1, parts 5, checks and fixes about 14, write 5, trace 5)"}
    (OUT / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(json.dumps({k: s[k] for k in ("code_vs_model_check", "glossary_cache", "glossary_check",
                                          "calls_per_paper_in_production")},
                     indent=1, ensure_ascii=False))


def main():
    n = int(sys.argv[sys.argv.index("--papers") + 1]) if "--papers" in sys.argv else len(P.PAPERS)
    gloss = cache.load()
    todo = [p for p in P.PAPERS[:n] if not (OUT / f"{p}.json").exists()]
    print(f"{n} papers, {len(todo)} to do; glossary cache has {len(gloss)} words", flush=True)
    for pid in todo:                                                   # in order: later papers reuse the cache
        weekly = (P.claude_usage() or {}).get("seven_day") or 0
        if weekly >= P.STOP_AT_WEEKLY_PERCENT:
            print(f"stop: weekly allowance at {weekly}%", flush=True)
            break
        try:
            d = do_paper(pid, gloss)
            cache.save(gloss)
            print(f"done {pid} · glossary {d['glossary_cache']} · calls so far {sum(P.CALLS.values())}", flush=True)
        except Exception as error:                                     # noqa: BLE001
            print(f"FAILED {pid}: {type(error).__name__}: {str(error)[:300]}", flush=True)
    report_4d()


if __name__ == "__main__":
    main()
