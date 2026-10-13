"""
Benchmark a new teacher on the 5 test papers
============================================

Runs the full production recipe (produce.do_paper: notes, checks, fixes, writing, trace,
revision) with a teacher on the 5 test papers of rounds 1-4, then compares, per part, its
final text with Opus's text of try 4e for the same part (fair blind judge, Opus, both orders)
and reports what the recipe itself measured: serious problems in the first notes, facts kept
by the writer, outside additions, revisions, calls and minutes.

    .venv/bin/python training/notes/bundle/bench_teachers.py sol
    .venv/bin/python training/notes/bundle/bench_teachers.py muse
Results: training/notes/bundle/out/bench/<teacher>/ and summary-<teacher>.json
"""
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import produce                                       # noqa: E402

import importlib.util                                # noqa: E402
_spec = importlib.util.spec_from_file_location("bundle_pilot", HERE / "pilot.py")
P = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = P
_spec.loader.exec_module(P)

PAPERS = ["PMC12291832", "PMC12010427", "PMC7616100", "PMC12466960", "PMC11227166"]
REFERENCE = HERE / "out" / "4e"                     # Opus, try 4e, same papers


def main(teacher: str):
    out = HERE / "out" / "bench"
    produce.OUT = out
    guard = produce.Guard(teacher)
    gloss, lock = {}, threading.Lock()
    produce.cache.PATH = out / teacher / "glossary_cache.json"
    import dpyr
    meta = {p["paper_id"]: p for p in dpyr.read_parquet(produce.ROOT / "paper_corpus/papers.parquet").to_dicts()}

    def one(pid):
        if (out / teacher / pid / "done.json").exists():
            return
        row = {k: meta[pid][k] for k in ("paper_id", "subfield", "words", "journal", "year")} | {"teacher": teacher}
        try:
            r = produce.do_paper(row, teacher, gloss, lock, guard)
            print(f"done {pid}: {r['calls']} calls, {r['minutes']} min", flush=True)
        except Exception as error:                                     # noqa: BLE001
            print(f"FAILED {pid}: {type(error).__name__}: {str(error)[:300]}", flush=True)

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(one, PAPERS))

    # judge each part against Opus 4e (Opus judge, prose only, both orders)
    rows = []
    for pid in PAPERS:
        folder = out / teacher / pid
        if not (folder / "done.json").exists():
            continue
        texts = json.loads((folder / "texts.json").read_text())
        done = json.loads((folder / "done.json").read_text())
        bundle = json.loads((folder / "bundle.json").read_text())
        ref = {p["section"]: p for p in json.loads((REFERENCE / f"{pid}.json").read_text())["parts"]}
        paper = json.loads((folder / "paper.json").read_text())
        figures = "\n\n".join(produce.render._figure(f) for f in paper["verbatim"]["figures"]) or "(none)"
        names = [n for n in texts if n in ref]
        for n in names:
            log = texts[n]
            tr = log["revised_trace"] if log["final"] == log.get("revised") else log["first_trace"]
            before = lambda t: "" if n == names[0] else t
            j = P.judge(ref[n]["original"], "\n\n".join(ref[m]["original"] for m in names if m != n), figures,
                        P.for_judge(before(texts[names[0]]["final"])), P.for_judge(log["final"]),
                        P.for_judge(before(ref[names[0]]["texts"]["bundle"])), P.for_judge(ref[n]["texts"]["bundle"]))
            rows.append({"paper": pid, "part": n, "judge": {"first": teacher, "second": "opus 4e"}.get(j["winner"], j["winner"]),
                         "reasons": j["reasons_in_order"], "facts": Counter(x["status"] for x in tr["facts"]),
                         "additions": Counter(x["kind"] for x in tr["additions"]), "revised": "revised" in log,
                         "notes_major": any(x["severity"] == "major" for x in bundle["notes"][n]["check_1"]["model"])})
        rows[-1]["paper_calls"], rows[-1]["paper_minutes"] = done["calls"], done["minutes"]
    facts = sum((r["facts"] for r in rows), Counter())
    s = {"teacher": teacher, "papers": len({r["paper"] for r in rows}), "parts": len(rows),
         "judge vs opus 4e": dict(Counter(r["judge"] for r in rows)),
         "writer: facts kept share": round(facts["kept"] / max(1, sum(facts.values())), 3), "writer: facts": dict(facts),
         "writer: additions": dict(sum((r["additions"] for r in rows), Counter())),
         "parts revised": sum(r["revised"] for r in rows),
         "notes: parts with a major problem at first check": sum(r["notes_major"] for r in rows),
         "calls per paper": round(sum(r.get("paper_calls", 0) for r in rows) / max(1, len({r['paper'] for r in rows})), 1),
         "minutes per paper": round(sum(r.get("paper_minutes", 0) for r in rows) / max(1, len({r['paper'] for r in rows})), 1)}
    (out / f"summary-{teacher}.json").write_text(json.dumps({"summary": s, "parts": rows}, indent=1, default=dict))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
