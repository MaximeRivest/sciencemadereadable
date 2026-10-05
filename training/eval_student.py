"""Score a training checkpoint with benchmark v0.3 (the Opus judge and the free checks),
on the first few of its test papers.

    .venv/bin/python training/eval_student.py generate NAME [--papers 3] [--port 8011]
    .venv/bin/python training/eval_student.py judge NAME [--papers 3] [--out FILE.json]

(main venv, from the repository root; normally run by training/bench.sh,
which also starts the checkpoint on the CPU with llama.cpp.)

generate: rewrites the papers with the student, through the same two functai
functions it was trained on (prepare.py), asking for the Opus style. Saved where the
benchmark expects candidates: model_baselines/rewrites/NAME/.
judge: the benchmark's Opus judge on each unit (opening + each section), then the
scores of NAME next to Opus and the untrained Qwen 4B on exactly the same units.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dpyr
import lm15
import tiktoken

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))
import eval_v3                                                     # noqa: E402
import translator                                                  # noqa: E402
import importlib.util                                              # noqa: E402

# rewrite_benchmark has a prepare.py too: load ours by its path
_spec = importlib.util.spec_from_file_location("student_prepare", HERE / "prepare.py")
student_prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(student_prepare)
rewrite_opening_student = student_prepare.rewrite_opening_student
rewrite_section_student = student_prepare.rewrite_section_student

REFERENCES = ["opus", "qwen4b"]
ENC = tiktoken.get_encoding("o200k_base")
sys.path.insert(0, str(ROOT / "glossary"))
from inputs import glossary_text                                   # noqa: E402

OPENING_G, SECTION_G = student_prepare.OPENING_G, student_prepare.SECTION_G


def papers(n):
    return eval_v3.test_papers()[:n]


def generate(name, n, port, use_glossary=False):
    # A CPU answer can take far longer than lm15's default 10-minute read timeout (a 4B
    # model writing a long section while 7 others share the processor): wait up to 3 h.
    from lm15.transports import StdlibTransport
    client = lm15.OpenAIChatLM(api_key="none", base_url=f"http://127.0.0.1:{port}/v1",
                               transport=StdlibTransport(read_timeout=3 * 3600))
    # api_retries=0: a timed-out request must not be sent a second time (it would still be
    # running on the server, and doubling the load makes everything slower)
    settings = dict(lm="local", client=client, retries=0, api_retries=0, cache_replies=False)
    out = eval_v3.BASE / "rewrites" / name / "rewrites"
    out.mkdir(parents=True, exist_ok=True)
    failures = []

    def one_paper(paper):
        pid = paper["paper_id"]
        text = "\n\n".join(f"## {s}\n\n{paper[s]}" for s in translator.ALL_SECTIONS if paper.get(s))
        extra = {"reference_glossary": glossary_text(pid, text)} if use_glossary else {}
        opening_fn = OPENING_G if use_glossary else rewrite_opening_student
        section_fn = SECTION_G if use_glossary else rewrite_section_student
        try:
            parts = opening_fn.using(**settings, max_tokens=6000)(paper=text, writer="opus", **extra)
            opening = dict(zip(translator.OPENING_SECTIONS, parts))
        except Exception as error:   # noqa: BLE001 - an unreadable answer counts as an empty rewrite
            failures.append(f"{pid} opening: {type(error).__name__}: {str(error)[:150]}")
            opening = {s: "" for s in translator.OPENING_SECTIONS}
        rewritten_opening = "\n\n".join(f"## {s}\n\n{opening[s]}" for s in translator.OPENING_SECTIONS
                                        if opening[s])

        def one_section(s):
            limit = min(16000, max(1500, 2 * len(ENC.encode(paper[s]))))
            try:
                more = {"reference_glossary": glossary_text(pid, paper[s])} if use_glossary else {}
                return section_fn.using(**settings, max_tokens=limit)(
                    section_name=s, original_section=paper[s], rewritten_opening=rewritten_opening, writer="opus",
                    **more)
            except Exception as error:   # noqa: BLE001
                failures.append(f"{pid} {s}: {type(error).__name__}: {str(error)[:150]}")
                return ""

        todo = [s for s in translator.OTHER_SECTIONS if paper.get(s)]
        with ThreadPoolExecutor(len(todo)) as pool:
            sections = dict(zip(todo, pool.map(one_section, todo)))
        rows = [{"paper_id": pid, "section": s, "original": paper[s], "rewrite": opening[s]}
                for s in translator.OPENING_SECTIONS if paper.get(s)]
        rows += [{"paper_id": pid, "section": s, "original": paper[s], "rewrite": sections[s]} for s in todo]
        dpyr.read(rows).write_parquet(out / translator.file_name(pid))

    started = time.time()
    with ThreadPoolExecutor(n) as pool:
        list(pool.map(one_paper, papers(n)))
    print(f"{name}: {n} papers in {(time.time() - started) / 60:.1f} min; {len(failures)} unreadable answers")
    for f in failures:
        print("  ", f)
    (out.parent / "failures.json").write_text(json.dumps(failures, indent=1))
    broken = [f for f in failures if "TransportError" in f or "Timeout" in f or "Connection" in f]
    if broken:
        raise SystemExit(f"{len(broken)} calls failed to reach the model (not the model's fault): not scoring")


def unit_rows(name, n):
    rewrites = eval_v3.load_rewrites(name)
    for paper in papers(n):
        for unit, original, new, context in eval_v3.units(paper, rewrites.get(paper["paper_id"], {})):
            yield paper["paper_id"], unit, original, new, context


def judge(name, n, out_file):
    tasks = [(c, pid, unit, o, new, ctx) for c in [name] + REFERENCES for pid, unit, o, new, ctx in unit_rows(c, n)]
    with ThreadPoolExecutor(12) as pool:
        results = list(pool.map(eval_v3.judge_one, tasks))
    by = {}
    for r in results:
        rec = dict(r)
        if "verdict" in rec:
            task = next(t for t in tasks if t[0] == r["candidate"] and t[1] == r["paper_id"] and t[2] == r["unit"])
            rec["status"] = "ok" if eval_v3.quote_rate(rec["verdict"], task[3], task[4], task[5]) >= \
                eval_v3.MIN_QUOTE_RATE else "invalid"
        by[(r["candidate"], r["paper_id"], r["unit"])] = rec
    units = sorted({(p, u) for _, p, u in by})
    common = [k for k in units if all(by.get((c, *k), {}).get("status") == "ok" for c in [name] + REFERENCES)]
    summary = {"name": name, "papers": [p["paper_id"] for p in papers(n)], "units": len(units),
               "units_compared": len(common), "candidates": {}}
    for c in [name] + REFERENCES:
        recs = [by[(c, *k)] for k in common]
        scores = {a: sum(r["verdict"][a] for r in recs) / len(recs) if recs else None for a in eval_v3.ASPECTS}
        scores["mean"] = sum(scores.values()) / 4 if recs else None
        allrecs = [by[(c, *k)] for k in units if (c, *k) in by]
        checks = [r["checks"] for r in allrecs]
        summary["candidates"][c] = {
            **scores, "major_issues": sum(i["severity"] != "minor" for r in recs for i in r["verdict"]["issues"]),
            "numbers_kept": sum(x["numbers_kept"] for x in checks) / len(checks),
            "headings_kept": sum(x["headings_kept"] for x in checks) / len(checks),
            "third_person": sum(x["third_person"] for x in checks),
            "bullets_added": sum(x["bullets_added"] for x in checks),
            "empty_units": sum(bool(x["empty"]) for x in checks),
            "status": {s: sum(r.get("status") == s for r in allrecs) for s in ("ok", "invalid", "empty", "error")}}
    print(f"{len(common)} of {len(units)} units judged validly for every candidate "
          f"({n} papers: {', '.join(summary['papers'])})")
    print(f"{'':<34}{'faithful':>9}{'underst.':>9}{'pleasant':>9}{'structure':>10}{'mean':>7}{'numbers':>9}{'empty':>7}")
    for c, s in summary["candidates"].items():
        f = lambda v: f"{v:.2f}" if v is not None else "–"
        print(f"{c:<34}{f(s['faithful_and_exact']):>9}{f(s['understandable']):>9}{f(s['pleasant_to_read']):>9}"
              f"{f(s['structure_and_voice']):>10}{f(s['mean']):>7}{s['numbers_kept']:>9.2f}{s['empty_units']:>7}")
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["generate", "judge"])
    ap.add_argument("name")
    ap.add_argument("--papers", type=int, default=3)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--out", default=None)
    ap.add_argument("--glossary", action="store_true", help="give the student the reference glossary")
    a = ap.parse_args()
    generate(a.name, a.papers, a.port, a.glossary) if a.command == "generate" else judge(a.name, a.papers, a.out)
