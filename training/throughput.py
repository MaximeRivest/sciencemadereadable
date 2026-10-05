"""Rewrite many papers at once with a student served by vLLM, and measure the speed.

    .venv/bin/python training/throughput.py NAME PORT ids.txt

Every paper starts at once (opening first, then its sections in parallel), so the
server always has a full queue. Uses the same functai functions and glossary input
as training. Saves the rewrites like the benchmark candidates and a timing table.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dpyr
import lm15
from lm15.transports import StdlibTransport

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import eval_student as E                                           # noqa: E402
import translator                                                  # noqa: E402

name, port, ids_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ids = set(Path(ids_file).read_text().split())
papers = [p for p in dpyr.read_parquet(ROOT / "paper_corpus/papers.parquet").to_dicts() if p["paper_id"] in ids]
client = lm15.OpenAIChatLM(api_key="none", base_url=f"http://127.0.0.1:{port}/v1",
                           transport=StdlibTransport(read_timeout=3 * 3600, max_connections=400))
settings = dict(lm="local", client=client, retries=0, api_retries=0, cache_replies=False)
out = E.eval_v3.BASE / "rewrites" / name / "rewrites"
out.mkdir(parents=True, exist_ok=True)
calls, lock = [], threading.Lock()


def call(fn, step, pid, max_tokens, **inputs):
    t = time.time()
    try:
        p = fn.using(**settings, max_tokens=max_tokens).predict(**inputs)
        usage, ok = p.usage or {}, True
        result = tuple(p[k] for k in translator.OPENING_SECTIONS) if step == "opening" else p.result
    except Exception as error:   # noqa: BLE001
        result, usage, ok = None, {}, f"{type(error).__name__}: {str(error)[:120]}"
    with lock:
        calls.append({"paper": pid, "step": step, "seconds": time.time() - t, "start": t,
                      "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
                      "ok": ok})
    return result


def one_paper(paper):
    pid, t0 = paper["paper_id"], time.time()
    text = "\n\n".join(f"## {s}\n\n{paper[s]}" for s in translator.ALL_SECTIONS if paper.get(s))
    parts = call(E.OPENING_G, "opening", pid, 6000, paper=text, writer="opus",
                 reference_glossary=E.glossary_text(pid, text))
    opening = dict(zip(translator.OPENING_SECTIONS, parts)) if parts else {s: "" for s in translator.OPENING_SECTIONS}
    t1 = time.time()
    rewritten_opening = "\n\n".join(f"## {s}\n\n{opening[s]}" for s in translator.OPENING_SECTIONS if opening[s])
    todo = [s for s in translator.OTHER_SECTIONS if paper.get(s)]

    def section(s):
        limit = min(16000, max(1500, 2 * len(E.ENC.encode(paper[s]))))
        return call(E.SECTION_G, s, pid, limit, section_name=s, original_section=paper[s],
                    rewritten_opening=rewritten_opening, writer="opus",
                    reference_glossary=E.glossary_text(pid, paper[s])) or ""
    with ThreadPoolExecutor(len(todo)) as pool:
        sections = dict(zip(todo, pool.map(section, todo)))
    rows = [{"paper_id": pid, "section": s, "original": paper[s], "rewrite": opening[s]}
            for s in translator.OPENING_SECTIONS if paper.get(s)]
    rows += [{"paper_id": pid, "section": s, "original": paper[s], "rewrite": sections[s]} for s in todo]
    dpyr.read(rows).write_parquet(out / translator.file_name(pid))
    return {"paper": pid, "opening_seconds": t1 - t0, "sections_seconds": time.time() - t1,
            "total_seconds": time.time() - t0, "words_in": sum(len((paper.get(s) or "").split()) for s in translator.ALL_SECTIONS)}


started = time.time()
with ThreadPoolExecutor(len(papers)) as pool:
    per_paper = list(pool.map(one_paper, papers))
wall = time.time() - started
report = {"papers": len(papers), "wall_seconds": wall, "per_paper": per_paper, "calls": calls}
(out.parent / "throughput.json").write_text(json.dumps(report, indent=1))
tin, tout = sum(c["input_tokens"] for c in calls), sum(c["output_tokens"] for c in calls)
fails = [c for c in calls if c["ok"] is not True]
print(f"{len(papers)} papers in {wall / 60:.1f} min = {len(papers) / wall * 60:.2f} papers/min; "
      f"{len(calls)} calls, {len(fails)} failed")
print(f"tokens: {tin:,} in, {tout:,} out; output {tout / wall:,.0f} tok/s, input+output {(tin + tout) / wall:,.0f} tok/s")
