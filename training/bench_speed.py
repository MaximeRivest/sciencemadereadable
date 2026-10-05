"""Time per paper and papers per hour for a student served by vLLM (OpenAI-style API).

    .venv/bin/python training/bench_speed.py latency    NAME PORT
    .venv/bin/python training/bench_speed.py throughput NAME PORT [--ids FILE] [--papers N]

Same requests as the scored benchmark runs (eval_student.py generate --glossary): the
opening from the whole paper (max 6000 tokens), then the four other sections at the
same time, each with its reference glossary and the eval's length limit. Glossaries are
looked up before the clock starts, so only the model's work is timed.

latency:    the 3 benchmark test papers, one at a time, alone on the server: what a
            single reader waits for a paper.
throughput: every paper sent at once (default: speed/throughput_ids.txt, 128 unused
            corpus papers), so the server stays full: papers per hour, for the price.

Writes speed/MODE-NAME.json (every call's timing and tokens) and the rewrites to
speed/rewrites/MODE-NAME/ (to check that faster settings give the same text).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dpyr
import lm15
from lm15.transports import StdlibTransport

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_student as E                                           # noqa: E402
import translator                                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["latency", "throughput"])
ap.add_argument("name")
ap.add_argument("port", type=int)
ap.add_argument("--ids", default=str(HERE / "speed/throughput_ids.txt"))
ap.add_argument("--papers", type=int, default=None)
ap.add_argument("--host", default="127.0.0.1")
a = ap.parse_args()

if a.mode == "latency":
    papers = E.papers(3)
else:
    ids = Path(a.ids).read_text().split()[:a.papers]
    corpus = dpyr.read_parquet(E.ROOT / "paper_corpus/papers.parquet").to_dicts()
    papers = [p for p in corpus if p["paper_id"] in set(ids)]

client = lm15.OpenAIChatLM(api_key="none", base_url=f"http://{a.host}:{a.port}/v1",
                           transport=StdlibTransport(read_timeout=3 * 3600, max_connections=1000))
settings = dict(lm="local", client=client, retries=0, api_retries=0, cache_replies=False)
out_dir = HERE / "speed/rewrites" / f"{a.mode}-{a.name}"
out_dir.mkdir(parents=True, exist_ok=True)

# everything the requests need, prepared before timing
prepared = []
for p in papers:
    pid = p["paper_id"]
    text = "\n\n".join(f"## {s}\n\n{p[s]}" for s in translator.ALL_SECTIONS if p.get(s))
    todo = [s for s in translator.OTHER_SECTIONS if p.get(s)]
    prepared.append({"paper": p, "text": text, "glossary": E.glossary_text(pid, text), "todo": todo,
                     "section_glossary": {s: E.glossary_text(pid, p[s]) for s in todo},
                     "limit": {s: min(16000, max(1500, 2 * len(E.ENC.encode(p[s])))) for s in todo}})

calls, lock = [], threading.Lock()


def call(fn, step, pid, max_tokens, **inputs):
    t = time.time()
    try:
        r = fn.using(**settings, max_tokens=max_tokens).predict(**inputs)
        usage, ok = dict(r.usage or {}), True
        result = tuple(r[k] for k in translator.OPENING_SECTIONS) if step == "opening" else r.result
    except Exception as error:   # noqa: BLE001
        result, usage, ok = None, {}, f"{type(error).__name__}: {str(error)[:160]}"
    with lock:
        calls.append({"paper": pid, "step": step, "start": t, "seconds": time.time() - t,
                      "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
                      "ok": ok})
    return result


def one_paper(x):
    p, pid, t0 = x["paper"], x["paper"]["paper_id"], time.time()
    parts = call(E.OPENING_G, "opening", pid, 6000, paper=x["text"], writer="opus", reference_glossary=x["glossary"])
    opening = dict(zip(translator.OPENING_SECTIONS, parts)) if parts else {s: "" for s in translator.OPENING_SECTIONS}
    t1 = time.time()
    rewritten_opening = "\n\n".join(f"## {s}\n\n{opening[s]}" for s in translator.OPENING_SECTIONS if opening[s])

    def section(s):
        return call(E.SECTION_G, s, pid, x["limit"][s], section_name=s, original_section=p[s],
                    rewritten_opening=rewritten_opening, writer="opus",
                    reference_glossary=x["section_glossary"][s]) or ""
    with ThreadPoolExecutor(len(x["todo"])) as pool:
        sections = dict(zip(x["todo"], pool.map(section, x["todo"])))
    t2 = time.time()
    rows = [{"paper_id": pid, "section": s, "original": p[s], "rewrite": opening[s]}
            for s in translator.OPENING_SECTIONS if p.get(s)]
    rows += [{"paper_id": pid, "section": s, "original": p[s], "rewrite": sections[s]} for s in x["todo"]]
    dpyr.read(rows).write_parquet(out_dir / translator.file_name(pid))
    return {"paper": pid, "opening_seconds": t1 - t0, "sections_seconds": t2 - t1, "total_seconds": t2 - t0,
            "words_in": sum(len((p.get(s) or "").split()) for s in translator.ALL_SECTIONS)}


started = time.time()
if a.mode == "latency":
    per_paper = [one_paper(x) for x in prepared]
else:
    with ThreadPoolExecutor(len(prepared)) as pool:
        per_paper = list(pool.map(one_paper, prepared))
wall = time.time() - started

fails = [c for c in calls if c["ok"] is not True]
tin, tout = sum(c["input_tokens"] for c in calls), sum(c["output_tokens"] for c in calls)
report = {"mode": a.mode, "name": a.name, "papers": len(prepared), "wall_seconds": wall, "failed_calls": len(fails),
          "input_tokens": tin, "output_tokens": tout, "per_paper": per_paper, "calls": calls}
(HERE / "speed" / f"{a.mode}-{a.name}.json").write_text(json.dumps(report, indent=1))
if a.mode == "latency":
    print(f"{a.name} latency: " + ", ".join(f"{r['paper']} {r['total_seconds']:.0f}s "
                                             f"(opening {r['opening_seconds']:.0f}s)" for r in per_paper)
          + f"; {len(fails)} failed calls")
else:
    print(f"{a.name} throughput: {len(prepared)} papers in {wall / 60:.1f} min = {len(prepared) / wall * 3600:.0f} "
          f"papers/hour; {tin:,} in, {tout:,} out tokens ({tout / wall:,.0f} out tok/s); {len(fails)} failed calls")
for f in fails[:5]:
    print("  ", f["paper"], f["step"], f["ok"])
