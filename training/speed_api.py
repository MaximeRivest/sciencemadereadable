"""Time and tokens per paper for the API writers, with the v8 recipe that was scored
(translator.translate_paper: the opening from the whole paper, then the other sections
at the same time). One paper at a time, so each time is what a single reader waits.

    .venv/bin/python training/speed_api.py opus|astra|luna [--papers 3]

Writes training/speed/api-WRITER.json (per paper: seconds, and every call's
full usage record). The rewrites themselves are not kept: the scored ones are in
model_baselines/rewrites/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))
import eval_v3      # noqa: E402
import recipes      # noqa: E402  (WRITER_SETTINGS: the reasoning level each writer was scored with)
import translator   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("writer", choices=["opus", "astra", "luna", "sonnet", "sol", "terra"])
ap.add_argument("--papers", type=int, default=3)
a = ap.parse_args()

translator.WRITERS.setdefault(a.writer, recipes.WRITER_SETTINGS[a.writer])
translator.use_writer(a.writer)

calls = []
_original = translator.call_with_one_retry


def recorded(program, **inputs):
    t = time.time()
    p = _original(program, **inputs)
    calls.append({"step": inputs.get("section_name", "opening"), "seconds": time.time() - t, "start": t,
                  "usage": dict(p.usage or {})})
    return p


translator.call_with_one_retry = recorded

out = {"writer": a.writer, "lm": translator.MODEL, "settings": str(recipes.WRITER_SETTINGS[a.writer]), "papers": []}
for paper in eval_v3.test_papers()[:a.papers]:
    calls.clear()
    t0 = time.time()
    translator.translate_paper(paper)
    out["papers"].append({"paper_id": paper["paper_id"], "seconds": time.time() - t0, "calls": list(calls)})
    tin = sum(c["usage"].get("input_tokens", 0) for c in calls)
    tout = sum(c["usage"].get("output_tokens", 0) for c in calls)
    print(f"{a.writer} {paper['paper_id']}: {time.time() - t0:.0f} s, {tin:,} in, {tout:,} out", flush=True)

target = ROOT / "training/speed" / f"api-{a.writer}.json"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(out, indent=1))
print("saved", target)
