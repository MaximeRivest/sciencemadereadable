"""Pilot: rewrite 50 papers with Opus and measure how much subscription they use.

1. Sets aside a test set of 500 papers, never used for training (test_ids.txt).
2. Rewrites 50 other papers, 10 at a time, into rewrites/ (they count toward the
   full run later: finished papers are skipped next time).
3. Reads Claude usage before and after each batch, into pilot_usage.jsonl.
4. Stops early if the 5-hour window passes 90%.

    .venv/bin/python paper_corpus/pilot.py
"""
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import dpyr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "rewrite_benchmark"))
sys.path.insert(0, str(HERE))
import translator
from usage import claude_usage

OUT = HERE / "rewrites"
TEST_IDS = HERE / "test_ids.txt"
USAGE_LOG = HERE / "pilot_usage.jsonl"
PILOT_SIZE, BATCH, STOP_AT = 50, 10, 90

translator.PAPERS_AT_ONCE = BATCH

papers = dpyr.read_parquet(HERE / "papers.parquet").to_dicts()
ids = sorted(p["paper_id"] for p in papers)

if not TEST_IDS.exists():
    TEST_IDS.write_text("\n".join(sorted(random.Random(2026).sample(ids, 500))) + "\n")
test = set(TEST_IDS.read_text().split())
pool = [p for p in papers if p["paper_id"] not in test]
random.Random(50).shuffle(pool)
pilot = pool[:PILOT_SIZE]


def record(stage, batch_number, done):
    u = claude_usage()
    row = {"at": datetime.now(timezone.utc).isoformat(), "stage": stage, "batch": batch_number,
           "papers_done": done, **u}
    with USAGE_LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[usage] {stage} batch {batch_number}: 5h {u['five_hour']:.0f}%, week {u['seven_day']:.0f}%", flush=True)
    return u


done = sum((OUT / "rewrites" / translator.file_name(p["paper_id"])).exists() for p in pilot)
for b in range(0, PILOT_SIZE, BATCH):
    u = record("before", b // BATCH + 1, done)
    if u["five_hour"] >= STOP_AT:
        print(f"Stopping: 5-hour window at {u['five_hour']:.0f}%. It resets at {u['five_hour_resets'][:16]} UTC.")
        break
    started = time.time()
    translator.translate_papers(dpyr.read(pilot[b:b + BATCH]), OUT)
    done = sum((OUT / "rewrites" / translator.file_name(p["paper_id"])).exists() for p in pilot)
    record("after", b // BATCH + 1, done)
    print(f"batch took {(time.time() - started) / 60:.1f} min", flush=True)

print(f"PILOT DONE: {done} of {PILOT_SIZE} papers rewritten.")
