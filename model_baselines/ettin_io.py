"""Host-side input/output for ettin_probe.py (the container has no parquet library).

    .venv/bin/python model_baselines/ettin_io.py prepare   # before the probe
    .venv/bin/python model_baselines/ettin_io.py collect   # after: JSON -> parquet
"""
import json
import sys
from pathlib import Path

import dpyr

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

if sys.argv[1] == "prepare":
    training = []
    for f in sorted((ROOT / "paper_corpus/rewrites/rewrites").glob("*.parquet"))[:40]:
        training += [r for r in dpyr.read_parquet(f).to_dicts() if r["section"] == "introduction_rest"]
    papers = dpyr.read_parquet(BASE / "test_papers.parquet").to_dicts()
    (BASE / "ettin_input.json").write_text(json.dumps({"training_sections": training, "test_papers": papers}))
    print(f"{len(papers)} test papers, {len(training)} training sections")
else:
    out = BASE / "rewrites/ettin1b/rewrites"
    for f in out.glob("*.json"):
        dpyr.read(json.loads(f.read_text())).write_parquet(f.with_suffix(".parquet"))
    print(f"{len(list(out.glob('*.parquet')))} papers collected")
