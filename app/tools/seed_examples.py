"""Copy the scored rewrites of the benchmark's 3 test papers (all nine models) into the demo's
store, so visitors can read them without an API key.

    .venv/bin/python app/tools/seed_examples.py

They are exactly the rewrites the judge scored (model_baselines/rewrites/NAME)."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))
import eval_v3   # noqa: E402

STORE = ROOT / "app/rewrites"
SCORED = {"opus": "opus", "sonnet": "v8-sonnet", "astra": "v8-astra", "sol": "v8-sol", "terra": "v8-terra",
          "luna": "v8-luna", "our-9b": "student-9b-glossary-final-greedy",
          "our-4b": "student-4b-glossary-final-greedy", "our-0.8b": "student-0.8b-glossary-final-greedy"}

def folder(doi: str) -> Path:
    return STORE / hashlib.sha256(doi.lower().encode()).hexdigest()[:24]

for paper in eval_v3.test_papers()[:3]:
    doi = paper["doi"].lower()
    f = folder(doi)
    f.mkdir(parents=True, exist_ok=True)
    (f / "paper.json").write_text(json.dumps({"doi": doi, "title": paper["title"], "example": True}, ensure_ascii=False))
    for model, name in SCORED.items():
        parts = eval_v3.load_rewrites(name)[paper["paper_id"]]
        parts = {k: v for k, v in parts.items() if (paper.get(k) or "").strip()}
        (f / f"{model}.json").write_text(json.dumps({
            "doi": doi, "model": model, "parts": parts, "source": "benchmark",
            "note": "the rewrite our judge scored", "created": "2026-10-04"}, ensure_ascii=False))
    print(doi, paper["title"][:60], "->", f.name, len(list(f.glob("*.json"))) - 1, "models")
