"""Collect every number the blog figures need into data.json (no model or judge calls:
judge verdicts are read from the benchmark's cache).

    .venv/bin/python training/figures/blog/export_data.py

(main venv, repository root; the PYTHONPATH is the functai version the students were
trained with, which eval_student.py needs to render prompts.)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE.parents[1]
sys.path.insert(0, str(T))
import eval_student as E   # noqa: E402

A = E.eval_v3.ASPECTS
CANDIDATES = {
    # label: rewrite folder name
    "opus_v8": "opus", "astra_v8": "v8-astra", "luna_v8": "v8-luna",
    "opus_v1": "v1-opus", "astra_v1": "v1-astra", "luna_v1": "v1-luna",
    "opus_v8_glossary": "opus-glossary",
    "q08_untrained": "qwen08b", "q4_untrained": "qwen4b", "q9_untrained": "qwen9b",
    "q08_trained_noglossary": "student-0.8b-full-final-greedy",
    "q08_trained": "student-0.8b-glossary-final-greedy",
    "q4_trained_noglossary": "student-4b-final-greedy",
    "q4_trained": "student-4b-glossary-final-greedy",
    "q4_s152_noglossary": "student-4b-s152-greedy",
    "q4_s152_glossary": "student-4b-s152-greedy-glossary",
    "q9_trained": "student-9b-glossary-final-greedy",
    "q9_s424": "student-9b-glossary-step-00424-greedy",
    "q9_rerun_h100": "speed-h100-9b",
    "q9_mtp_h100": "speed-h100-9b-mtp",
}

units = {}
for label, name in CANDIDATES.items():
    rows = []
    for pid, unit, o, new, ctx in E.unit_rows(name, 3):
        r = E.eval_v3.judge_one((name, pid, unit, o, new, ctx))
        v = r.get("verdict")
        rows.append({"paper": pid, "unit": unit, "status": r.get("status"),
                     "scores": {a: v[a] for a in A} if v else None,
                     "issues": [i["severity"] for i in v["issues"]] if v else []})
    units[label] = rows


def val_loss(run):
    out = []
    for line in open(T / "runs" / run / "log.jsonl"):
        d = json.loads(line)
        if d.get("kind") == "validation":
            out.append({"step": d["step"], "loss": d["all"], "answer_tokens": d.get("seen_answer_tokens")})
    return out


train = {}
for key, run in {"q08": "qwen35-0.8b-glossary-scratch", "q4": "qwen35-4b-glossary-scratch",
                 "q9": "qwen35-9b-full-glossary", "q4_first": "qwen35-4b"}.items():
    c = json.loads((T / "runs" / run / "config.json").read_text())
    c.setdefault("method", "lora" if c.get("lora_rank") else c.get("method"))
    s = json.loads((T / "runs" / run / "status.json").read_text())
    train[key] = {"run": run, "method": c.get("method"), "gpu": c.get("gpu"), "hours": s.get("train_hours"),
                  "steps": c.get("total_steps"), "conversations": c.get("train_conversations"),
                  "train_tokens": c.get("train_tokens"), "trainable": c.get("trainable_parameters"),
                  "parameters": c.get("parameters"), "validation": val_loss(run)}

speed = json.loads((T / "speed/summary.json").read_text())
(HERE / "data.json").write_text(json.dumps({"aspects": list(A), "units": units, "train": train, "speed": speed,
                                            "data": {"examples": 21069, "papers": 4416, "papers_opus": 2934,
                                                     "papers_astra": 1482, "corpus": 5000,
                                                     "test_papers": 3, "test_units": 15}}, indent=1))
print("wrote", HERE / "data.json", {k: sum(r["status"] == "ok" for r in v) for k, v in units.items()})
