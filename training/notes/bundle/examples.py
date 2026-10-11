"""
Training examples for the writer: bundle -> text, from the 300-paper run
========================================================================

One example per part of a paper: the prompt is the writer's brief and reader (system) and the
bundle, the part to write and what the reader already read (user); the answer is the final
text of that part. Same columns as training/data/examples.parquet, so training/train.py reads it.

Kept out of training:
  - 20 papers (10 per teacher, fixed seed) as validation and later test papers;
  - parts whose final text still has a serious problem after the revision: a fact changed or
    missing, a changed certainty, a wrong explanation, or a form error. Parts with only small
    outside additions are kept (counted in the report).

    .venv/bin/python training/notes/bundle/examples.py   ->  training/data/examples-bundle.parquet
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prompts as PR                                # noqa: E402

DATA = HERE / "out" / "data300"
OUT = HERE.parents[1] / "data" / "examples-bundle.parquet"
VALIDATION_PER_TEACHER = 10

SYSTEM = (f"{PR.BRIEF}\n\nThe reader: {PR.READER}")


def user(bundle: str, part: str, already: str) -> str:
    return (f"Write part {part} of the paper in this bundle.\n\n"
            f"Already read by the reader (empty for the opening):\n{already or '(nothing yet)'}\n\n"
            f"Bundle:\n{bundle}")


def serious(log: dict) -> list[str]:
    """Problems still in the final text (from its own form check and trace)."""
    final_is_revised = log["final"] == log.get("revised")
    form = log["revised_form"] if final_is_revised else log["first_form"]
    tr = log["revised_trace"] if final_is_revised else log["first_trace"]
    out = [k for k, v in form.items() if v and k != "citations_not_in_facts"]
    out += [x["status"] for x in tr["facts"] if x["status"] != "kept"]
    out += ["wrong explanation" for x in tr["additions"] if x["kind"] == "wrong explanation"]
    return out


def main():
    papers = json.loads((DATA / "papers.json").read_text())["papers"]
    done = [p for p in papers if (DATA / p["teacher"] / p["paper_id"] / "done.json").exists()]
    rng = random.Random(2026)
    validation = set()
    for t in ("opus", "astra"):
        ids = sorted(p["paper_id"] for p in done if p["teacher"] == t)
        validation |= set(rng.sample(ids, VALIDATION_PER_TEACHER))
    rows, dropped, small_additions = [], Counter(), 0
    for p in done:
        folder = DATA / p["teacher"] / p["paper_id"]
        bundle = (folder / "bundle.txt").read_text()
        texts = json.loads((folder / "texts.json").read_text())
        names = list(texts)
        split = "validation" if p["paper_id"] in validation else "train"
        for n in names:
            problems = serious(texts[n])
            if problems and split == "train":
                dropped.update(problems)
                continue
            log = texts[n]
            tr = log["revised_trace"] if log["final"] == log.get("revised") else log["first_trace"]
            small_additions += any(x["kind"] == "outside fact" for x in tr["additions"])
            already = "" if n == names[0] else texts[names[0]]["final"]
            prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user(bundle, n, already)}]
            rows.append({"paper_id": p["paper_id"], "writer": p["teacher"], "split": split, "step": "bundle",
                         "section": n, "prompt": json.dumps(prompt, ensure_ascii=False), "answer": log["final"]})
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT)
    c = Counter(zip(frame.split, frame.writer))
    print(f"{len(done)} papers, {len(frame)} examples -> {OUT}")
    print("examples by split and teacher:", dict(c))
    print("train parts dropped, by problem:", dict(dropped), "· kept parts with a small outside addition:", small_additions)


if __name__ == "__main__":
    main()
