"""
Training examples for a bundle-maker: one section of a paper -> its dense facts
===============================================================================

The step that costs the teachers the most (make_paragraph_notes, then checks and fixes), as
one model job: the input is the section's headings and numbered paragraphs (citations already
[n], as code makes them), the paper's things and the figures and tables placed in the section;
the answer is the section's FINAL facts (after the code check, the meaning check and the fixes),
in a compact line format that code reads back (parse()):

    ¶1 drought shortens both; TX7078 keeps its height better
    R1 [main · result] drought → height ↓ in both accessions (F1a; Fig S1)
    D7 [main · claim] findings suggest: ... {hedge: "suggest", "may"} ← D5, D6

Choices: the rest of the paper is left out of the input (the teachers had it "only for
understanding"; it would triple the length); the things list is the paper's final one (made by
the teacher in a separate step that a later model can learn too). The 20 held-back papers of
the writer are held back here too.

    .venv/bin/python training/notes/bundle/bundlemaker_examples.py -> training/data/examples-bundlemaker.parquet
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util                               # noqa: E402

_spec = importlib.util.spec_from_file_location("bundle_render", HERE / "render.py")
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)

DATA = HERE / "out" / "data300"
OUT = HERE.parents[1] / "data" / "examples-bundlemaker.parquet"
HELD_OUT = HERE.parents[1] / "data" / "examples-bundle-300.parquet"

SYSTEM = """You turn one section of a scientific paper into dense notes for a science writer
who knows nothing about the field and will never see the paper.

For every numbered paragraph (¶1, ¶2...), in order: a line "¶n purpose" (a few words: what the
paragraph does), then its facts, one per line:
  <id> [<importance> · <kind>] <note> {hedge: "<words>"} ← <evidence>
id: the section letter given and a running number. importance: main, support or detail. kind:
background, aim, method, result, claim, limitation or next step. {hedge} only when the authors
hedge (their exact words); ← evidence only for claims (fact ids, figures F2, tables T1).

Notes are dense, not sentences: no articles, no "we observed that"; short names from THINGS;
symbols ↑ rises, ↓ falls, → leads to, ~ linked with, vs compared with, n.s. not significant,
sig. significant. Dense, never lossy: every number with its unit and what it is compared with,
every name, condition, relation the authors state, citation [n] exactly as given, figure or
table pointer by id. Nothing that the section does not say."""


def fact_line(f: dict) -> str:
    hedge = f' {{hedge: "{f["hedge"]}"}}' if f.get("hedge") else ""
    evidence = f" ← {f['evidence']}" if f.get("evidence") else ""
    return f"{f['id']} [{f['importance']} · {f['kind']}] {f['note']}{hedge}{evidence}"


def answer_text(notes: dict) -> str:
    out = []
    for p in sorted(notes["paragraphs"], key=lambda x: x["paragraph"]):
        out.append(f"¶{p['paragraph']} {p['purpose']}")
        out += [fact_line(f) for f in p["facts"]]
    return "\n".join(out)


LINE = re.compile(r"^(?P<id>[A-Z]\d+\w*) \[(?P<importance>main|support|detail) · (?P<kind>[a-z ]+)\] (?P<rest>.*)$")


def parse(text: str) -> dict:
    """The compact answer back into notes ({"paragraphs": [...]}), as the checks read them."""
    paragraphs = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^¶(\d+)\s*(.*)$", line)
        if m:
            paragraphs.append({"paragraph": int(m.group(1)), "purpose": m.group(2), "facts": []})
            continue
        m = LINE.match(line)
        if m and paragraphs:
            rest, evidence, hedge = m.group("rest"), "", ""
            if " ← " in rest:
                rest, evidence = rest.rsplit(" ← ", 1)
            h = re.search(r'\s*\{hedge: "(.*)"\}\s*$', rest)
            if h:
                hedge, rest = h.group(1), rest[:h.start()]
            paragraphs[-1]["facts"].append({"id": m.group("id"), "importance": m.group("importance"),
                                             "kind": m.group("kind"), "note": rest, "hedge": hedge, "evidence": evidence})
    return {"paragraphs": paragraphs}


def user_text(part: str, letter: str, section: str, things: str, figures: str) -> str:
    return (f"Section: {part} (ids start with {letter})\n\nTHINGS\n{things or '(none)'}\n\n"
            f"FIGURES AND TABLES IN THIS SECTION\n{figures or '(none)'}\n\nTHE SECTION\n{section}")


def main():
    held = set(pd.read_parquet(HELD_OUT).query("split == 'validation'").paper_id)
    papers = json.loads((DATA / "papers.json").read_text())["papers"]
    rows, skipped = [], Counter()
    for p in papers:
        folder = DATA / p["teacher"] / p["paper_id"]
        if not (folder / "done.json").exists():
            continue
        b = json.loads((folder / "bundle.json").read_text())
        paper = json.loads((folder / "paper.json").read_text())
        things = "\n".join(f"- {t['name']}: {t['what']}" for t in b["front"]["things"])
        v = paper["verbatim"]
        for part, log in b["notes"].items():
            if log.get("code_check_final"):                               # numbers, citations or hedges still lost
                skipped["code check not clean after fixes"] += 1
                continue
            figures = "\n\n".join(render._figure(f) for f in v["figures"] if f["part"] == part)
            prompt = [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user_text(part, render.LETTER[part], paper["part_texts"][part], things, figures)}]
            rows.append({"paper_id": p["paper_id"], "writer": p["teacher"],
                         "split": "validation" if p["paper_id"] in held else "train", "step": "bundlemaker",
                         "section": part, "prompt": json.dumps(prompt, ensure_ascii=False),
                         "answer": answer_text(log["final"])})
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT)
    print(f"{len(frame)} examples -> {OUT}; {dict(Counter(zip(frame.split, frame.writer)))}; skipped {dict(skipped)}")
    sample = frame.iloc[0]
    assert parse(sample.answer)["paragraphs"], "the answer format must read back"


if __name__ == "__main__":
    main()
