"""
Checks by code: does each paragraph's notes keep what the paragraph says?
=========================================================================

Free and exact, so they can run on every paper. For each paragraph of the paper and the dense
facts made from it:

  numbers    every number of the paragraph appears in its facts (citation numbers excluded)
  citations  every [n] of the paragraph appears in its facts
  hedges     every hedge word of the paragraph ("may", "suggest", "likely"...) appears in its
             facts (in the note or the hedge field)

A model check is then needed only where code finds a problem, plus a random sample. In 4d both
run on every part, to see how much of what the model check finds code also finds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "rewrite_benchmark"))
from eval_v3 import numbers                         # noqa: E402  numbers in a text, citation brackets ignored

# "potential" and "possible" are left out ("potential energy", "potential habitats" are not
# hedges), and "could not" / "cannot" are about ability, not certainty (see hedges()).
HEDGES = ["may", "might", "could", "possibly", "likely", "unlikely",
          "probably", "perhaps", "suggest", "suggests", "suggested", "suggesting", "appear", "appears",
          "seem", "seems", "presumably", "putative", "tentative", "uncertain", "unclear", "speculate"]
_STEMS = {"suggest": "suggest", "appear": "appear", "seem": "seem", "possib": "possib", "likel": "likel",
          "probabl": "probabl"}


def cited(text: str) -> set[int]:
    """Reference numbers in [n] or [n,m] brackets (and ranges [66-69], expanded)."""
    out = {int(x) for m in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text) for x in m.split(",")}
    for a, b in re.findall(r"\[(\d+)\s*[-–]\s*(\d+)\]", text):
        out |= set(range(int(a), int(b) + 1))
    return out


def hedges(text: str) -> set[str]:
    """Hedge words in a text, as stems (suggests, suggested -> suggest)."""
    words = re.findall(r"[a-z]+", re.sub(r"\bcould(?:\s+not|n't)\b", " ", text.lower()))
    found = set()
    for w in words:
        if w in HEDGES:
            found.add(next((s for p, s in _STEMS.items() if w.startswith(p)), w))
    return found


def _facts_text(facts: list[dict]) -> str:
    return " ".join(f"{f['note']} {f.get('hedge', '')} {f.get('evidence', '')}" for f in facts)


def paragraph_problems(paragraph: dict, facts: list[dict]) -> list[dict]:
    text, notes = paragraph["text"], _facts_text(facts)
    plain = re.sub(r"\[[\d,\s]+\]", " ", text)
    problems = []
    lost = sorted(numbers(plain) - numbers(notes), key=lambda x: (len(x), x))
    if lost:
        problems.append({"paragraph": paragraph["n"], "kind": "numbers missing", "detail": ", ".join(lost)})
    lost = sorted(cited(text) - cited(notes))
    if lost:
        problems.append({"paragraph": paragraph["n"], "kind": "citations missing", "detail": ", ".join(map(str, lost))})
    lost = sorted(hedges(plain) - hedges(notes))
    if lost:
        problems.append({"paragraph": paragraph["n"], "kind": "hedge words missing", "detail": ", ".join(lost)})
    return problems


def part_problems(part_structure: dict, notes: dict) -> list[dict]:
    """All problems of one part. notes: {"paragraphs": [{"paragraph": n, "facts": [...]}, ...]}."""
    by_n = {p["paragraph"]: p["facts"] for p in notes["paragraphs"]}
    out = []
    for p in part_structure["paragraphs"]:
        if p["n"] not in by_n:
            out.append({"paragraph": p["n"], "kind": "paragraph without notes", "detail": ""})
        else:
            out += paragraph_problems(p, by_n[p["n"]])
    return out
