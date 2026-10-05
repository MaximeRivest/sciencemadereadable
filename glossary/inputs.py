"""The reference glossary as the rewriter reads it: one line per term that occurs in the
text it rewrites. Shared by training (training/prepare.py) and evaluation
(training/eval_student.py), so both see exactly the same input."""
from __future__ import annotations

import json
import re
from pathlib import Path

GLOSSARIES = Path(__file__).resolve().parent / "out"


def glossary_text(pid: str, scope: str) -> str:
    """The glossary entries of paper `pid` whose term (or long form) occurs in `scope`."""
    path = GLOSSARIES / f"{pid}.json"
    if not path.exists():
        return ""
    return glossary_lines(json.loads(path.read_text()), scope)


def glossary_lines(entries: list[dict], scope: str) -> str:
    """The same, for glossary entries already in hand (glossary.glossary's output)."""
    lines = []
    for e in entries:
        if not e.get("explanation"):
            continue
        names = [e["term"]] + ([e["stands_for"]] if e.get("stands_for") else [])
        if any(re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", scope, re.I) for n in names):
            head = e["term"] + (f" ({e['stands_for']})" if e.get("stands_for") else "")
            lines.append(f"- {head}: {e['explanation']} [{e['source']}]")
    return "\n".join(lines)
