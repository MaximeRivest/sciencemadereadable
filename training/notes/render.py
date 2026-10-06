"""
How the notes are shown to each reader of them
==============================================

The notes are stored as data (prompts.Notes). This file turns them into the plain text that each
step reads. Three views:

  full    for the checkers: every fact with its id and importance, the plan, the definitions
  level1  for the writer, with all the help: headings, paragraph plan with purposes, facts in
          plan order with their importance, definitions with pictures
  level3  for the writer, with no help: headings, facts under each heading in a shuffled order,
          no importance, no plan, definitions without pictures or "central" marks

The writer never sees fact ids: they are only for tracing.
"""
from __future__ import annotations

import random


def _defs(notes: dict, pictures: bool) -> str:
    """pictures=True (level 1, checkers): central ideas are marked, with their picture."""
    out = []
    for d in notes["definitions"]:
        line = f"- {d['term']}" + (" (central)" if pictures and d.get("central") else "") + f": {d['meaning']}"
        if pictures and d.get("picture"):
            line += f" (picture: {d['picture']})"
        out.append(line)
    return "Definitions\n" + "\n".join(out) if out else "Definitions\n(none)"


def plan(notes: dict) -> list[dict]:
    """The paragraph plan, repaired so that every fact is in exactly one paragraph: a fact the
    plan forgot goes at the end of the last paragraph of its heading (or a new one)."""
    facts = {f["id"]: f for f in notes["facts"]}
    seen, paragraphs = set(), []
    for p in notes["paragraphs"]:
        ids = [i for i in p["facts"] if i in facts and i not in seen]
        seen |= set(ids)
        if ids:
            paragraphs.append({**p, "facts": ids})
    for f in notes["facts"]:
        if f["id"] not in seen:
            same = [p for p in paragraphs if p["heading"] == f["heading"]]
            if same:
                same[-1]["facts"].append(f["id"])
            else:
                paragraphs.append({"heading": f["heading"], "purpose": "(not planned)", "facts": [f["id"]]})
            seen.add(f["id"])
    return paragraphs


def _by_heading(notes: dict, items: list[dict]) -> list[tuple[str, list[dict]]]:
    """Group items by heading, in the order of notes['headings'] (unknown headings last)."""
    order = list(notes["headings"]) + sorted({i["heading"] for i in items} - set(notes["headings"]))
    return [(h, [i for i in items if i["heading"] == h]) for h in order]


def full(notes: dict) -> str:
    facts = {f["id"]: f for f in notes["facts"]}
    out = []
    for heading, paragraphs in _by_heading(notes, plan(notes)):
        out.append(heading)
        for n, p in enumerate(paragraphs, 1):
            out.append(f"  Paragraph {n}: {p['purpose']}")
            out += [f"    {i} [{facts[i]['importance']}] {facts[i]['statement']}" for i in p["facts"]]
    return "\n".join(out) + "\n\n" + _defs(notes, pictures=True)


def level1(notes: dict) -> str:
    facts = {f["id"]: f for f in notes["facts"]}
    out = []
    for heading, paragraphs in _by_heading(notes, plan(notes)):
        out.append(heading)
        for n, p in enumerate(paragraphs, 1):
            out.append(f"  Paragraph {n}: {p['purpose']}")
            out += [f"    - [{facts[i]['importance']}] {facts[i]['statement']}" for i in p["facts"]]
    return "\n".join(out) + "\n\n" + _defs(notes, pictures=True)


def level3(notes: dict, seed: str) -> str:
    rng = random.Random(seed)
    out = []
    for heading, facts in _by_heading(notes, notes["facts"]):
        facts = facts[:]
        rng.shuffle(facts)
        out.append(heading)
        out += [f"  - {f['statement']}" for f in facts]
    return "\n".join(out) + "\n\n" + _defs(notes, pictures=False)
