"""
How the bundle is shown
=======================

The bundle is stored as data: verbatim pieces (sources.py), the front (things, glossary) and
one Part per part of the paper (prompts.Part). This file turns it into text:

  bundle(...)      the whole bundle, as the writer reads it (the same text for every part)
  part_view(...)   one part with what it relies on, for the notes checker
  trace_view(...)  one part's items with ids, for the trace checker

Order of the bundle: TITLE, ABSTRACT, THINGS, GLOSSARY (with reference sheets), FIGURES AND
TABLES (for understanding only: the writer places [F2] / [T1] and never rewrites them),
REFERENCES (cited as [n]), OUTLINE (every part: headings, planned paragraphs, facts and
placeholders in place).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "training/v3"))
sys.path.insert(0, str(ROOT / "glossary"))
from pilot50 import STATISTICS_SHEET, methods_lines     # noqa: E402  the always-available sheets
from units_sheet import units_lines                     # noqa: E402

PARTS = ["opening", "introduction_rest", "methods", "results", "discussion"]
LETTER = {"opening": "O", "introduction_rest": "I", "methods": "M", "results": "R", "discussion": "D"}


def sheets(paper_text: str) -> str:
    """The reference sheets: statistics always; methods and units entries that the paper uses."""
    return "\n".join(x for x in (STATISTICS_SHEET, methods_lines(paper_text), units_lines(paper_text)) if x)


def fact_line(f: dict) -> str:
    tags = f"[{f['importance']} · {f['kind']}]"
    hedge = f' {{hedge: "{f["hedge"]}"}}' if f.get("hedge") else ""
    evidence = f" ← {f['evidence']}" if f.get("evidence") else ""
    return f"{f['id']} {tags} {f['note']}{hedge}{evidence}"


def plan(part: dict, labels_here: list[str]) -> list[dict]:
    """The paragraph plan, repaired: every fact and every figure/table of this part in exactly
    one paragraph. Forgotten items go at the end of the last paragraph of their heading (facts)
    or of the part (figures and tables)."""
    ids = {f["id"]: f for f in part["facts"]}
    valid = set(ids) | set(labels_here)
    seen, out = set(), []
    for p in part["plan"]:
        items = [i for i in p["items"] if i in valid and i not in seen]
        seen |= set(items)
        if items:
            out.append({**p, "items": items})
    for f in part["facts"]:
        if f["id"] not in seen:
            same = [p for p in out if p["heading"] == f["heading"]]
            if not same:
                out.append({"heading": f["heading"], "purpose": "(not planned)", "items": []})
                same = [out[-1]]
            same[-1]["items"].append(f["id"])
            seen.add(f["id"])
    for label in labels_here:
        if label not in seen:
            if not out:
                out.append({"heading": part["headings"][0], "purpose": "(not planned)", "items": []})
            out[-1]["items"].append(label)
    return out


def outline_of(name: str, part: dict, labels_here: list[str]) -> str:
    """One part of the OUTLINE: headings in order, planned paragraphs, facts in place."""
    ids = {f["id"]: f for f in part["facts"]}
    paragraphs = plan(part, labels_here)
    lines = [f"=== PART: {name} ==="]
    if name == "opening":
        lines += ["## title", "  ¶ a plain title → TITLE", "## abstract", "  ¶ the abstract, plainly → ABSTRACT"]
    headings = list(part["headings"]) + sorted({p["heading"] for p in paragraphs} - set(part["headings"]))
    for h in headings:
        lines.append(h)
        for p in (p for p in paragraphs if p["heading"] == h):
            lines.append(f"  ¶ {p['purpose']}")
            for i in p["items"]:
                lines.append(f"    {fact_line(ids[i])}" if i in ids else f"    [{i}]")
    return "\n".join(lines)


def _front(front: dict) -> str:
    things = "\n".join(f"- {t['name']}: {t['what']}" for t in front["things"]) or "(none)"
    gloss = "\n".join(f"- {d['term']}" + (" (central)" if d["central"] else "") + f": {d['meaning']}"
                      + (f" (picture: {d['picture']})" if d.get("picture") else "") for d in front["glossary"])
    return f"# THINGS\n{things}\n\n# GLOSSARY\n{gloss}"


def _figure(f: dict) -> str:
    where = f"placed in {f['part']}" if f["part"] != "appendix" else "appendix, not placed"
    return f"{f['id']} = {f['label']} ({where}): {f['caption']}" + (f"\n{f['table']}" if f["table"] else "")


def _references(v: dict) -> str:
    return "# REFERENCES (cite as [n])\n" + ("; ".join(f"{r['n']} {r['short']}" for r in v["references"]) or "(none)")


def bundle(v: dict, front: dict, parts: dict[str, dict], sheet_text: str) -> str:
    figures = "\n\n".join(_figure(f) for f in v["figures"]) or "(none)"
    outline = "\n\n".join(outline_of(n, parts[n], [f["id"] for f in v["figures"] if f["part"] == n])
                          for n in PARTS if n in parts)
    return (f"# TITLE\n{v['title']}\n\n# ABSTRACT\n{v['abstract']}\n\n{_front(front)}\n\n"
            f"## Reference sheets\n{sheet_text}\n\n# FIGURES AND TABLES (for understanding only; never write them)\n"
            f"{figures}\n\n{_references(v)}\n\n# OUTLINE\n{outline}")


def part_view(v: dict, front: dict, name: str, part: dict) -> str:
    here = [f for f in v["figures"] if f["part"] == name]
    top = f"# TITLE\n{v['title']}\n\n# ABSTRACT\n{v['abstract']}\n\n" if name == "opening" else ""
    figures = "\n\n".join(_figure(f) for f in here) or "(none)"
    return (f"{top}{_front(front)}\n\n# FIGURES AND TABLES IN THIS PART\n{figures}\n\n{_references(v)}\n\n# OUTLINE\n"
            + outline_of(name, part, [f["id"] for f in here]))


def trace_view(v: dict, name: str, part: dict) -> str:
    """The items the text must hold, each with an id: the facts of this part, and for the opening
    the title and the abstract. (Placeholders and citations are checked by code: pilot.form_checks.)"""
    lines = []
    if name == "opening":
        lines += [f"TITLE: {v['title']}", f"ABSTRACT: {v['abstract']}"]
    lines += [fact_line(f) for f in part["facts"]]
    return "Items, each with its id:\n" + "\n".join(lines)
