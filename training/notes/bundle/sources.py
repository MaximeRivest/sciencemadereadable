"""
The verbatim pieces of the bundle
=================================

What the bundle copies from the paper word for word, with no model involved: the title, the
abstract, every figure caption and every table (with its caption and footnotes), from the
paper's JATS XML (paper_corpus/xml/<id>.xml). Papers with invented species names get the same
renaming here as everywhere else.

    from sources import verbatim
    v = verbatim(record)   # record: one paper of training/v3/pilot50
    v["title"], v["abstract"]
    v["figures"]     -> [{"id", "label", "kind", "caption", "table", "part"}]   id: F2, T1, TA1
    v["references"]  -> [{"n", "short"}]   n: the number the writer cites, [n]; short: "Lodders 2021"

Figures and tables are never rewritten: the writer puts a placeholder ([F2], [T1]) where they go
and code shows the original. Citations are written [n] or [n,m] with n from this list, and code
renders them. So the model learns one way of doing each, and presentation stays in code.
"""
from __future__ import annotations

import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "training/v3"))
from build import apply_map                         # noqa: E402  the renaming used for v3

XML = ROOT / "paper_corpus/xml"


def _tag(node) -> str:
    return node.tag.rsplit("}", 1)[-1]


def _text(node) -> str:
    """All the text inside a node, on one line (graphics and formula markup dropped)."""
    if node is None:
        return ""
    if _tag(node) in {"graphic", "inline-graphic", "object-id", "alternatives"}:
        return ""
    s = (node.text or "") + "".join(_text(c) + (c.tail or "") for c in node)
    return re.sub(r"\s+", " ", s).strip()


def _table(node) -> str:
    """An XML table as a markdown table (row and column spans are not expanded)."""
    rows = []
    for tr in node.iter():
        if _tag(tr) == "tr":
            cells = [_text(c).replace("|", "/") for c in tr if _tag(c) in {"td", "th"}]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
    if len(rows) > 1:
        rows.insert(1, "|" + " --- |" * rows[0].count(" | ") + " --- |")
    return "\n".join(rows)


def _opening_piece(opening: str, label: str) -> str:
    m = re.search(rf"^## {label}\s*\n(.*?)(?=^## |\Z)", opening, re.S | re.M)
    return m.group(1).strip() if m else ""


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _where(label: str, caption: str, convs: dict[str, str]) -> str:
    """The part whose text holds the caption; else the first part that mentions the label
    ("Fig. 3", "Figure 3"); else "appendix"."""
    start = _squash(caption)[:40]
    for part, text in convs.items():
        if start and start in _squash(text):
            return part
    number = re.sub(r"^\D+", "", label)
    word = r"Tables?" if label.lower().startswith("tab") else r"Fig(?:ure)?s?\.?"
    for part, text in convs.items():
        if re.search(rf"\b{word}\s*{re.escape(number)}(?![\d.])", text):
            return part
    return "appendix"


def verbatim(record: dict) -> dict:
    """Title, abstract, figures and tables of one paper, renamed like the rest of its record.
    Each figure or table also says which part of the paper shows it (where its caption is)."""
    convs = {c["section"]: c["original"] for c in record["conversations"]}
    opening = convs.get("opening", "")
    mapping = {form: new for r in (record.get("renamed") or []) for form, new in r["forms"].items()}

    root = ET.parse(XML / f"{record['paper_id']}.xml").getroot()
    items = []
    for node in root.iter():
        kind = _tag(node)
        if kind not in {"fig", "table-wrap"}:
            continue
        label = _text(node.find("label")) or ("Figure" if kind == "fig" else "Table")
        caption = _text(node.find("caption"))
        table = ""
        if kind == "table-wrap":
            t = next((c for c in node.iter() if _tag(c) == "table"), None)
            table = _table(t) if t is not None else ""
            notes = " ".join(_text(f) for f in node.iter() if _tag(f) == "table-wrap-foot")
            if notes:
                caption = (caption + " " + notes).strip()
        caption, table = apply_map(caption, mapping), apply_map(table, mapping)
        part = _where(label, caption, convs)
        number = re.sub(r"^\D*?(?=[A-Z]?\d)", "", label).strip(" .")
        items.append({"id": ("T" if kind == "table-wrap" else "F") + number, "label": label,
                      "kind": "table" if kind == "table-wrap" else "figure",
                      "caption": caption, "table": table, "part": part})
    return {"title": _opening_piece(opening, "title"), "abstract": _opening_piece(opening, "abstract"),
            "figures": items, "references": references(record["paper_id"])}


def references(paper_id: str) -> list[dict]:
    """The reference list, in the paper's order, each as its number and first author + year."""
    xml = (XML / f"{paper_id}.xml").read_text()
    out = []
    for n, ref in enumerate(re.findall(r"<ref[ >].*?</ref>", xml, re.S), 1):
        text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", ref))).strip()
        text = re.sub(r"^\d+\.?\s*", "", text)                          # the printed label "12."
        author = (re.match(r"[^\s,;.]+(?:\s+(?:de|van|von|da|del|der|le|la)\s+[^\s,;.]+)?", text) or [""])[0]
        year = re.search(r"\b(1[89]\d\d|20\d\d)[a-z]?\b", text)
        out.append({"n": n, "short": f"{author} {year.group(0) if year else 'n.d.'}".strip()})
    return out
