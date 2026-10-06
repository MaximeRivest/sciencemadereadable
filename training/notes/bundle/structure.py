"""
The paper's structure, by code
==============================

From the paper's JATS XML, with no model: every part (opening, introduction_rest, methods,
results, discussion), its headings, its paragraphs in order, citations already turned into
[n] (numbers into the reference list), and which figures and tables each paragraph mentions.

    from structure import paper_structure
    s = paper_structure(record)
    s["parts"]["methods"]["paragraphs"] -> [{"n": 1, "heading": "### 2.1. Study Area",
                                             "text": "... [3,5] ...", "mentions": ["F1", "T1"]}]
    s["placement"] -> {"F1": ("methods", 1), ...}   after which paragraph each placeholder goes

Conventions (the same for every paper, so the model learns one way):
  opening            ## title, ## abstract (verbatim, in the bundle's front), ## introduction_first,
                     ## conclusion (its subsections ###)
  introduction_rest  ## introduction_rest, then its subsections as ###
  other parts        ## the section's own heading, subsections ### and ####

Sections are found the way paper_corpus/harvest.py found them (first top-level section whose
title says introduction/background, method/material, result, discussion, conclusion), so the
parts are the same as in the corpus.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from sources import XML, _tag, apply_map

KINDS = [("introduction", ("introduction", "background")), ("methods", ("method", "material")),
         ("results", ("result",)), ("discussion", ("discussion",)), ("conclusion", ("conclusion",))]
SKIP = {"fig", "table-wrap", "graphic", "inline-graphic", "object-id", "alternatives", "label", "caption"}


def _ids(root) -> tuple[dict[str, int], dict[str, str]]:
    """XML id -> reference number (order of the reference list), and XML id -> F2 / T1 (as in
    sources.verbatim)."""
    refs = {r.get("id"): n for n, r in enumerate((x for x in root.iter() if _tag(x) == "ref"), 1)}
    figs = {}
    for node in root.iter():
        if _tag(node) in {"fig", "table-wrap"}:
            label = "".join(node.find("label").itertext()).strip() if node.find("label") is not None else ""
            number = re.sub(r"^\D*?(?=[A-Z]?\d)", "", label).strip(" .")
            figs[node.get("id")] = ("T" if _tag(node) == "table-wrap" else "F") + number
    return refs, figs


def _inline(node, refs, figs, mentions) -> str:
    """The text of a paragraph, with citations as ⟨n⟩ tokens (joined later) and figure or table
    mentions kept as words and noted."""
    kind = _tag(node)
    if kind in SKIP:
        return node.tail or ""
    if kind == "xref":
        rid = (node.get("rid") or "").split()[0] if node.get("rid") else ""
        text = "".join(node.itertext())
        if node.get("ref-type") == "bibr" and rid in refs:
            out = f"⟨{refs[rid]}⟩"
        else:
            if node.get("ref-type") in {"fig", "table"} and rid in figs:
                mentions.append(figs[rid])
            out = text
        return out + (node.tail or "")
    if kind in {"list-item"}:
        inner = (node.text or "") + "".join(_inline(c, refs, figs, mentions) for c in node)
        return inner.strip() + "; " + (node.tail or "")
    return (node.text or "") + "".join(_inline(c, refs, figs, mentions) for c in node) + (node.tail or "")


def _citations(text: str) -> str:
    """⟨n⟩ tokens -> [n,m]: ranges (⟨9⟩−⟨14⟩) expanded, neighbours joined, and brackets or
    parentheses that held only citations dropped."""
    text = re.sub(r"⟨(\d+)⟩\s*[−–-]\s*⟨(\d+)⟩",
                  lambda m: "".join(f"⟨{i}⟩" for i in range(int(m.group(1)), int(m.group(2)) + 1)), text)
    group = r"⟨\d+⟩(?:\s*[,;]?\s*(?:and\s+)?⟨\d+⟩)*"
    text = re.sub(group, lambda m: "[" + ",".join(re.findall(r"\d+", m.group(0))) + "]", text)
    text = re.sub(r"[\[(]\s*(\[[\d,]+\])\s*[\])]", r"\1", text)                  # ([3]) or [[3]] -> [3]
    text = re.sub(r"\s*(\[[\d,]+\])", r" \1", text)
    return re.sub(r"\s+", " ", text).strip()


def _walk(sec, depth, refs, figs, mapping, out, top_heading=None):
    """Headings and paragraphs of a section, in reading order, as a flat list of events:
    {"heading": "### 2.1. Study Area"} or {"text": ..., "mentions": [...]}. depth 0 is the
    part's own heading (top_heading)."""
    if depth == 0:
        out.append({"heading": top_heading})
    else:
        title, label = sec.find("title"), sec.find("label")
        name = " ".join(x for x in ("".join(label.itertext()).strip() if label is not None else "",
                                    "".join(title.itertext()).strip() if title is not None else "") if x)
        if name:
            out.append({"heading": "#" * min(depth + 2, 4) + " " + apply_map(re.sub(r"\s+", " ", name), mapping)})
    for child in sec:
        kind = _tag(child)
        if kind in {"p", "list"}:
            mentions = []
            text = _citations(apply_map(_inline(child, refs, figs, mentions).strip(), mapping))
            if text:
                out.append({"text": text, "mentions": list(dict.fromkeys(mentions))})
        elif kind == "sec":
            _walk(child, depth + 1, refs, figs, mapping, out)
    return out


def paper_structure(record: dict) -> dict:
    root = ET.parse(XML / f"{record['paper_id']}.xml").getroot()
    refs, figs = _ids(root)
    mapping = {form: new for r in (record.get("renamed") or []) for form, new in r["forms"].items()}
    body = next((x for x in root.iter() if _tag(x) == "body"), None)
    found = {}
    for sec in (body.findall("sec") if body is not None else []):
        heading = "".join(sec.find("title").itertext()).lower() if sec.find("title") is not None else ""
        for kind, words in KINDS:
            if kind not in found and any(w in heading for w in words):
                found[kind] = sec
                break

    def heading_of(sec):
        label, title = sec.find("label"), sec.find("title")
        name = " ".join(x for x in ("".join(label.itertext()).strip() if label is not None else "",
                                    "".join(title.itertext()).strip() if title is not None else "") if x)
        return "## " + apply_map(re.sub(r"\s+", " ", name), mapping)

    events = {}
    intro = _walk(found["introduction"], 0, refs, figs, mapping, [], "## introduction_rest") if "introduction" in found else []
    # the introduction's first paragraph (only if the section opens with a paragraph, as in the corpus)
    first = intro[1:2] if len(intro) > 1 and "text" in intro[1] else []
    rest = intro[:1] + intro[2:] if first else intro
    conclusion = _walk(found["conclusion"], 0, refs, figs, mapping, [], "## conclusion") if "conclusion" in found else []
    events["opening"] = ([{"heading": "## introduction_first"}] + first if first else []) + conclusion
    events["introduction_rest"] = rest
    for kind in ("methods", "results", "discussion"):
        if kind in found:
            events[kind] = _walk(found[kind], 0, refs, figs, mapping, [], heading_of(found[kind]))

    out, placement = {}, {}
    for name, items in events.items():
        headings, paragraphs, heading = [], [], None
        for e in items:
            if "heading" in e:
                heading = e["heading"]
                headings.append(heading)
            else:
                paragraphs.append({"n": len(paragraphs) + 1, "heading": heading, **e})
                for m in e["mentions"]:
                    placement.setdefault(m, (name, len(paragraphs)))         # first mention, in reading order
        out[name] = {"headings": headings, "paragraphs": paragraphs}
    return {"parts": out, "placement": placement}


def part_text(part: dict) -> str:
    """A part as plain text: every heading in order, each followed by its numbered paragraphs
    (citations as [n]). For the model that writes the facts, and for the checker."""
    lines = []
    for h in part["headings"]:
        lines.append(h)
        lines += [f"(¶{p['n']}) {p['text']}" for p in part["paragraphs"] if p["heading"] == h]
    return "\n\n".join(lines)
