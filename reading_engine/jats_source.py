"""Canonical JATS rendering, section splitting and licence rules.
Shared by corpus preparation, the paper service and source-version verification.
"""

import re
import xml.etree.ElementTree as ET

BLOCKS = {
    "article-title",
    "abstract",
    "sec",
    "title",
    "p",
    "list",
    "list-item",
    "fig",
    "caption",
    "table-wrap",
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "disp-formula",
    "ref",
    "ref-list",
    "ack",
    "app",
    "app-group",
    "fn",
    "fn-group",
    "supplementary-material",
    "statement",
}
KINDS = [
    ("introduction", ("introduction", "background")),
    ("methods", ("method", "material")),
    ("results", ("result",)),
    ("discussion", ("discussion",)),
    ("conclusion", ("conclusion",)),
]


def tag(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def render(node: ET.Element) -> str:
    name = tag(node)
    if name in {"graphic", "inline-graphic", "object-id"}:
        return ""
    if name == "alternatives":
        for preferred in ("table", "tex-math", "math"):
            match = next((child for child in node if tag(child) == preferred), None)
            if match is not None:
                return render(match)
        return ""
    value = (node.text or "") + "".join(
        render(child) + (child.tail or "") for child in node
    )
    if name in BLOCKS:
        return "\n\n" + value + "\n\n"
    if name in {"td", "th"}:
        return " | " + value + " "
    return value


def text(node: ET.Element) -> str:
    return "\n\n".join(
        clean
        for paragraph in re.split(r"\n\s*\n", render(node))
        if (clean := re.sub(r"\s+", " ", paragraph).strip())
    )


def split_sections(root):
    """The translator's sections; None when required research sections are absent."""
    meta = root.find("./front/article-meta")
    found, order = {}, []
    for sec in root.findall("./body/sec"):
        heading = (sec.findtext("title") or "").lower()
        for kind, words in KINDS:
            if kind not in found and any(w in heading for w in words):
                found[kind] = sec
                order.append(kind)
                break
    if (
        not all(k in found for k in ("introduction", "methods", "results"))
        or meta is None
    ):
        return None
    intro = found["introduction"]
    paragraphs = intro.findall("./p")
    sections = {
        "title": text(meta.find("./title-group/article-title")),
        "abstract": text(meta.find("abstract"))
        if meta.find("abstract") is not None
        else "",
        "introduction_first": text(paragraphs[0]) if paragraphs else "",
        "introduction_rest": "\n\n".join(
            [text(p) for p in paragraphs[1:]]
            + [text(s) for s in intro.findall("./sec")]
        ),
    }
    for kind in ("methods", "results", "discussion", "conclusion"):
        sections[kind] = text(found[kind]) if kind in found else ""
    sections["original_section_order"] = ",".join(order)
    return sections


def license_ok(root):
    """CC BY, excluding non-commercial, no-derivatives and share-alike variants."""
    for lic in root.iter():
        if not lic.tag.endswith("license"):
            continue
        links = [v for e in lic.iter() for v in e.attrib.values()]
        blob = (" ".join(links) + " " + " ".join(lic.itertext())).lower()
        if any(
            x in blob
            for x in (
                "licenses/by-nc",
                "licenses/by-nd",
                "licenses/by-sa",
                "noncommercial",
                "no derivatives",
            )
        ):
            return False
        if (
            "creativecommons.org/licenses/by/" in blob
            or "creative commons attribution" in blob
            or "cc by license" in blob
            or "(cc by)" in blob
        ):
            return True
        if "broadest form of re-use" in blob and "commercial" in blob:
            return True
    return False
