"""Prepare source sections and an exact, text-only export of the selected rewrites.

No model calls. No authentication files, system prompts, tool messages, or thinking
blocks are exported. The raw session file is read only, never copied or modified.

python rewrite_benchmark/prepare.py --session /path/to/session.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DOI = "10.1371/journal.pone.0300008"
SESSION_ID = "01a0e736-a851-75b8-8af7-9bab57c64275"
MESSAGE_IDS = {"opening": "7edc86a7", "introduction_rest": "c2d8f99b", "methods": "ebe8df96", "results": "f08d074c", "discussion": "2b066f5c"}
ORDER = ["title", "abstract", "introduction_first", "introduction_rest", "methods", "results", "discussion", "conclusion"]
BLOCKS = {"article-title", "abstract", "sec", "title", "p", "list", "list-item", "fig", "caption", "table-wrap", "table", "thead", "tbody", "tfoot", "tr", "disp-formula", "ref", "ref-list", "ack", "app", "app-group", "fn", "fn-group", "supplementary-material", "statement"}


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


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
    value = (node.text or "") + "".join(render(child) + (child.tail or "") for child in node)
    if name in BLOCKS:
        return "\n\n" + value + "\n\n"
    if name in {"td", "th"}:
        return " | " + value + " "
    return value


def text(node: ET.Element) -> str:
    return "\n\n".join(clean for paragraph in re.split(r"\n\s*\n", render(node)) if (clean := re.sub(r"\s+", " ", paragraph).strip()))


def prepare_source(xml_path: Path, output: Path = DATA) -> dict:
    """Read publisher JATS, keeping source mistakes and tables as printed."""
    raw = xml_path.read_bytes()
    root = ET.fromstring(raw)
    article_doi = root.findtext("./front/article-meta/article-id[@pub-id-type='doi']")
    if article_doi != DOI:
        raise ValueError(f"This paired pilot expects {DOI}, not {article_doi}. A different paper needs its own section map and reference export.")
    meta = root.find("./front/article-meta")
    by_title = {sec.findtext("title", "").lower(): sec for sec in root.findall("./body/sec")}
    intro = by_title["introduction"]
    paragraphs = intro.findall("./p")
    if len(paragraphs) != 4:
        raise ValueError("Introduction changed: inspect and revise the source pairing explicitly")
    sections = {
        "title": text(meta.find("./title-group/article-title")),
        "abstract": text(meta.find("abstract")),
        "introduction_first": text(paragraphs[0]),
        "introduction_rest": "\n\n".join(text(p) for p in paragraphs[1:]),
        "methods": text(by_title["material and methods"]),
        "results": text(by_title["results"]),
        "discussion": text(by_title["discussion"]),
        "conclusion": text(by_title["conclusions"]),
    }
    packet = {
        "doi": DOI, "title": sections["title"], "source_url": f"https://journals.plos.org/plosone/article/file?id={DOI}&type=manuscript",
        "xml_path": str(xml_path.relative_to(PROJECT)), "xml_sha256": digest(raw),
        "authors": [" ".join(filter(None, (n.findtext("given-names"), n.findtext("surname")))) for n in meta.findall("./contrib-group/contrib/name")],
        "license": text(meta.find("./permissions/license")), "order": ORDER,
        "sections": sections,
        "locators": {"title": "front/article-meta/title-group/article-title", "abstract": "front/article-meta/abstract", "introduction_first": "body/Introduction/p[1]", "introduction_rest": "body/Introduction/p[2:4]", "methods": "body/Material and methods", "results": "body/Results", "discussion": "body/Discussion", "conclusion": "body/Conclusions"},
        "references": text(root.find("./back/ref-list")),
        "extraction_limits": "XML markup removed; actual table text preferred over graphic alternatives; whitespace normalized; MathML flattened. Text errors retained. Image content is not extracted automatically; see the separately identified Figure 1 transcription.",
    }
    packet["source_text"] = "\n\n".join(f"## {key}\n\n{sections[key]}" for key in ORDER)
    write_json(output / "paper.json", packet)
    return packet


def visible_text(message: dict) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if part.get("type") == "text")


def ancestry(entries: list[dict], leaf: str) -> list[dict]:
    indexed = {e["id"]: e for e in entries if e.get("type") != "session" and "id" in e}
    result, seen = [], set()
    current = leaf
    while current:
        if current in seen or current not in indexed:
            raise ValueError(f"Broken session ancestry at {current}")
        seen.add(current)
        entry = indexed[current]
        result.append(entry)
        current = entry.get("parentId")
    return result[::-1]


def bounded_slice(full: str, start_marker: str, stop_marker: str | None = None) -> tuple[str, int, int]:
    if full.count(start_marker) != 1:
        raise ValueError(f"Expected one heading: {start_marker!r}")
    start = full.index(start_marker) + len(start_marker)
    end = full.index(stop_marker, start) if stop_marker else len(full)
    while start < end and full[start].isspace(): start += 1
    while end > start and full[end - 1].isspace(): end -= 1
    return full[start:end], start, end


def export_reference(session: Path, paper: dict, output: Path = DATA) -> list[dict]:
    lines = session.read_text().splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    header = next(e for e in entries if e.get("type") == "session")
    if header.get("id") != SESSION_ID:
        raise ValueError("Wrong conversation: explicit selectors only apply to the pinned pilot session")
    branch = ancestry(entries, MESSAGE_IDS["discussion"])
    active = {e["id"]: e for e in branch}
    exported = {}
    for stage, entry_id in MESSAGE_IDS.items():
        e = active[entry_id]
        m = e["message"]
        if m.get("role") != "assistant" or m.get("stopReason") not in {"stop", "end_turn"}:
            raise ValueError(f"Not a complete assistant answer: {entry_id}")
        content = visible_text(m)
        # Allowlist only: no thinking, tool calls, provider payloads, or raw usage.
        exported[stage] = {
            "stage": stage, "entry_id": entry_id, "session_id": header["id"],
            "timestamp": e["timestamp"], "provider": m.get("provider"), "model": m.get("model"),
            "response_model": m.get("responseModel"), "reasoning_level": m.get("providerThinkingLevel"),
            "text": content, "text_sha256": digest(content),
        }
    # Keep all text verbatim in the message export, including editorial notes.
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference_messages.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported.values()))
    specs = {
        "title": ("opening", "## Title\n", "\n## Abstract"),
        "abstract": ("opening", "## Abstract\n", "\n## First paragraph"),
        "introduction_first": ("opening", "## First paragraph\n", "\n## Last paragraph"),
        "conclusion": ("opening", "## Last paragraph\n", "\n---"),
        "introduction_rest": ("introduction_rest", "### Paragraph 2\n", None),
        "methods": ("methods", "## Methods\n", "\n---"),
        "results": ("results", "## Results\n", "\n---"),
        "discussion": ("discussion", "## Discussion\n", None),
    }
    rows = []
    for key in ORDER:
        stage, start_marker, stop_marker = specs[key]
        message = exported[stage]
        selected, start, end = bounded_slice(message["text"], start_marker, stop_marker)
        notes = message["text"].split("\n---\n", 1)[1].strip() if "\n---\n" in message["text"] else ""
        row = {
            "paper_id": DOI, "section_id": key, "source_locator": paper["locators"][key],
            "source_text": paper["sections"][key], "rewrite_text": selected,
            "editorial_notes": notes, "candidate_id": "conversation_reference",
            "origin": "tool-assisted, user-guided conversation; not a gold answer or a held-out test",
            "stage": stage, "session_id": header["id"], "entry_id": message["entry_id"],
            "provider": message["provider"], "model": message["model"], "timestamp": message["timestamp"],
            "character_start": start, "character_end": end,
            "message_text_sha256": message["text_sha256"], "rewrite_sha256": digest(selected),
        }
        assert message["text"][start:end] == selected
        rows.append(row)
    (output / "reference_pairs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (output / "reference_document.md").write_text("# Conversation rewrite — verbatim excerpts\n\n" + "\n\n".join(f"## {r['section_id']}\n\n{r['rewrite_text']}" for r in rows) + "\n\n## Editorial notes\n\n" + "\n\n".join(dict.fromkeys(r["editorial_notes"] for r in rows if r["editorial_notes"])) + "\n")
    write_json(output / "provenance.json", {
        "session_id": header["id"], "session_date": header["timestamp"],
        "session_filename": session.name, "branch_leaf": MESSAGE_IDS["discussion"], "selected_entry_ids": MESSAGE_IDS,
        "source_xml_sha256": paper["xml_sha256"],
        "pairs_sha256": digest((output / "reference_pairs.jsonl").read_bytes()),
        "messages_sha256": digest((output / "reference_messages.jsonl").read_bytes()),
        "export_policy": "Only the five selected assistant messages' visible text and allowlisted provenance. No thinking, tool messages, system prompts, credentials, or unrelated conversation.",
        "pairing_policy": "Eight source sections paired to exact character spans; outer section headings and conversational wrappers omitted. Inner headings unchanged. Editorial notes retained separately. No reference text rewritten by the exporter.",
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=PROJECT / "article-tokens/xml/0300008.xml")
    args = parser.parse_args()
    paper = prepare_source(args.xml.resolve())
    rows = export_reference(args.session.expanduser(), paper)
    print(f"Prepared {len(rows)} source/rewrite pairs from {len(MESSAGE_IDS)} messages.\n{DATA}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- any IMRaD paper

KINDS = [("introduction", ("introduction", "background")), ("methods", ("method",)),
         ("results", ("result",)), ("discussion", ("discussion",)), ("conclusion", ("conclusion",))]


def prepare_any(xml_path: Path) -> dict:
    """Split a JATS article into the pilot's section names, by heading keywords.

    title, abstract, introduction_first (first paragraph), introduction_rest,
    methods, results, discussion and, when the paper has one, conclusion.
    Other body sections (supporting information, extra headings) are left out.
    """
    raw = Path(xml_path).read_bytes()
    root = ET.fromstring(raw)
    meta = root.find("./front/article-meta")
    found = {}
    for sec in root.findall("./body/sec"):
        heading = (sec.findtext("title") or "").lower()
        for kind, words in KINDS:
            if kind not in found and any(w in heading for w in words):
                found[kind] = sec
                break
    missing = [k for k in ("introduction", "methods", "results", "discussion") if k not in found]
    if missing:
        raise ValueError(f"{xml_path.name}: no section found for {missing}")
    intro = found["introduction"]
    paragraphs = intro.findall("./p")
    first = text(paragraphs[0]) if paragraphs else ""
    rest = [text(p) for p in paragraphs[1:]] + [text(s) for s in intro.findall("./sec")]
    sections = {
        "title": text(meta.find("./title-group/article-title")),
        "abstract": text(meta.find("abstract")),
        "introduction_first": first,
        "introduction_rest": "\n\n".join(rest),
        "methods": text(found["methods"]),
        "results": text(found["results"]),
        "discussion": text(found["discussion"]),
    }
    if "conclusion" in found:
        sections["conclusion"] = text(found["conclusion"])
    order = [k for k in ORDER if k in sections]
    doi = meta.findtext("./article-id[@pub-id-type='doi']")
    return {
        "doi": doi, "title": sections["title"], "order": order, "sections": sections,
        "xml_sha256": digest(raw),
        "source_text": "\n\n".join(f"## {k}\n\n{sections[k]}" for k in order),
        "references": text(root.find("./back/ref-list")) if root.find("./back/ref-list") is not None else "",
    }
