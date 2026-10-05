# /// script
# requires-python = ">=3.11"
# dependencies = ["tiktoken==0.12.0"]
# ///
"""Download ten PLOS ONE articles and measure normalized full-text token counts.

Run: uv run article-tokens/measure.py
Selection: first ten research-article XML records with a body and bibliography,
starting at DOI 10.1371/journal.pone.0300001 in ascending identifier order.
This is a convenience sample, not a random or representative sample.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import time
from datetime import datetime, timezone
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

import tiktoken

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "xml"
TEXT = ROOT / "text"
RAW.mkdir(exist_ok=True)
TEXT.mkdir(exist_ok=True)
ENC = tiktoken.get_encoding("o200k_base")
BLOCKS = {
    "article-title", "abstract", "sec", "title", "p", "list", "list-item",
    "fig", "caption", "table-wrap", "table", "thead", "tbody", "tfoot", "tr",
    "disp-formula", "ref", "ref-list", "ack", "app", "app-group", "fn", "fn-group",
    "supplementary-material", "statement", "bio", "verse-group", "def-item",
}


def tag(el: ET.Element) -> str:
    return el.tag.rsplit("}", 1)[-1]


def render(el: ET.Element) -> str:
    """Preserve inline text; separate structural blocks and table cells.

    For alternative representations of one equation, prefer TeX over MathML,
    so the same equation is not counted twice. Images themselves have no text.
    """
    name = tag(el)
    if name in {"object-id", "graphic", "inline-graphic"}:
        return ""
    if name == "alternatives":
        children = list(el)
        if not children:
            return el.text or ""
        # PLOS also wraps tables in alternatives, with a graphic listed first.
        # Prefer actual table text rather than silently choosing the empty image.
        for preferred_tag in ("table", "tex-math", "math"):
            preferred = next((c for c in children if tag(c) == preferred_tag), None)
            if preferred is not None:
                return render(preferred)
        preferred = next((c for c in children if tag(c) not in {"graphic", "inline-graphic"}), None)
        return render(preferred) if preferred is not None else ""
    pieces = [el.text or ""]
    for child in el:
        pieces.extend([render(child), child.tail or ""])
    result = "".join(pieces)
    if name in BLOCKS:
        return "\n\n" + result + "\n\n"
    if name in {"td", "th", "label"}:
        return " " + result + " "
    return result


def normalize(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n\n".join(p for s in paragraphs if (p := re.sub(r"\s+", " ", s).strip()))


def article_text(root: ET.Element, include_references: bool) -> str:
    nodes = [
        root.find("./front/article-meta/title-group/article-title"),
        *root.findall("./front/article-meta/abstract"),
        root.find("./body"),
        root.find("./back"),
    ]
    parts = []
    for node in nodes:
        if node is None:
            continue
        node = copy.deepcopy(node)
        if not include_references:
            for parent in node.iter():
                for child in list(parent):
                    if tag(child) == "ref-list":
                        parent.remove(child)
        parts.append(render(node))
    return normalize("\n\n".join(parts)) + "\n"


def get_xml(url: str, destination: Path) -> bytes:
    if destination.exists():
        return destination.read_bytes()
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ScholarsReadingList-token-measurement/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            ET.fromstring(data)  # Do not cache an HTML error response as XML.
            destination.write_bytes(data)
            return data
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("download failed")


def main() -> None:
    rows = []
    skipped = []
    for number in range(300001, 300051):
        suffix = f"{number:07d}"
        doi = f"10.1371/journal.pone.{suffix}"
        url = f"https://journals.plos.org/plosone/article/file?id={doi}&type=manuscript"
        try:
            raw = get_xml(url, RAW / f"{suffix}.xml")
            root = ET.fromstring(raw)
            if (root.get("article-type") != "research-article"
                    or root.find("body") is None or root.find(".//ref-list") is None):
                skipped.append({"doi": doi, "reason": "Not a research-article with body and references"})
                continue
            title = normalize(render(root.find("./front/article-meta/title-group/article-title")))
            full = article_text(root, True)
            without = article_text(root, False)
            assert len(full) >= len(without) > 1000, "Unexpectedly short or malformed article"
            full_path = TEXT / f"{suffix}.full.txt"
            without_path = TEXT / f"{suffix}.without-references.txt"
            full_path.write_text(full)
            without_path.write_text(without)
            pub = root.find("./front/article-meta/pub-date[@pub-type='epub']")
            date = "-".join((pub.findtext(k) or "").zfill(2) for k in ("year", "month", "day")) if pub is not None else ""
            row = {
                "doi": doi, "title": title, "publication_date": date,
                "source_url": url,
                "tokens_without_references": len(ENC.encode(without, disallowed_special=())),
                "tokens_with_references": len(ENC.encode(full, disallowed_special=())),
                "words_with_references": len(full.split()),
                "characters_with_references": len(full),
                "reference_count": len(root.findall(".//ref-list/ref")),
                "xml_bytes": len(raw),
                "xml_sha256": hashlib.sha256(raw).hexdigest(),
                "full_text_sha256": hashlib.sha256(full.encode()).hexdigest(),
                "full_text_path": str(full_path.relative_to(ROOT)),
                "without_references_path": str(without_path.relative_to(ROOT)),
                "license": normalize(render(root.find("./front/article-meta/permissions/license"))),
            }
            rows.append(row)
            print(json.dumps({k:row[k] for k in ("doi", "title", "tokens_without_references", "tokens_with_references")}), flush=True)
            if len(rows) == 10:
                break
            time.sleep(1)
        except (urllib.error.HTTPError, urllib.error.URLError, ET.ParseError) as exc:
            skipped.append({"doi": doi, "reason": str(exc)})
            print(f"SKIP {doi}: {exc}", flush=True)
    if len(rows) != 10:
        raise RuntimeError(f"Only retrieved {len(rows)} suitable articles; refusing to report a ten-paper sample")
    summary = {}
    for column in ("tokens_without_references", "tokens_with_references"):
        values = [r[column] for r in rows]
        summary[column] = {"total": sum(values), "mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}
    result = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": "o200k_base", "tiktoken_version": importlib.metadata.version("tiktoken"),
        "selection": "First ten eligible research-article records in ascending PLOS ONE DOI order starting at 0300001; convenience sample, not random or representative. Reviews may be classified as research-article by the publisher.",
        "included": "Title, abstract, body, headings, textual figure/table captions, table cell text, in-text citation markers, and back matter such as acknowledgments. Full version also includes bibliography.",
        "excluded": "XML markup, author/affiliation metadata, publisher/license metadata, image pixels, linked supplementary file contents, and any text/equations only available as images. Alternative equation representations are counted once, preferring TeX.",
        "normalization": "Structural blocks separated by blank lines; whitespace collapsed within paragraphs; UTF-8 with one trailing newline. Table alternatives prefer actual table text over images. MathML is flattened to its text, so mathematical layout is not fully preserved; these are text-only counts, not multimodal counts. No chat framing or prompt overhead.",
        "summary": summary, "articles": rows, "skipped": skipped,
    }
    (ROOT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with (ROOT / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    full_summary = summary["tokens_with_references"]
    no_summary = summary["tokens_without_references"]
    report = [
        "# Scientific article token measurements", "",
        f"Measured: {result['measured_at_utc']}. Tokenizer: **o200k_base** (GPT-4o), tiktoken {result['tiktoken_version']}.", "",
        "## Results", "",
        "| Article | Without bibliography | With bibliography |",
        "|---|---:|---:|",
    ]
    for row in rows:
        report.append(f"| [{row['title']}](https://doi.org/{row['doi']}) | {row['tokens_without_references']:,} | {row['tokens_with_references']:,} |")
    report.extend([
        "", "## Summary", "",
        f"- With bibliography: mean **{full_summary['mean']:,.0f}**, median **{full_summary['median']:,.0f}**, range **{full_summary['min']:,}–{full_summary['max']:,}** tokens.",
        f"- Without bibliography: mean **{no_summary['mean']:,.0f}**, median **{no_summary['median']:,.0f}**, range **{no_summary['min']:,}–{no_summary['max']:,}** tokens.",
        f"- All ten together, with bibliography: **{full_summary['total']:,}** tokens.",
        "", "## Method and limits", "",
        result["selection"], "", "**Included:** " + result["included"], "",
        "**Excluded:** " + result["excluded"], "", "**Normalization:** " + result["normalization"], "",
        "These are exact token counts for the saved extracted text—not estimates from word counts. PDF extraction, another tokenizer, or adding images/supplements will change the result. This one-journal sample does not estimate the average length of all scientific articles.",
        "", "## Reproduce", "",
        "From the project root:", "", "```sh", "uv run article-tokens/measure.py", "```", "",
        "The script reuses saved publisher XML in `xml/`. Exact counted inputs are in `text/`. Detailed provenance and checksums are in `results.json`, with a spreadsheet-friendly copy in `results.csv`.",
    ])
    (ROOT / "report.md").write_text("\n".join(report) + "\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
