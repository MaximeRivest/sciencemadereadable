"""Make one reading edition, EPUB, print HTML, and optional Kokoro audio.

Consumes a finished, reviewed PDF rewrite. No new rewriting is performed.
Reviewed original figures and tables can be placed inline. Generated cells are
never presented as source tables.
"""

from __future__ import annotations
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
import requests
from extract import save
from visual_assets import add_visuals, html_for_format, visual_title

HERE = Path(__file__).resolve().parent
MARKDOWN = MarkdownIt("commonmark", {"html": True}).enable("table")
ORDER = [
    ("abstract", "Abstract"),
    ("introduction", "Introduction"),
    ("methods", "Methods"),
    ("results", "Results"),
    ("discussion", "Discussion"),
    ("conclusion", "Conclusion"),
]
ALLOWED = {
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "em",
    "strong",
    "ul",
    "ol",
    "li",
    "blockquote",
    "code",
    "pre",
    "sup",
    "sub",
    "br",
    "hr",
    "a",
}
NOTICE = (
    "This is a simplified research paper, not a fully fact-checked edition. "
    "The wording has changed; consult the original paper for exact claims, figures, and numbers."
)
AUDIO_NOTICE = (
    "You are listening to a simplified research paper. This edition has not been fully fact checked. "
    "Citation numbers are left out of the narration. Tables and mathematical expressions are announced, "
    "not read as rows or symbols. You can find the original paper and the narration transcript with this recording."
)
PRINT_CSS = """
@page { size: A5; margin: 17mm 15mm 20mm; }
body { font-family: 'Noto Serif', Georgia, serif; font-size: 14pt; line-height: 1.55; color: var(--color-text-primary, #111); background: var(--color-background-primary, #fff); }
h1 { font-size: 24pt; line-height: 1.2; } h2 { font-size: 20pt; line-height: 1.25; } h3,h4 { font-size: 15pt; line-height: 1.3; }
h1,h2,h3,h4 { break-after: avoid; } p { orphans: 3; widows: 3; } a { color: inherit; overflow-wrap: anywhere; }
.chapter { break-before: page; } .notice { border-left: 2pt solid currentColor; padding-left: 10pt; font-size: 11pt; }
.meta { font-size: 11pt; } pre { white-space: pre-wrap; } .cover { break-after: page; } img { max-width: 100%; } li { margin: .4em 0; }
@page source-portrait { size: A4 portrait; margin: 14mm; }
@page source-landscape { size: A4 landscape; margin: 14mm; }
.source-visual { margin: 0; break-before: page; break-after: page; break-inside: avoid; }
.source-visual.portrait { page: source-portrait; }
.source-visual.landscape { page: source-landscape; }
.source-visual figcaption { font: 11pt/1.4 sans-serif; margin: 0 0 5mm; }
.source-visual img { display: block; width: 100%; height: 225mm; object-fit: contain; }
.source-visual.landscape img { height: 150mm; }
"""


def sha(value):
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def clean_chapter(markdown):
    """Whitelist markup and replace generated table cells, never pass scripts to readers."""
    soup = BeautifulSoup(MARKDOWN.render(markdown), "html.parser")
    count = 0
    for table in list(soup.find_all("table")):
        note = soup.new_tag("p")
        note.string = "[Original table: consult the source pages. Generated table cells are not reproduced in this edition.]"
        table.replace_with(note)
        count += 1
    for tag in list(soup.find_all(True)):
        if tag.name is None:  # already removed with a forbidden parent
            continue
        if tag.name in {
            "script",
            "style",
            "iframe",
            "object",
            "svg",
            "math",
            "img",
            "video",
            "audio",
            "form",
        }:
            tag.decompose()
        elif tag.name not in ALLOWED:
            tag.unwrap()
        else:
            href = tag.get("href", "")
            tag.attrs = {}
            if tag.name == "a" and (
                href.startswith("https://")
                or href.startswith("http://")
                or href.startswith("#")
            ):
                tag["href"] = href
    # Safe XML-compatible fragment, needed by EPUB as well as the web edition.
    fragment = soup.decode(formatter="minimal")
    ET.fromstring("<div>" + fragment + "</div>")
    return fragment, count


def narration(fragment):
    soup = BeautifulSoup(fragment, "html.parser")
    for br in soup.find_all("br"):
        br.replace_with(" ")
    chunks = []
    for node in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre"]):
        if node.find_parent(["li", "pre"]):
            continue
        text = node.get_text("", strip=False).strip()
        if not text:
            continue
        text = text.replace(
            "[Original table: consult the source pages. Generated table cells are not reproduced in this edition.]",
            "A table appears here in the original paper. Please consult the source pages for its values.",
        )
        text = re.sub(r"\[(?:\d+[\s,;–—-]*)+\]", "", text)
        text = re.sub(
            r"\$\$[\s\S]*?\$\$|\$[^$\n]+\$|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]",
            " A mathematical expression appears here; please consult the written version. ",
            text,
        )
        # Explicit unit pronunciation only; leave other wording unchanged.
        units = [
            (r"ng\s*g(?:⁻¹|\^\{-1\}|−1)", "nanograms per gram"),
            (r"μg\s*g(?:⁻¹|\^\{-1\}|−1)", "micrograms per gram"),
            (r"\bng\b", "nanograms"),
            (r"\bmL\b", "millilitres"),
            (r"[µμ]L", "microlitres"),
            (r"[µμ]g", "micrograms"),
            (r"°\s*C", " degrees Celsius"),
            (r"\bLC[–-]MS/MS\b", "L C M S M S"),
            (r"\bEDCs\b", "E D Cs"),
            (r"\bEDC\b", "E D C"),
        ]
        for pattern, replacement in units:
            text = re.sub(pattern, replacement, text)
        # Spell out comparison symbols; do not turn an inequality into an equality.
        text = text.replace("≤", " less than or equal to ").replace(
            "≥", " greater than or equal to "
        )
        text = (
            text.replace("<", " less than ")
            .replace(">", " greater than ")
            .replace("±", " plus or minus ")
        )
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def split_speech(text, limit=1100):
    """Bound Kokoro input; preserve every non-whitespace character in order."""
    result = []
    for paragraph in text.split("\n\n"):
        rest = paragraph.strip()
        while len(rest) > limit:
            window = rest[: limit + 1]
            boundaries = list(re.finditer(r"[.!?;:]\s+", window))
            at = (
                boundaries[-1].end()
                if boundaries and boundaries[-1].end() > limit // 3
                else window.rfind(" ")
            )
            if at <= 0:
                raise ValueError(
                    "Unbroken speech token exceeds the chunk limit; inspect the transcript."
                )
            result.append(rest[:at].strip())
            rest = rest[at:].strip()
        if rest:
            result.append(rest)
    assert re.sub(r"\s", "", "".join(result)) == re.sub(r"\s", "", text)
    return result


def make_edition(run, out, visual_manifest=None, regenerate_pdf=False, automatic=False):
    rewrite = json.loads((run / "rewrite.json").read_text())
    labels = json.loads((run / "labelled.json").read_text())
    paper = json.loads((run / "paper.json").read_text())
    if (
        not labels["ready"]
        or rewrite["sectionsSha256"] != paper["sections_sha256"]
        or paper["sections"] != labels["sections"]
    ):
        raise ValueError("Rewrite/source mismatch or unresolved labels.")
    comparison = (
        json.loads((run / "comparison.json").read_text())
        if (run / "comparison.json").exists()
        else {}
    )
    extraction = json.loads((run / "extraction.json").read_text())
    parts = dict(rewrite["parts"])
    parts["introduction"] = "\n\n".join(
        parts.get(k, "") for k in ("introduction_first", "introduction_rest")
    )
    chapters = []
    for key, title in ORDER:
        if not parts.get(key):
            continue
        if key == "results" and any(
            b["section"] == "results_discussion" for b in labels["ledger"]
        ):
            title = "Results and discussion"
        fragment, tables = clean_chapter(parts[key])
        destinations = (
            {"introduction_first", "introduction_rest"}
            if key == "introduction"
            else {key}
        )
        pages = sorted(
            {b["page"] for b in labels["ledger"] if b["destination"] in destinations}
        )
        spoken = title + ".\n\n" + narration(fragment)
        chapters.append(
            {
                "id": key,
                "title": title,
                "html": fragment,
                "speech": spoken,
                "source_pages": pages,
                "generated_tables_omitted": tables,
            }
        )
    title = (
        BeautifulSoup(MARKDOWN.render(parts["title"]), "html.parser").get_text().strip()
    )
    original_title = re.sub(r"^#+\s*", "", labels["sections"]["title"])
    edition = {
        "version": 1,
        "id": sha(json.dumps(rewrite["parts"], sort_keys=True)),
        "title": title,
        "original_title": original_title,
        "doi": comparison.get("doi", ""),
        "authors": comparison.get("authors", []),
        "year": comparison.get("year", ""),
        "license": comparison.get("license", "not established"),
        "notice": NOTICE,
        "source_pdf_sha256": extraction["signature"]["pdf_sha256"],
        "source_version": extraction.get("source_record", {}).get("version"),
        "source_page_count": extraction["pages"],
        # Pages after the main text (references, appendices) were not read; they
        # stay in original.pdf and the combined PDF, not as page pictures.
        "source_pages_read": extraction.get("read_pages")
        or list(range(1, extraction["pages"] + 1)),
        "reviewed_blocks": sum(bool(b["manual_review"]) for b in labels["ledger"]),
        "chapters": chapters,
    }
    if (out / "edition.json").exists() and json.loads(
        (out / "edition.json").read_text()
    )["id"] != edition["id"]:
        raise ValueError(
            "Output folder belongs to a different rewrite; choose a new folder."
        )
    manifest_path = visual_manifest or run / "original-visuals.json"
    edition["visuals"], edition["visuals_sha256"] = [], None
    if automatic:
        from automatic_visuals import manifest_for

        save(manifest_path, manifest_for(run, chapters, edition["id"]))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest_hash = sha(json.dumps(manifest, sort_keys=True))
        previous = (
            json.loads((out / "edition.json").read_text())
            if (out / "edition.json").exists()
            else {}
        )
        if (
            (out / "paper.pdf").exists()
            and previous.get("visuals_sha256") != manifest_hash
            and not regenerate_pdf
        ):
            raise ValueError(
                "Visuals changed: include --pdf to rebuild the existing PDF, or choose a fresh output folder."
            )
        edition["visuals"], edition["visuals_sha256"] = add_visuals(
            run / "original.pdf", edition["id"], chapters, out, manifest
        )
        # Keep the reviewed mapping with the run so ordinary rebuilds do not drop images.
        if manifest_path.resolve() != (run / "original-visuals.json").resolve():
            save(run / "original-visuals.json", manifest)
    elif visual_manifest:
        raise ValueError("Visual manifest does not exist.")
    placed = {v["label"] for v in edition["visuals"]}
    references = set()
    for value in parts.values():
        for kind, number in re.findall(
            r"\b(Figure|Fig\.?|Table)\s+(S?\d+)\b", value, re.I
        ):
            references.add(
                ("Table" if kind.lower() == "table" else "Figure")
                + " "
                + number.upper()
            )
    notes = []
    fallback = (
        extraction.get("artifact", {})
        .get("meta", {})
        .get("extra", {})
        .get("fallback_pages")
        or []
    )
    if fallback:
        notes.append(
            "Original page(s) "
            + ", ".join(map(str, fallback))
            + " were read with a simpler backup OCR engine and checked by the reader before rewriting. Compare details on those pages with the original."
        )
    unplaced = sorted(references - placed)
    if unplaced:
        notes.append(
            "Not confidently placed in this edition: "
            + ", ".join(unplaced[:20])
            + ". These may refer to supplementary material or need checking against the original."
        )
    if any("$$" in c["html"] or "\\[" in c["html"] for c in chapters):
        notes.append(
            "Some formulas retain their written notation rather than full mathematical typesetting. The original PDF shows their layout."
        )
    skipped = [
        p
        for p in range(1, edition["source_page_count"] + 1)
        if p not in read_pages(edition)
    ]
    if skipped:
        notes.append(
            f"Only the main text was read and rewritten (original pages {page_ranges(read_pages(edition))}). "
            f"Pages {page_ranges(skipped)} (references, appendices and similar) are unchanged in the original PDF."
        )
    edition["quality_notes"] = notes
    save(out / "edition.json", edition)
    transcript = (
        title
        + ".\n\n"
        + AUDIO_NOTICE
        + "\n\n"
        + "\n\n".join(c["speech"] for c in chapters)
    )
    (out / "narration.txt").write_text(transcript + "\n")
    shutil.copyfile(run / "original.pdf", out / "original.pdf")
    # The diagnostic thumbnails are too small for comfortable EPUB zooming.
    # Keep colour and render a separate 200-dpi source-page appendix.
    import pymupdf

    (out / "pages").mkdir(exist_ok=True)
    with pymupdf.open(out / "original.pdf") as source:
        for number in read_pages(edition):
            source[number - 1].get_pixmap(dpi=200, alpha=False).save(
                out / "pages" / f"{number}.png"
            )
    return edition


def read_pages(edition):
    """Original pages that were read; editions made before page sorting read all."""
    return edition.get("source_pages_read") or list(
        range(1, edition["source_page_count"] + 1)
    )


def page_ranges(pages):
    """[1, 2, 3, 5] -> '1–3, 5'."""
    out, start = [], None
    for i, p in enumerate(pages):
        if start is None:
            start = p
        if i + 1 == len(pages) or pages[i + 1] != p + 1:
            out.append(str(start) if start == p else f"{start}–{p}")
            start = None
    return ", ".join(out)


def quality_notes_html(edition):
    return "".join(
        '<p class="notice">' + html.escape(n) + "</p>"
        for n in edition.get("quality_notes", [])
    )


def attribution(edition):
    authors = ", ".join(edition["authors"]) or "See the original paper for its authors"
    doi = edition["doi"]
    link = (
        f'<a href="https://doi.org/{html.escape(doi)}">{html.escape(doi)}</a>'
        if doi
        else "See original PDF"
    )
    version = (
        f" Archived source version {edition['source_version']}; its text matches the rewrite input."
        if edition.get("source_version")
        else ""
    )
    return (
        f'<p class="meta">Adapted from <em>{html.escape(edition["original_title"])}</em>. '
        f"{html.escape(authors)}. {html.escape(edition['year'] or '')}. {link}. "
        f"Original licence: {html.escape(edition['license'])}. Wording simplified by ScienceMadeReadable; "
        f"this is not the publisher’s original text.{version} Original page images are included for checking.</p>"
    )


def xhtml(title, body, css="style.css"):
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en" xml:lang="en">'
        f'<head><title>{html.escape(title)}</title><link rel="stylesheet" type="text/css" href="{css}"/></head>'
        f"<body>{body}</body></html>"
    )


def make_epub(edition, out):
    files = {}
    intro = (
        f"<h1>{html.escape(edition['title'])}</h1><p>{html.escape(NOTICE)}</p>"
        + attribution(edition)
    )
    intro += (
        "<p>Reflowable reading edition with original figures and tables alongside the rewritten passages. Images retain source labels and values. The full original-page appendix is also included.</p>"
        if edition.get("visuals")
        else "<p>Reflowable reading edition. Generated table cells are omitted; consult the original page appendix for tables and figures.</p>"
    )
    files["front.xhtml"] = xhtml(
        "About this edition", intro + quality_notes_html(edition)
    )
    toc = [("front.xhtml", "About this edition")]
    for c in edition["chapters"]:
        links = ", ".join(
            f'<a href="page-{p}.xhtml">{p}</a>' for p in c["source_pages"]
        )
        body = f'<section><h1>{html.escape(c["title"])}</h1><p class="source">Original pages: {links}.</p>{html_for_format(c["html"], "epub")}</section>'
        files[c["id"] + ".xhtml"] = xhtml(c["title"], body)
        toc.append((c["id"] + ".xhtml", c["title"]))
    for p in read_pages(edition):
        files[f"page-{p}.xhtml"] = xhtml(
            f"Original page {p}",
            f'<h1>Original page {p}</h1><p class="source">Unchanged source page image; enlarge in your reading app to inspect small numbers.</p><img src="pages/{p}.png" alt="Original research paper, page {p}"/>',
        )
        toc.append((f"page-{p}.xhtml", f"Original page {p}"))
    for visual in edition.get("visuals", []):
        for part in visual["parts"]:
            body = (
                f"<h1>{html.escape(visual_title(visual, part))}</h1>"
                f"<p>Original PDF page {part['page']}. Enlarge in your reading app to inspect small labels.</p>"
                f'<p><a href="{visual["chapter"]}.xhtml#visual-{part["id"]}">Return to the passage</a></p>'
                f'<img src="{part["image"]}" alt="{html.escape(visual["caption"], quote=True)}"/>'
            )
            files[f"visual-{part['id']}.xhtml"] = xhtml(
                visual["label"] + " — original", body
            )
    nav = (
        '<nav epub:type="toc" id="toc"><h1>Contents</h1><ol>'
        + "".join(f'<li><a href="{f}">{html.escape(t)}</a></li>' for f, t in toc)
        + "</ol></nav>"
    )
    files["nav.xhtml"] = xhtml("Contents", nav)
    files["style.css"] = (
        "body{line-height:1.55}p{orphans:2;widows:2}h1,h2,h3,h4{page-break-after:avoid}img{max-width:100%;height:auto}.source{font-size:.85em}pre{white-space:pre-wrap}a{color:inherit}sup,sub{font-size:.8em}"
    )
    files["style.css"] += (
        ".source-visual{margin:1em 0;break-inside:avoid;break-before:page;break-after:page}.source-visual img{display:block;max-height:75vh;width:auto;max-width:100%;object-fit:contain}.source-visual figcaption{font-size:.85em;margin-bottom:.6em}"
    )
    manifest = [
        '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
        '<item id="css" href="style.css" media-type="text/css"/>',
    ]
    spine = []
    for i, (f, _) in enumerate(toc):
        manifest.append(
            f'<item id="s{i}" href="{f}" media-type="application/xhtml+xml"/>'
        )
        spine.append(f'<itemref idref="s{i}"/>')
    for p in read_pages(edition):
        manifest.append(
            f'<item id="p{p}" href="pages/{p}.png" media-type="image/png"/>'
        )
    for visual in edition.get("visuals", []):
        for part in visual["parts"]:
            manifest.append(
                f'<item id="image-{part["id"]}" href="{part["image"]}" media-type="image/png"/>'
            )
            manifest.append(
                f'<item id="view-{part["id"]}" href="visual-{part["id"]}.xhtml" media-type="application/xhtml+xml"/>'
            )
            spine.append(f'<itemref idref="view-{part["id"]}" linear="no"/>')
    creators = "".join(
        f"<dc:creator>{html.escape(a)}</dc:creator>" for a in edition["authors"]
    )
    modified = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    opf = (
        '<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f'<dc:identifier id="book-id">urn:sha256:{edition["id"]}</dc:identifier><dc:title>{html.escape(edition["title"])}</dc:title>'
        f"<dc:language>en</dc:language>{creators}<dc:contributor>ScienceMadeReadable — simplified edition</dc:contributor>"
        f"<dc:source>{html.escape('https://doi.org/' + edition['doi'] if edition['doi'] else 'Private PDF upload')}</dc:source><dc:rights>Original: {html.escape(edition['license'])}; simplified adaptation, not fully fact-checked.</dc:rights>"
        f'<meta property="dcterms:modified">{modified}</meta></metadata><manifest>{"".join(manifest)}</manifest><spine>{"".join(spine)}</spine></package>'
    )
    with zipfile.ZipFile(out / "paper.epub", "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="EPUB/package.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
            compress_type=zipfile.ZIP_DEFLATED,
        )
        z.writestr("EPUB/package.opf", opf, compress_type=zipfile.ZIP_DEFLATED)
        for name, content in files.items():
            if name.endswith(".xhtml"):
                ET.fromstring(content)
            z.writestr("EPUB/" + name, content, compress_type=zipfile.ZIP_DEFLATED)
        for p in read_pages(edition):
            z.write(
                out / "pages" / f"{p}.png",
                f"EPUB/pages/{p}.png",
                compress_type=zipfile.ZIP_DEFLATED,
            )
        for visual in edition.get("visuals", []):
            for part in visual["parts"]:
                z.write(
                    out / part["image"],
                    "EPUB/" + part["image"],
                    compress_type=zipfile.ZIP_DEFLATED,
                )
    print("EPUB written", flush=True)


def make_print(edition, out):
    front = (
        f'<section class="cover"><p>ScienceMadeReadable · reading edition</p><h1>{html.escape(edition["title"])}</h1><p class="notice">{html.escape(NOTICE)}</p>'
        + attribution(edition)
    )
    front += (
        '<p class="meta">Large-print A5 text, with original figures and tables inserted on larger A4 pages at the relevant passages. The original PDF is also appended for reference.</p><h2>Contents</h2><ol>'
        if edition.get("visuals")
        else '<p class="meta">Large-print A5 edition. Generated tables are omitted. The unchanged original PDF is appended after the rewrite, preserving its tables and figures.</p><h2>Contents</h2><ol>'
    )
    front += (
        "".join(
            f'<li><a href="#{c["id"]}">{html.escape(c["title"])}</a></li>'
            for c in edition["chapters"]
        )
        + "</ol>"
        + quality_notes_html(edition)
        + "</section>"
    )
    body = "".join(
        f'<section class="chapter" id="{c["id"]}"><h2>{html.escape(c["title"])}</h2><p class="meta">Original paper pages {", ".join(map(str, c["source_pages"]))}.</p>{html_for_format(c["html"], "pdf")}</section>'
        for c in edition["chapters"]
    )
    (out / "print.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>'
        + html.escape(edition["title"])
        + "</title><style>"
        + PRINT_CSS
        + "</style></head><body>"
        + front
        + body
        + "</body></html>"
    )


def make_pdf(edition, out, chromium):
    from playwright.sync_api import sync_playwright
    import pymupdf

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=chromium or shutil.which("chromium"), headless=True
        )
        page = browser.new_page()
        page.goto((out / "print.html").resolve().as_uri(), wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        page.pdf(
            path=str(out / "rewrite-only.pdf"),
            prefer_css_page_size=True,
            print_background=False,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template='<div style="width:100%;font-size:9px;text-align:center;color:var(--color-text-secondary,#555)"><span class="pageNumber"></span> · ScienceMadeReadable · simplified edition</div>',
        )
        browser.close()
    with (
        pymupdf.open(out / "rewrite-only.pdf") as doc,
        pymupdf.open(out / "original.pdf") as original,
    ):
        # Preserve visible annotations/form values, not interactive actions, in
        # the combined edition. The separate original.pdf stays byte-for-byte intact.
        original.bake(annots=True, widgets=True)
        # Chromium determines the mixed-size layout. Replace its PNG placements
        # with the source PDF content so table text stays sharp at any zoom.
        for visual in edition.get("visuals", []):
            for part in visual["parts"]:
                title = visual_title(visual, part) + " — original"
                matches = [
                    p
                    for p in doc
                    if title in " ".join(p.get_text().split())
                    and f"Source PDF page {part['page']}."
                    in " ".join(p.get_text().split())
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Cannot uniquely locate the PDF placement for {part['id']}"
                    )
                page = matches[0]
                images = [
                    i
                    for i in page.get_image_info(xrefs=True)
                    if i["width"] == part["width"] and i["height"] == part["height"]
                ]
                if len(images) != 1 or not images[0]["xref"]:
                    raise ValueError(
                        f"Cannot locate source-image placement for {part['id']}"
                    )
                rect = pymupdf.Rect(images[0]["bbox"])
                page.delete_image(images[0]["xref"])
                page.show_pdf_page(
                    rect, original, part["page"] - 1, clip=pymupdf.Rect(part["rect"])
                )
        start = len(doc)
        doc.insert_pdf(original, links=False, annots=False, widgets=False)
        toc = [
            [1, "Simplified research paper", 1],
            [1, "Original paper — tables, figures and references", start + 1],
        ]
        toc += [
            [2, f"Original page {i + 1}", start + i + 1] for i in range(len(original))
        ]
        doc.set_toc(toc)
        doc.set_metadata(
            {
                "title": edition["title"],
                "author": ", ".join(edition["authors"]),
                "subject": NOTICE,
            }
        )
        doc.save(out / "paper.pdf", garbage=4, deflate=True)
        print(
            f"PDF written: {start} reading pages + {len(original)} unchanged original pages",
            flush=True,
        )
    (out / "rewrite-only.pdf").unlink()


def tts_chunk(text, cache, service, voice, speed):
    payload = {"text": text, "voice": voice, "speed": speed}
    key = sha(
        json.dumps(
            {
                "request": payload,
                "service": service,
                "pcm_contract": "kokoro-s16le-24k-mono-v1",
            },
            sort_keys=True,
        )
    )
    path = cache / f"{key}.pcm"
    info = cache / f"{key}.json"
    if path.exists() and info.exists():
        record = json.loads(info.read_text())
        if (
            path.stat().st_size == record["bytes"]
            and sha(path.read_bytes()) == record["sha256"]
        ):
            return path, record, True
    start = time.monotonic()
    tmp = path.with_suffix(".part")
    try:
        with requests.post(
            service.rstrip("/") + "/tts", json=payload, stream=True, timeout=(10, 180)
        ) as r:
            r.raise_for_status()
            if (
                r.headers.get("X-Sample-Rate") != "24000"
                or r.headers.get("X-Channels") != "1"
                or r.headers.get("X-Format") != "s16le"
            ):
                raise ValueError(
                    "Unexpected Kokoro PCM format; refusing to encode invalid audio."
                )
            with tmp.open("wb") as f:
                for chunk in r.iter_content(64 * 1024):
                    f.write(chunk)
                    if f.tell() > 48_000 * 300:
                        raise ValueError(
                            "Single audio chunk exceeded five minutes; inspect before continuing."
                        )
        size = tmp.stat().st_size
        if size < 2400 or size % 2:
            raise ValueError("Kokoro returned empty, too-short or malformed PCM.")
        record = {
            "request": payload,
            "bytes": size,
            "sha256": sha(tmp.read_bytes()),
            "audio_seconds": size / 48_000,
            "synthesis_seconds": time.monotonic() - start,
        }
        tmp.replace(path)
        save(info, record)
        return path, record, False
    finally:
        tmp.unlink(missing_ok=True)


def _ffmpeg_mp3(source_args, target, title, stdin=None):
    tmp = target.with_suffix(".tmp.mp3")
    command = (
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        + source_args
        + [
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "64k",
            "-metadata",
            "title=" + title,
            "-metadata",
            "artist=ScienceMadeReadable · Kokoro",
            str(tmp),
        ]
    )
    return subprocess.Popen(command, stdin=stdin), tmp


def encode(pcm, target, title):
    process, tmp = _ffmpeg_mp3(
        ["-f", "s16le", "-ar", "24000", "-ac", "1", "-i", str(pcm)], target, title
    )
    if process.wait():
        raise RuntimeError("ffmpeg could not encode the audio.")
    tmp.replace(target)


SPOKEN_CHARS_PER_SECOND = 14  # Kokoro bm_george at speed 1.0 (measured 14.2)


def make_audio(edition, run, out, service, voice, speed):
    """Chapter MP3s are encoded while Kokoro speaks, straight from its cached
    chunks; the whole-paper MP3 joins the chapter files. No full-length raw audio
    is written, so a four-hour narration needs about 0.2 GB rather than 1.5 GB."""
    cache = run / "audio-cache"
    cache.mkdir(exist_ok=True)
    audio = out / "audio"
    audio.mkdir(exist_ok=True)
    chapters = [
        {
            "id": "about",
            "title": "About this recording",
            "speech": edition["title"] + ".\n\n" + AUDIO_NOTICE,
        }
    ] + edition["chapters"]
    limit = float(os.environ.get("SMR_AUDIO_MAX_SECONDS", "14400"))
    estimate = sum(len(c["speech"]) for c in chapters) / (SPOKEN_CHARS_PER_SECOND * speed)
    if estimate > limit:
        raise ValueError(
            f"The narration would last about {estimate / 3600:.1f} hours; the audio limit is {limit / 3600:.1f} hours. The reading formats remain available."
        )
    records, tracks = [], []
    started = time.monotonic()
    total = 0  # bytes of 24 kHz 16-bit mono written so far
    for index, c in enumerate(chapters):
        chunks = split_speech(c["speech"])
        filename = f"{index:02d}-{c['id']}.mp3"
        process, tmp = _ffmpeg_mp3(
            ["-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0"],
            audio / filename,
            c["title"],
            stdin=subprocess.PIPE,
        )
        written, used = 0, []
        try:
            for j, text in enumerate(chunks):
                source, record, cached = tts_chunk(text, cache, service, voice, speed)
                used.append(source)
                records.append(
                    {
                        **{k: v for k, v in record.items() if k != "request"},
                        "cached": cached,
                        "chapter": c["id"],
                    }
                )
                if (total + written + source.stat().st_size) / 48000 > limit:
                    raise ValueError(
                        "Audio exceeds the current duration limit; the reading formats remain available."
                    )
                data = source.read_bytes() + b"\x00" * 12_000  # quarter-second pause
                process.stdin.write(data)
                written += len(data)
                print(
                    f"Kokoro {c['id']} {j + 1}/{len(chunks)} {'cached' if cached else ''}",
                    flush=True,
                )
            process.stdin.write(b"\x00" * 48_000)  # one second between chapters
            process.stdin.close()
            if process.wait():
                raise RuntimeError("ffmpeg could not encode the chapter audio.")
            tmp.replace(audio / filename)
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            tmp.unlink(missing_ok=True)
            raise
        # The chapter is safely encoded; its raw speech is no longer needed.
        for source in used:
            source.unlink(missing_ok=True)
        tracks.append(
            {
                "id": c["id"],
                "title": c["title"],
                "file": "audio/" + filename,
                "start_seconds": total / 48_000,
                "seconds": (written - 12_000) / 48_000,
            }
        )
        total += written + 48_000
    listing = cache / "chapters.txt"
    listing.write_text(
        "".join(f"file '{(audio / t['file'][6:]).resolve()}'\n" for t in tracks)
    )
    # Joined without re-encoding (no second quality loss). Each chapter keeps its
    # encoder padding, so later chapters start up to ~0.05 s late per chapter.
    tmp = out / "paper.tmp.mp3"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
         "-safe", "0", "-i", str(listing), "-c", "copy",
         "-metadata", "title=" + edition["title"],
         "-metadata", "artist=ScienceMadeReadable · Kokoro", str(tmp)],
        check=True,
    )
    tmp.replace(out / "paper.mp3")
    listing.unlink()
    result = {
        "edition_id": edition["id"],
        "voice": voice,
        "speed": speed,
        "sample_rate": 24000,
        "tracks": tracks,
        "seconds": total / 48_000,
        "build_seconds": time.monotonic() - started,
        "chunks": len(records),
        "new_synthesis_chunks": sum(not r["cached"] for r in records),
        "synthesis_seconds": sum(r["synthesis_seconds"] for r in records),
        "records": records,
        "transcript": "narration.txt",
        "note": "Not listened through end to end. Pronunciation of technical terms needs review.",
    }
    save(out / "audio.json", result)
    (out / "chapters.txt").write_text(
        "\n".join(
            f"{int(t['start_seconds']) // 3600:02}:{int(t['start_seconds']) // 60 % 60:02}:{int(t['start_seconds']) % 60:02} {t['title']}"
            for t in tracks
        )
        + "\n"
    )
    return result


def install_reader(edition, out, audio):
    shutil.copyfile(HERE / "reader.html", out / "index.html")
    data = (
        json.dumps(
            {
                "edition": edition,
                "audio": audio,
                "downloads": {"pdf": (out / "paper.pdf").exists(), "bundle": True},
            },
            ensure_ascii=False,
        )
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    template = (out / "index.html").read_text()
    (out / "index.html").write_text(
        template.replace("/* READING_DATA */", "window.READING = " + data + ";")
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=HERE / "reading")
    p.add_argument("--audio", action="store_true")
    p.add_argument("--pdf", action="store_true")
    p.add_argument(
        "--visuals",
        type=Path,
        help="Reviewed original-PDF crop/placement manifest; saved with the run for subsequent exports",
    )
    p.add_argument("--voice", default="bm_george")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--tts-url", default="http://127.0.0.1:8880")
    p.add_argument("--chromium", default=os.environ.get("CHROMIUM_BIN"))
    a = p.parse_args()
    if not 0.5 <= a.speed <= 2 or not re.fullmatch(r"[ab][fm]_[a-z]+", a.voice):
        p.error("Use an English Kokoro voice and speed between 0.5 and 2.")
    a.out.mkdir(parents=True, exist_ok=True)
    edition = make_edition(a.run, a.out, a.visuals, regenerate_pdf=a.pdf)
    make_epub(edition, a.out)
    make_print(edition, a.out)
    if a.pdf:
        make_pdf(edition, a.out, a.chromium)
    audio = (
        make_audio(edition, a.run, a.out, a.tts_url, a.voice, a.speed)
        if a.audio
        else None
    )
    if audio is None and (a.out / "audio.json").exists():
        audio = json.loads((a.out / "audio.json").read_text())
        if audio.get("edition_id") != edition["id"]:
            raise ValueError("Existing audio belongs to another edition.")
    install_reader(edition, a.out, audio)
    with zipfile.ZipFile(
        a.out / "reading.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for file in sorted(a.out.rglob("*")):
            if file.is_file() and file.name != "reading.zip":
                archive.write(file, file.relative_to(a.out))
    print(f"Reading edition: {a.out}", flush=True)
