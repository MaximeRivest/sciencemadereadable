"""Conservative automatic source visuals: crop a clearly paired embedded figure;
otherwise keep the original whole page. Never guess a table's cell boundaries.
"""

import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup
import pymupdf
from visual_assets import caption_candidates, TABLE_PLACEHOLDER, normalized, find_anchor


def native_continuations(doc, page_number):
    pages = [page_number]
    for index in range(page_number, min(len(doc), page_number + 4)):
        page = doc[index]
        blocks = sorted(
            (
                b
                for b in page.get_text("blocks")
                if b[1] > 75 and b[3] < page.rect.height - 35 and b[4].strip()
            ),
            key=lambda b: (b[1], b[0]),
        )
        if not blocks:
            break
        first = blocks[0][4]
        words = first.split()
        numeric = sum(bool(re.search(r"\d", w)) for w in words) / max(1, len(words))
        if numeric < 0.3 or re.match(
            r"^\s*(?:\d+\.)+\s+[A-Z]|\s*(?:Figure|Table)\s+\d+\.", first
        ):
            break
        pages.append(index + 1)
    return pages


def manifest_for(run, chapters, rewrite_id):
    pdf = Path(run) / "original.pdf"
    labelled = json.loads((Path(run) / "labelled.json").read_text())
    ledger = labelled.get("ledger", [])
    candidates = caption_candidates(pdf)
    # Scanned pages have no native text. Keep OCR's page provenance, but use a
    # whole-page panel rather than pretending OCR supplied a reliable crop.
    for b in ledger:
        text = re.sub(r"^[#*\s]+", "", b["text"])
        match = re.match(r"^(Table|Figure)\s+(\d+)\.", text)
        if match and not any(
            c["label"] == f"{match[1]} {match[2]}" for c in candidates
        ):
            candidates.append(
                {
                    "label": f"{match[1]} {match[2]}",
                    "page": b["page"],
                    "text": text[:400],
                    "bbox": None,
                }
            )
    candidates.sort(key=lambda c: (c["page"], (c["bbox"] or [0, 0])[1]))
    specs, used = [], set()
    with pymupdf.open(pdf) as doc:
        for candidate in candidates:
            label, number = candidate["label"], candidate["page"]
            if label in used or len(specs) >= 24:
                continue
            used.add(label)
            chapter, anchor = None, None
            for c in chapters:
                soup = BeautifulSoup(c["html"], "html.parser")
                try:
                    hit = find_anchor(
                        soup,
                        {"id": label, "label": label, "anchor_prefix": label + "."},
                    )
                except ValueError:
                    continue
                chapter, anchor = c, hit
                break
            if chapter is None:
                # Preserve availability without inventing a position in the prose.
                destination = next(
                    (
                        b.get("destination")
                        for b in ledger
                        if re.match(r"^[#*\s]*" + re.escape(label) + r"\.", b["text"])
                    ),
                    None,
                )
                chapter = next(
                    (c for c in chapters if c["id"] == destination), chapters[-1]
                )
                soup = BeautifulSoup(chapter["html"], "html.parser")
                anchor = soup.new_tag("p")
                anchor.string = (
                    label
                    + ". Original source visual; shown here because its caption could not be matched confidently to the rewrite."
                )
                soup.append(anchor)
                chapter["html"] = soup.decode(formatter="minimal")
            before = after = 0
            cursor = anchor.find_previous_sibling()
            while cursor and normalized(cursor.get_text()) == TABLE_PLACEHOLDER:
                before += 1
                cursor = cursor.find_previous_sibling()
            cursor = anchor.find_next_sibling()
            while cursor and normalized(cursor.get_text()) == TABLE_PLACEHOLDER:
                after += 1
                cursor = cursor.find_next_sibling()
            page = doc[number - 1]
            if page.rotation:
                continue  # full original remains in the appendix; no rotated crop guess
            rect = page.rect
            scope = "source-page"
            if label.startswith("Figure") and candidate["bbox"]:
                cap = pymupdf.Rect(candidate["bbox"])
                for block in page.get_text("blocks"):
                    if (
                        normalized(block[4]).startswith(
                            normalized(candidate["text"])[:40]
                        )
                        and abs(block[1] - cap.y0) < 3
                    ):
                        cap = pymupdf.Rect(block[:4])
                options = [
                    pymupdf.Rect(im["bbox"])
                    for im in page.get_image_info()
                    if 0 <= cap.y0 - im["bbox"][3] <= 25
                    and im["bbox"][2] > cap.x0
                    and im["bbox"][0] < cap.x1
                ]
                if len(options) == 1 and cap.height < 70:
                    union = options[0] | cap
                    proposed = (union + (-3, -3, 3, 3)) & page.rect
                    # Do not crop through another image on the page.
                    if all(
                        not proposed.intersects(pymupdf.Rect(im["bbox"]))
                        or proposed.contains(pymupdf.Rect(im["bbox"]))
                        for im in page.get_image_info()
                    ):
                        rect, scope = proposed, "paired-figure"
            pages = [number]
            if label.startswith("Table"):
                matches = [
                    i
                    for i, b in enumerate(ledger)
                    if re.match(r"^[#*\s]*" + re.escape(label) + r"\.", b["text"])
                ]
                if matches:
                    for b in ledger[matches[0] + 1 :]:
                        if b["role"] == "heading" or re.match(
                            r"^[#*\s]*(Table|Figure)\s+\d+\.", b["text"]
                        ):
                            break
                        if b["role"] == "table" and b["page"] not in pages:
                            pages.append(b["page"])
                else:
                    pages = native_continuations(doc, number)
            parts = [
                {
                    "page": p,
                    "rect": list(rect if p == number else doc[p - 1].rect),
                    "scope": scope if p == number else "source-page",
                    "checks": [candidate.get("source_label", label + ".")]
                    if p == number and candidate["bbox"]
                    else [],
                }
                for p in pages
            ]
            specs.append(
                {
                    "id": label.lower().replace(" ", "-"),
                    "kind": "figure" if label.startswith("Figure") else "table",
                    "label": label,
                    "caption": candidate["text"],
                    "chapter": chapter["id"],
                    "anchor_prefix": label + ".",
                    "placeholders_before": before,
                    "placeholders_after": after,
                    "parts": parts,
                }
            )
    return {
        "version": 1,
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "rewrite_sha256": rewrite_id,
        "review": "Automatic conservative placement. Whole source pages are used for tables and uncertain boundaries; no manual crop approval is claimed.",
        "automatic": True,
        "visuals": specs,
    }
