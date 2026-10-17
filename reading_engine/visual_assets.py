"""Reviewed original-PDF crops and their placement in the visual reading editions.

No table cells are reconstructed. PDF content, including vector text and embedded
figures, is rendered directly. Coordinates and continuations must be reviewed;
this is not a general automatic figure/table detector.
"""

from __future__ import annotations
import hashlib
import html
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import pymupdf

TABLE_PLACEHOLDER = "[Original table: consult the source pages. Generated table cells are not reproduced in this edition.]"


def digest(value):
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def caption_candidates(pdf):
    """Inventory only: native PDF caption starts, not inferred crop boundaries."""
    found = []
    with pymupdf.open(pdf) as doc:
        for i, page in enumerate(doc):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    match = re.match(
                        r"^(Table|Figure|Fig\.?)\s*(\d+)[.:|]\s", text, re.I
                    )
                    if match:
                        kind = "Table" if match[1].lower() == "table" else "Figure"
                        found.append(
                            {
                                "label": f"{kind} {match[2]}",
                                "source_label": text[: match.end()].strip(),
                                "page": i + 1,
                                "text": text,
                                "bbox": list(line["bbox"]),
                            }
                        )
    return found


def find_anchor(soup, visual):
    if visual.get("anchor_key"):
        # Chosen with Jev's caption labels (caption_labels.py): the paragraph
        # whose start matches the rewritten caption, punctuated or not.
        from caption_labels import key

        matches = [
            n
            for n in soup.find_all(["p", "h2", "h3", "h4"])
            if key(n.get_text(" ", strip=True)) == visual["anchor_key"]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{visual['id']}: expected one caption anchor, found {len(matches)}; inspect placement."
            )
        return matches[0]
    number = re.escape(visual["label"].split()[-1])
    kind = r"(?:Figure|Fig\.?)" if visual["label"].startswith("Figure") else "Table"
    pattern = re.compile(r"^" + kind + r"\s*" + number + r"[.:|](?:\s|$)", re.I)
    matches = [
        n
        for n in soup.find_all(["p", "h2", "h3", "h4"])
        if pattern.match(normalized(n.get_text(" ", strip=True)))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{visual['id']}: expected one caption anchor, found {len(matches)}; inspect placement."
        )
    return matches[0]


def validate_manifest(manifest, pdf, rewrite_id, chapters):
    if manifest.get("version") != 1 or not manifest.get("review"):
        raise ValueError("Visual manifest needs version 1 and an explicit review note.")
    if manifest.get("pdf_sha256") != digest(Path(pdf).read_bytes()):
        raise ValueError("Visual manifest belongs to a different source PDF.")
    if manifest.get("rewrite_sha256") != rewrite_id:
        raise ValueError("Visual placement was reviewed for a different rewrite.")
    chapter_ids = {c["id"] for c in chapters}
    ids, labels = set(), set()
    with pymupdf.open(pdf) as doc:
        for visual in manifest.get("visuals", []):
            vid = visual.get("id", "")
            if not re.fullmatch(r"[a-z]+-\d+", vid) or vid in ids:
                raise ValueError(
                    "Visual IDs must be unique safe names such as table-1."
                )
            ids.add(vid)
            if (
                visual.get("kind") not in {"table", "figure"}
                or visual.get("chapter") not in chapter_ids
            ):
                raise ValueError(f"Invalid kind or chapter for {vid}")
            label = visual.get("label", "")
            if label in labels or not re.fullmatch(r"(?:Table|Figure) \d+", label):
                raise ValueError(
                    "Source labels must be unique numbered figures/tables."
                )
            labels.add(label)
            if visual.get("anchor_key") is not None and (
                not isinstance(visual["anchor_key"], str)
                or not 1 <= len(visual["anchor_key"]) <= 80
                or not re.fullmatch(r"[a-z0-9 ]+", visual["anchor_key"])
            ):
                raise ValueError(f"Invalid caption anchor for {vid}")
            if not visual.get("caption") or visual.get("anchor_prefix") != label + ".":
                raise ValueError(
                    f"Caption and exact numbered anchor required for {vid}"
                )
            if not visual.get("parts"):
                raise ValueError(f"{vid} has no source parts")
            for side in ("before", "after"):
                n = visual.get("placeholders_" + side, 0)
                if type(n) is not int or not 0 <= n <= 20:
                    raise ValueError("Invalid placeholder count")
            previous_page = 0
            for part in visual["parts"]:
                number = part.get("page")
                if (
                    type(number) is not int
                    or not 1 <= number <= len(doc)
                    or number < previous_page
                ):
                    raise ValueError(
                        "Visual source pages must be in bounds and in order"
                    )
                previous_page = number
                page = doc[number - 1]
                if page.rotation:
                    raise ValueError(
                        "Rotated source pages need a separately reviewed coordinate convention."
                    )
                coords = part.get("rect", [])
                if len(coords) != 4 or not all(
                    isinstance(x, (float, int)) and math.isfinite(x) for x in coords
                ):
                    raise ValueError("Crop must have four finite coordinates")
                rect = pymupdf.Rect(coords)
                if (
                    rect.is_empty
                    or not page.rect.contains(rect)
                    or rect.get_area() * (300 / 72) ** 2 > 20_000_000
                ):
                    raise ValueError(
                        "Crop is empty, outside the source page, or too large"
                    )
                native = normalized(page.get_text(clip=rect))
                for marker in part.get("checks", []):
                    if normalized(marker) not in native:
                        raise ValueError(
                            f"{vid}, page {number}: source marker not inside crop: {marker!r}"
                        )
                if "contained_images" in part:
                    contained = sum(
                        rect.contains(pymupdf.Rect(x["bbox"]))
                        for x in page.get_image_info()
                    )
                    if contained != part["contained_images"]:
                        raise ValueError(
                            f"{vid}: crop does not contain the expected complete source images"
                        )
    candidates = caption_candidates(pdf)
    if manifest.get("require_all_native_captions", False) and labels != {
        c["label"] for c in candidates
    }:
        raise ValueError(
            "Reviewed assets do not match the source PDF's numbered caption inventory."
        )
    # Validate all anchors before writing any assets.
    for visual in manifest["visuals"]:
        chapter = next(c for c in chapters if c["id"] == visual["chapter"])
        anchor = find_anchor(BeautifulSoup(chapter["html"], "html.parser"), visual)
        for side in ("before", "after"):
            cursor = anchor
            for _ in range(visual.get("placeholders_" + side, 0)):
                cursor = (
                    cursor.find_previous_sibling()
                    if side == "before"
                    else cursor.find_next_sibling()
                )
                if cursor is None or normalized(cursor.get_text()) != TABLE_PLACEHOLDER:
                    raise ValueError(
                        f"{visual['id']}: missing adjacent placeholder {side} caption; inspect placement."
                    )
    if manifest.get("require_all_placeholders", False):
        expected = sum(
            v.get("placeholders_before", 0) + v.get("placeholders_after", 0)
            for v in manifest["visuals"]
        )
        actual = sum(c["html"].count(TABLE_PLACEHOLDER) for c in chapters)
        if expected != actual:
            raise ValueError(
                "Visual manifest does not account for every generated-table placeholder."
            )
    return candidates


def visual_title(visual, part):
    unit = "source page" if part.get("scope") == "source-page" else "part"
    return visual["label"] + (
        f" · {unit} {part['part']} of {part['parts']}" if part["parts"] > 1 else ""
    )


def part_html(visual, part):
    title = visual_title(visual, part)
    alt = f"Original {title}, page {part['page']}. {visual['caption']}"
    context = (
        "Whole source page; precise table or figure boundaries were not inferred. "
        if part.get("scope") == "source-page"
        else ""
    )
    orientation = "landscape" if part["width"] / part["height"] > 1.2 else "portrait"
    return (
        f'<figure class="source-visual {orientation}" id="visual-{part["id"]}" data-visual="{part["id"]}">'
        f"<figcaption><strong>{html.escape(title)} — original</strong><br/>Source PDF page {part['page']}. "
        f'{context}Original labels, values and caption preserved. <span class="visual-hint">Tap image to enlarge.</span></figcaption>'
        f'<a class="visual-image" href="{part["image"]}">'
        f'<img src="{part["image"]}" width="{part["width"]}" height="{part["height"]}" alt="{html.escape(alt, quote=True)}"/></a>'
        "</figure>"
    )


def add_visuals(pdf, rewrite_id, chapters, out, manifest):
    candidates = validate_manifest(manifest, pdf, rewrite_id, chapters)
    assets = Path(out) / "assets"
    assets.mkdir(exist_ok=True)
    visuals = []
    with pymupdf.open(pdf) as doc:
        for spec in manifest["visuals"]:
            visual = {k: spec[k] for k in ("id", "kind", "label", "chapter", "caption")}
            visual["parts"] = []
            for index, location in enumerate(spec["parts"]):
                asset_id = f"{spec['id']}-part-{index + 1}"
                rect = pymupdf.Rect(location["rect"])
                page = doc[location["page"] - 1]
                pix = page.get_pixmap(dpi=300, clip=rect, alpha=False)
                image_file = assets / (asset_id + ".png")
                pix.save(image_file)
                with pymupdf.open() as crop:
                    target = crop.new_page(width=rect.width, height=rect.height)
                    target.show_pdf_page(
                        target.rect, doc, location["page"] - 1, clip=rect
                    )
                    crop.set_metadata(
                        {
                            "title": spec["label"]
                            + f" — original page {location['page']}"
                        }
                    )
                    crop.save(assets / (asset_id + ".pdf"), garbage=4, deflate=True)
                visual["parts"].append(
                    {
                        "id": asset_id,
                        "page": location["page"],
                        "rect": location["rect"],
                        "part": index + 1,
                        "parts": len(spec["parts"]),
                        "scope": location.get("scope", "reviewed-crop"),
                        "width": pix.width,
                        "height": pix.height,
                        "dpi": 300,
                        "image": "assets/" + asset_id + ".png",
                        "pdf": "assets/" + asset_id + ".pdf",
                        "image_sha256": digest(image_file.read_bytes()),
                    }
                )
            visuals.append(visual)
    placements = []
    for chapter in chapters:
        soup = BeautifulSoup(chapter["html"], "html.parser")
        for spec, visual in zip(manifest["visuals"], visuals):
            if spec["chapter"] != chapter["id"]:
                continue
            anchor = find_anchor(soup, spec)
            removed = 0
            for side in ("before", "after"):
                for _ in range(spec.get("placeholders_" + side, 0)):
                    node = (
                        anchor.find_previous_sibling()
                        if side == "before"
                        else anchor.find_next_sibling()
                    )
                    if node is None or normalized(node.get_text()) != TABLE_PLACEHOLDER:
                        raise ValueError(
                            f"{spec['id']}: expected adjacent generated-table placeholder {side} caption; inspect placement."
                        )
                    node.decompose()
                    removed += 1
            cursor = anchor
            for part in visual["parts"]:
                card = BeautifulSoup(part_html(visual, part), "html.parser").figure
                cursor.insert_after(card)
                cursor = card
            placements.append(
                {
                    "id": spec["id"],
                    "chapter": chapter["id"],
                    "caption_anchor": normalized(anchor.get_text(" ", strip=True)),
                    "replaced_placeholder_blocks": removed,
                }
            )
        chapter["html"] = soup.decode(formatter="minimal")
        ET.fromstring("<div>" + chapter["html"] + "</div>")
        if (
            manifest.get("require_all_placeholders", False)
            and TABLE_PLACEHOLDER in chapter["html"]
        ):
            raise ValueError(
                "Some generated-table placeholders were not matched to reviewed originals."
            )
    audit = {
        "manifest": manifest,
        "manifest_sha256": digest(json.dumps(manifest, sort_keys=True)),
        "visuals": visuals,
        "placements": placements,
        "native_caption_inventory": candidates,
        "method": "Reviewed PDF-point rectangles, direct 300-dpi source rendering and vector PDF crops. No OCR cells or regenerated plots.",
    }
    (Path(out) / "visuals.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    return visuals, audit["manifest_sha256"]


def html_for_format(fragment, target):
    soup = BeautifulSoup(fragment, "html.parser")
    for hint in soup.select(".visual-hint"):
        if target == "pdf":
            hint.decompose()
        elif target == "epub":
            hint.string = "Open the image to inspect it."
    for anchor in soup.select("a.visual-image"):
        if target == "pdf":
            anchor.unwrap()
        elif target == "epub":
            anchor["href"] = "visual-" + anchor.parent["data-visual"] + ".xhtml"
    return soup.decode(formatter="minimal")
