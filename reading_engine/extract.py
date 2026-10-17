"""Render a bounded local PDF, then read it through the isolated Attachments service.

No native PDF text is used in the downstream rewrite. The native extraction is
saved only as a diagnostic. Image-only rendering makes Attachments OCR every page
without needing a library change or access to the model's private socket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import pymupdf
import requests

VERSION = 2
MAX_BYTES = 25 * 1024 * 1024
MAX_PAGES = 200


from io_utils import save


THUMB_LONGEST, THUMB_SHORTEST = 560, 280  # Jev accepts 280-1568 px per side


def render_pdf(data: bytes, run: Path, raw_pixels=False, only_pages=None, thumbs=False):
    """Local parser operation. Public uploads call this only in the networkless sandbox.

    thumbs: small raw-RGB pictures of every page (for page sorting), the page
    count and the native-text diagnostic. raw_pixels with only_pages: full-size
    raw RGB of just those pages (OCR happens in batches of a few pages)."""
    if not (only_pages and raw_pixels):
        (run / "original.pdf").write_bytes(data)
    with pymupdf.open(stream=data, filetype="pdf") as doc, pymupdf.open() as raster:
        if doc.needs_pass:
            raise ValueError("Unlock the PDF before uploading it.")
        if not 1 <= len(doc) <= MAX_PAGES:
            raise ValueError(
                f"Please use a research paper of 1–{MAX_PAGES} pages, not {len(doc)}."
            )
        for page in doc:
            w, h = page.rect.width, page.rect.height
            if not 50 <= w <= 2500 or not 50 <= h <= 2500 or max(w, h) / min(w, h) > 5:
                raise ValueError(
                    "This PDF has unusually sized pages; please export a standard-page PDF."
                )
        if thumbs:
            (run / "thumbs").mkdir(exist_ok=True)
            native, sizes = [], []
            for i, page in enumerate(doc):
                native.append({"page": i + 1, "text": page.get_text(sort=True)})
                w, h = page.rect.width, page.rect.height
                zoom = max(THUMB_LONGEST / max(w, h), THUMB_SHORTEST / min(w, h))
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(zoom, zoom), alpha=False, colorspace=pymupdf.csRGB
                )
                (run / "thumbs" / f"{i + 1}.rgb").write_bytes(pix.samples)
                sizes.append({"page": i + 1, "width": pix.width, "height": pix.height})
            save(run / "thumbs.json", sizes)
            save(run / "native-diagnostic.json", native)
            return len(doc)
        if only_pages is not None:
            if not only_pages or any(
                type(n) is not int or not 1 <= n <= len(doc) for n in only_pages
            ) or len(set(only_pages)) != len(only_pages):
                raise ValueError("Invalid page selection.")
        wanted = only_pages or list(range(1, len(doc) + 1))
        native, pixel_pages = [], []
        (run / "pages").mkdir(exist_ok=True)
        if raw_pixels:
            (run / "rgb").mkdir(exist_ok=True)
        for number in wanted:
            i, page = number - 1, doc[number - 1]
            native.append({"page": i + 1, "text": page.get_text(sort=True)})
            zoom = min(
                400 / 72, math.sqrt(5_000_000 / (page.rect.width * page.rect.height))
            )
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(zoom, zoom), alpha=False, colorspace=pymupdf.csRGB
            )
            if raw_pixels:
                (run / "rgb" / f"{i + 1}.rgb").write_bytes(pix.samples)
                pixel_pages.append(
                    {"page": i + 1, "width": pix.width, "height": pix.height}
                )
            else:
                output = raster.new_page(width=page.rect.width, height=page.rect.height)
                output.insert_image(output.rect, stream=pix.tobytes("png"))
                page.get_pixmap(matrix=pymupdf.Matrix(1.1, 1.1), alpha=False).save(
                    run / "pages" / f"{i + 1}.png"
                )
        if not raw_pixels:
            raster.save(run / "rendered.pdf", deflate=True)
        else:
            save(run / "pixel-pages.json", pixel_pages)
        pages = len(doc)
    if not only_pages:
        save(run / "native-diagnostic.json", native)
    return pages


def extract(pdf: Path, run: Path, service: str, renderer=None):
    run.mkdir(parents=True, exist_ok=True)
    if pdf.stat().st_size > MAX_BYTES:
        raise ValueError("Please upload a PDF under 25 MiB. No pages were skipped.")
    data = pdf.read_bytes()
    if not data.startswith(b"%PDF-"):
        raise ValueError("Input is not a PDF.")
    digest = hashlib.sha256(data).hexdigest()
    signature = {
        "pdf_sha256": digest,
        "version": VERSION,
        "service": service,
        "mode": "all-pages-lighton",
        "max_pixels": 5_000_000,
        "dpi": 400,
    }
    cached = run / "extraction.json"
    if cached.exists():
        result = json.loads(cached.read_text())
        if result["signature"] != signature:
            raise ValueError(
                "Run directory belongs to different input/settings; use a new directory."
            )
        print("Extraction cache hit", flush=True)
        return result
    started = time.monotonic()
    pages = (renderer or render_pdf)(data, run)
    rendered = run / "rendered.pdf"
    if rendered.stat().st_size > MAX_BYTES:
        raise ValueError(
            "Rendered document exceeds the Attachments upload limit; split support is not implemented."
        )
    print(f"Reading {pages} rendered pages with Attachments + LightOnOCR…", flush=True)
    # The hosted service caps OCR at ten pages per request. Five-page windows
    # stay below that cap; every response must explicitly confirm every page.
    texts, segments, chunks = [], [], []
    offset = 0
    for first in range(1, pages + 1, 5):
        last = min(pages, first + 4)
        chunk_key = hashlib.sha256(
            json.dumps([signature, first, last], sort_keys=True).encode()
        ).hexdigest()[:20]
        chunk_file = run / f"attachments-{chunk_key}.json"
        if chunk_file.exists():
            part = json.loads(chunk_file.read_text())
        else:
            with rendered.open("rb") as f:
                response = requests.post(
                    service.rstrip("/") + "/process",
                    files={"file": ("rendered.pdf", f, "application/pdf")},
                    data={
                        "ocr": "true",
                        "ocr_engine": "lighton",
                        "images": "false",
                        "pages": f"{first}-{last}",
                    },
                    timeout=(15, 900),
                )
            response.raise_for_status()
            part = response.json()
        meta = part.get("meta", {})
        extra = meta.get("extra", {})
        if meta.get("error") or not part.get("text", "").strip():
            raise ValueError(f"Attachments extraction failed: {meta}")
        if (
            extra.get("ocr_fallback_pages")
            or "lighton" not in str(extra.get("ocr_backend", "")).lower()
        ):
            raise ValueError(
                f"This run requires LightOnOCR, not a silent fallback: {extra}"
            )
        expected = set(range(first, last + 1))
        if set(extra.get("ocr_pages", [])) != expected:
            raise ValueError(f"Not every requested page was read: {meta}")
        if meta.get("warnings"):
            raise ValueError(
                f"Extraction needs review before proceeding: {meta['warnings']}"
            )
        if {s.get("page") for s in meta.get("segments", [])} != expected:
            raise ValueError("Extraction did not return every page segment.")
        save(chunk_file, part)
        for s in meta["segments"]:
            value = part["text"][s["start"] : s["end"]]
            if not value.strip():
                raise ValueError(
                    f"Page {s['page']} has no readable text; review before proceeding."
                )
            segments.append({**s, "start": offset, "end": offset + len(value)})
            texts.append(value)
            offset += len(value) + 2
        chunks.append({"file": chunk_file.name, "meta": meta})
        print(f"Read pages {first}–{last}", flush=True)
    artifact = {
        "text": "\n\n".join(texts),
        "images": [],
        "meta": {
            "source": "original.pdf",
            "kind": "pdf",
            "segments": segments,
            "warnings": [],
            "extra": {
                "ocr_backend": "lighton",
                "ocr_pages": list(range(1, pages + 1)),
                "chunks": chunks,
            },
        },
    }
    save(run / "attachments.json", artifact)
    result = {
        "signature": signature,
        "pages": pages,
        "seconds": time.monotonic() - started,
        "service_version": requests.get(service.rstrip("/") + "/health", timeout=10)
        .json()
        .get("version"),
        "artifact": artifact,
    }
    save(cached, result)
    (run / "extracted.md").write_text(artifact["text"])
    print(
        f"Extracted {len(artifact['text']):,} characters in {result['seconds']:.1f}s",
        flush=True,
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("--service", default="http://127.0.0.1:8820")
    a = p.parse_args()
    extract(a.pdf, a.run, a.service)
