"""Public-upload OCR boundary: only bounded raw RGB leaves the PDF sandbox.
The controller constructs a minimal PNG itself; no parser-produced PDF or image
container is forwarded to another service. Attachments still supplies LightOnOCR.
"""

import binascii
import hashlib
import http.client
import socket
import json
from pathlib import Path
import shutil
import struct
import time
import zlib

import requests
import sandbox


def png_from_rgb(width, height, pixels):
    if (
        type(width) is not int
        or type(height) is not int
        or not 1 <= width <= 20000
        or not 1 <= height <= 20000
        or width * height > 5_100_000
    ):
        raise ValueError("Invalid rendered page dimensions.")
    if len(pixels) != width * height * 3:
        raise ValueError(
            "Rendered page did not contain exactly the declared RGB pixels."
        )

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = b"".join(
        b"\0" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def process_image(service, image, number):
    request = requests.Request(
        "POST",
        "http://localhost/process",
        files={"file": (f"page-{number}.png", image, "image/png")},
        data={"ocr": "true", "ocr_engine": "lighton", "max_dim": "256"},
    ).prepare()
    if service.startswith("unix:"):

        class UnixConnection(http.client.HTTPConnection):
            def connect(self):
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(180)
                self.sock.connect(service.removeprefix("unix:"))

        conn = UnixConnection("localhost", timeout=180)
        try:
            conn.request(
                "POST", "/process", body=request.body, headers=dict(request.headers)
            )
            response = conn.getresponse()
            raw = response.read(2_000_001)
            if response.status != 200 or len(raw) > 2_000_000:
                raise RuntimeError(
                    "The OCR service did not return a bounded successful response."
                )
            return json.loads(raw)
        finally:
            conn.close()
    response = requests.post(
        service.rstrip("/") + "/process",
        data=request.body,
        headers={"Content-Type": request.headers["Content-Type"]},
        timeout=(15, 180),
        allow_redirects=False,
    )
    response.raise_for_status()
    if len(response.content) > 2_000_000:
        raise ValueError("The OCR service response exceeded the page limit.")
    return response.json()


MAX_READ_PAGES = 100  # pages actually read; the whole PDF may have up to 200
RENDER_BATCH = 8  # full-size pages per sandbox call (each up to ~15 MB of raw RGB)
# The main text is checked again exactly after labelling. This early estimate
# from the PDF's own text layer only stops clearly oversized documents before
# minutes of reading; 1.25x the rewrite limit leaves room for headers, the
# reference list on the last main-text page, and equations.
EARLY_TEXT_LIMIT = 200_000


def _thumbnails(pdf, run):
    staging = run / "thumbs-sandbox"
    shutil.rmtree(staging, ignore_errors=True)
    sandbox.run("thumbs", pdf, staging, timeout=240)
    count = json.loads((staging / "render.json").read_text()).get("pages")
    sizes = json.loads((staging / "thumbs.json").read_text())
    if (
        type(count) is not int
        or not 1 <= count <= 200
        or not isinstance(sizes, list)
        or [p.get("page") for p in sizes] != list(range(1, count + 1))
    ):
        raise ValueError("Incomplete page rendering; no pages were silently skipped.")
    pictures = []
    for p in sizes:
        w, h = p.get("width"), p.get("height")
        if type(w) is not int or type(h) is not int or not 280 <= min(w, h) or max(w, h) > 1568:
            raise ValueError("Invalid page picture size.")
        raw = staging / "thumbs" / f"{p['page']}.rgb"
        if raw.stat().st_size != w * h * 3:
            raise ValueError("Rendered page did not contain exactly the declared pixels.")
        pictures.append(png_from_rgb(w, h, raw.read_bytes()))
    native = json.loads((staging / "native-diagnostic.json").read_text())
    return count, pictures, native, staging


def plan_pages(pdf, run, digest):
    """Sort pages with Jev from small pictures, then decide which pages to read."""
    from page_sort import sort_pages
    from io_utils import save

    cached = run / "page-sort.json"
    if cached.exists():
        plan = json.loads(cached.read_text())
        if plan.get("pdf_sha256") == digest:
            return plan
    count, pictures, native, staging = _thumbnails(pdf, run)
    original = staging / "original.pdf"
    if hashlib.sha256(original.read_bytes()).hexdigest() != digest:
        raise ValueError("The original PDF changed during rendering.")
    shutil.copyfile(original, run / "original.pdf")
    shutil.copyfile(staging / "native-diagnostic.json", run / "native-diagnostic.json")
    try:
        plan = sort_pages(pictures, run / "jev-cache")
    except (requests.RequestException, RuntimeError, ValueError, KeyError) as e:
        print("Page sorting failed:", type(e).__name__, str(e)[:300], flush=True)
        if count > 20:
            raise ValueError(
                "We could not sort this PDF's pages right now, so we cannot tell which pages are the main text. Please retry later."
            )
        # Short papers were always read in full before page sorting existed.
        plan = {"read": list(range(1, count + 1)), "skipped": [], "main_text_ends": None,
                "pages": count, "fallback": "page sorting unavailable; every page read"}
    plan["pdf_sha256"] = digest
    texts = {
        x.get("page"): x.get("text", "")
        for x in native
        if isinstance(x, dict) and isinstance(x.get("text"), str)
    }
    plan["native_chars_read_pages"] = sum(len(" ".join(texts.get(p, "").split())) for p in plan["read"])
    save(cached, plan)
    shutil.rmtree(staging, ignore_errors=True)
    return plan


def check_plan(plan):
    read = plan["read"]
    if len(read) > MAX_READ_PAGES:
        raise ValueError(
            f"The main part of this PDF looks longer than {MAX_READ_PAGES} pages ({len(read)} pages before its references or appendix). Please upload just the article."
        )
    if plan["native_chars_read_pages"] > EARLY_TEXT_LIMIT:
        words = round(plan["native_chars_read_pages"] / 6.5, -3)
        raise ValueError(
            f"The main part of this paper (before its references and appendices) has roughly {words:,.0f} words. The current rewriting limit is about 25,000 words of main text."
        )


def extract_images(pdf, run, service):
    from io_utils import save

    if pdf.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("Please use a PDF under 25 MiB.")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    signature = {
        "pdf_sha256": digest,
        "mode": "sandbox-rgb-lighton-v2-sorted",
        "service": service,
    }
    cached = run / "extraction.json"
    if cached.exists():
        previous = json.loads(cached.read_text())
        old = {**signature, "mode": "sandbox-rgb-lighton-v1"}
        if previous["signature"] not in (signature, old):
            raise ValueError(
                "This job used an older extraction format. Please start a fresh upload."
            )
        return previous
    started = time.monotonic()
    plan = plan_pages(pdf, run, digest)
    check_plan(plan)
    count, read = plan["pages"], plan["read"]
    (run / "pages").mkdir(exist_ok=True)
    text_parts, segments, models, fallback_pages = [], [], [], []
    offset = 0
    for first in range(0, len(read), RENDER_BATCH):
        batch = read[first : first + RENDER_BATCH]
        rendered = run / "render-sandbox"
        shutil.rmtree(rendered, ignore_errors=True)
        sandbox.run("render-rgb", pdf, rendered, timeout=240, pages=batch)
        info = json.loads((rendered / "render.json").read_text())
        pages = json.loads((rendered / "pixel-pages.json").read_text())
        if (
            info.get("pages") != count
            or not isinstance(pages, list)
            or not all(isinstance(p, dict) and type(p.get("page")) is int for p in pages)
            or [p["page"] for p in pages] != batch
        ):
            raise ValueError("Incomplete page rendering; no pages were silently skipped.")
        for p in pages:
            number = p["page"]
            raw = rendered / "rgb" / f"{number}.rgb"
            if raw.stat().st_size > 15_300_000:
                raise ValueError("Rendered page exceeds the pixel budget.")
            image = png_from_rgb(p["width"], p["height"], raw.read_bytes())
            image_path = run / "pages" / f"{number}.png"
            tmp = image_path.with_suffix(".tmp")
            tmp.write_bytes(image)
            tmp.replace(image_path)
            key = hashlib.sha256(image).hexdigest()
            record = run / f"ocr-page-{number}-{key[:16]}.json"
            if record.exists():
                artifact = json.loads(record.read_text())
            else:
                artifact = process_image(service, image, number)
                artifact["images"] = []  # never keep the service's encoded preview
            meta = artifact.get("meta", {})
            extra = meta.get("extra", {})
            warnings = meta.get("warnings") or []
            # LightOnOCR occasionally gets stuck repeating itself on a dense page;
            # Attachments then reads that page with RapidOCR. Accept only that exact,
            # disclosed fallback: the page is then forced into the reader's review.
            fell_back = (
                extra.get("ocr_backend") == "rapidocr"
                and extra.get("ocr_engine_fallback") == "rapidocr"
                and len(warnings) == 1
                and "RapidOCR" in str(warnings[0])
            )
            if meta.get("error") or (
                not fell_back
                and (
                    warnings
                    or extra.get("ocr_backend") != "lighton"
                    or extra.get("ocr_engine_fallback")
                )
            ):
                raise ValueError(
                    f"LightOnOCR could not reliably read page {number}. Please retry later or use a clearer PDF."
                )
            text = artifact.get("text", "")
            if not isinstance(text, str) or not text.strip() or len(text) > 50000:
                raise ValueError(
                    f"Page {number} has no usable text or too much text. Check the original PDF."
                )
            save(record, artifact)
            segments.append(
                {
                    "kind": "page",
                    "page": number,
                    "label": f"page {number}",
                    "start": offset,
                    "end": offset + len(text),
                }
            )
            offset += len(text) + 2
            text_parts.append(text)
            models.append(extra.get("ocr_model"))
            if fell_back:
                fallback_pages.append(number)
            raw.unlink()  # raw RGB is a bounded intermediate, not a permanent duplicate
            print(
                f"{'Fallback OCR' if fell_back else 'LightOnOCR'} page {number} ({len(segments)}/{len(read)} read; {count} in the PDF)",
                flush=True,
            )
        shutil.rmtree(rendered, ignore_errors=True)
    artifact = {
        "text": "\n\n".join(text_parts),
        "images": [],
        "meta": {
            "source": "original.pdf",
            "kind": "pdf",
            "segments": segments,
            "warnings": [],
            "extra": {
                "ocr_backend": "lighton",
                "ocr_pages": read,
                "skipped_pages": plan["skipped"],
                "ocr_models": models,
                "fallback_pages": fallback_pages,
            },
        },
    }
    result = {
        "signature": signature,
        "pages": count,
        "read_pages": read,
        "page_sort": {k: plan.get(k) for k in ("main_text_ends", "skipped", "model", "input_tokens", "estimated_usd", "fallback")},
        "seconds": time.monotonic() - started,
        "artifact": artifact,
    }
    save(cached, result)
    save(run / "attachments.json", artifact)
    (run / "extracted.md").write_text(artifact["text"])
    return result
