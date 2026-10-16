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


def extract_images(pdf, run, service):
    from io_utils import save

    if pdf.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("Please use a PDF under 25 MiB.")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    signature = {
        "pdf_sha256": digest,
        "mode": "sandbox-rgb-lighton-v1",
        "service": service,
    }
    cached = run / "extraction.json"
    if cached.exists():
        previous = json.loads(cached.read_text())
        if previous["signature"] != signature:
            raise ValueError(
                "This job used an older extraction format. Please start a fresh upload."
            )
        return previous
    started = time.monotonic()
    rendered = run / "render-sandbox"
    sandbox.run("render-rgb", pdf, rendered, timeout=240)
    info = json.loads((rendered / "render.json").read_text())
    pages = json.loads((rendered / "pixel-pages.json").read_text())
    count = info.get("pages")
    if (
        type(count) is not int
        or not 1 <= count <= 20
        or not isinstance(pages, list)
        or len(pages) != count
        or not all(isinstance(p, dict) and type(p.get("page")) is int for p in pages)
        or [p["page"] for p in pages] != list(range(1, count + 1))
    ):
        raise ValueError("Incomplete page rendering; no pages were silently skipped.")
    if hashlib.sha256((rendered / "original.pdf").read_bytes()).hexdigest() != digest:
        raise ValueError("The original PDF changed during rendering.")
    shutil.copyfile(rendered / "original.pdf", run / "original.pdf")
    shutil.copyfile(rendered / "native-diagnostic.json", run / "native-diagnostic.json")
    (run / "pages").mkdir(exist_ok=True)
    text_parts, segments, models = [], [], []
    offset = 0
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
        if (
            meta.get("error")
            or meta.get("warnings")
            or extra.get("ocr_backend") != "lighton"
            or extra.get("ocr_engine_fallback")
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
        raw.unlink()  # raw RGB is a bounded intermediate, not a permanent duplicate
        print(f"LightOnOCR page {number}/{count}", flush=True)
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
                "ocr_pages": list(range(1, count + 1)),
                "ocr_models": models,
            },
        },
    }
    result = {
        "signature": signature,
        "pages": count,
        "seconds": time.monotonic() - started,
        "artifact": artifact,
    }
    save(cached, result)
    save(run / "attachments.json", artifact)
    (run / "extracted.md").write_text(artifact["text"])
    return result
