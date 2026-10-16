import struct
import zlib
import pytest
from image_ocr import png_from_rgb


def test_png_has_only_controller_generated_structure():
    pixels = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
    data = png_from_rgb(2, 2, pixels)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    chunks = []
    at = 8
    while at < len(data):
        size = struct.unpack(">I", data[at : at + 4])[0]
        kind = data[at + 4 : at + 8]
        payload = data[at + 8 : at + 8 + size]
        chunks.append((kind, payload))
        at += size + 12
    assert [k for k, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    assert zlib.decompress(chunks[1][1]) == b"\0" + pixels[:6] + b"\0" + pixels[6:]


@pytest.mark.parametrize(
    "w,h,data",
    [(True, 1, b"abc"), (1, 1, b"abcd"), (1, 1, b"ab"), (3000, 3000, b""), (0, 1, b"")],
)
def test_invalid_raw_pixel_protocol_is_rejected(w, h, data):
    with pytest.raises(ValueError):
        png_from_rgb(w, h, data)


def fake_run(pages):
    import json

    def run(operation, source, output, timeout=0):
        from pathlib import Path

        out = Path(output)
        (out / "rgb").mkdir(parents=True)
        (out / "original.pdf").write_bytes(Path(source).read_bytes())
        (out / "native-diagnostic.json").write_text("[]")
        (out / "render.json").write_text(json.dumps({"pages": pages}))
        (out / "pixel-pages.json").write_text(
            json.dumps(
                [{"page": n, "width": 2, "height": 2} for n in range(1, pages + 1)]
            )
        )
        for n in range(1, pages + 1):
            (out / "rgb" / f"{n}.rgb").write_bytes(bytes(12))

    return run


def artifact(backend="lighton", warnings=(), fallback=None):
    extra = {"ocr_backend": backend, "ocr_model": "m"}
    if fallback:
        extra["ocr_engine_fallback"] = fallback
    return {
        "text": "Readable page text.",
        "meta": {"extra": extra, "warnings": list(warnings)},
    }


STUCK = "ocr: LightOnOCR did not read the image: it did not finish the page (it repeated itself). RapidOCR, a simpler engine, read it instead"


def test_disclosed_backup_ocr_page_is_kept_and_recorded(tmp_path, monkeypatch):
    import image_ocr

    (tmp_path / "source.pdf").write_bytes(b"%PDF-1.7 fixture")
    monkeypatch.setattr(image_ocr.sandbox, "run", fake_run(2))
    answers = {1: artifact(), 2: artifact("rapidocr", [STUCK], "rapidocr")}
    monkeypatch.setattr(
        image_ocr, "process_image", lambda service, image, n: answers[n]
    )
    result = image_ocr.extract_images(tmp_path / "source.pdf", tmp_path, "unix:/x")
    assert result["artifact"]["meta"]["extra"]["fallback_pages"] == [2]


@pytest.mark.parametrize(
    "bad",
    [
        artifact("rapidocr", ["something else"], "rapidocr"),
        artifact("rapidocr", [STUCK, "extra"], "rapidocr"),
        artifact("lighton", ["unexpected warning"]),
        {"text": "x", "meta": {"error": "failed", "extra": {}}},
    ],
)
def test_other_ocr_problems_still_stop(tmp_path, monkeypatch, bad):
    import image_ocr

    (tmp_path / "source.pdf").write_bytes(b"%PDF-1.7 fixture")
    monkeypatch.setattr(image_ocr.sandbox, "run", fake_run(1))
    monkeypatch.setattr(image_ocr, "process_image", lambda service, image, n: bad)
    with pytest.raises(ValueError):
        image_ocr.extract_images(tmp_path / "source.pdf", tmp_path, "unix:/x")
