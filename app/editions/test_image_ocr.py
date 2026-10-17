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


def fake_run(pages, native_words=0):
    import json

    def run(operation, source, output, timeout=0, pages_wanted=None, **kw):
        from pathlib import Path

        wanted = kw.get("pages") or list(range(1, pages + 1))
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "render.json").write_text(json.dumps({"pages": pages}))
        if operation == "thumbs":
            (out / "thumbs").mkdir()
            (out / "original.pdf").write_bytes(Path(source).read_bytes())
            (out / "native-diagnostic.json").write_text(
                json.dumps([{"page": n, "text": "word " * native_words} for n in range(1, pages + 1)])
            )
            (out / "thumbs.json").write_text(
                json.dumps([{"page": n, "width": 280, "height": 396} for n in range(1, pages + 1)])
            )
            for n in range(1, pages + 1):
                (out / "thumbs" / f"{n}.rgb").write_bytes(bytes(280 * 396 * 3))
            return
        (out / "rgb").mkdir()
        (out / "pixel-pages.json").write_text(
            json.dumps([{"page": n, "width": 2, "height": 2} for n in wanted])
        )
        for n in wanted:
            (out / "rgb" / f"{n}.rgb").write_bytes(bytes(12))

    return run


def fake_sort(kinds):
    """kinds: one Jev page choice per page, all confident."""

    def sort(pictures, cache):
        from page_sort import choose_pages

        answers = {n: {"choice": k, "confidence": 0.95} for n, k in enumerate(kinds, 1)}
        return {**choose_pages(answers, len(kinds)), "pages": len(kinds)}

    return sort


@pytest.fixture(autouse=True)
def all_main_text(monkeypatch):
    import page_sort

    monkeypatch.setattr(
        page_sort, "sort_pages", lambda pictures, cache: fake_sort(["main_text"] * len(pictures))(pictures, cache)
    )


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


def test_only_main_text_pages_are_read(tmp_path, monkeypatch):
    import image_ocr, page_sort

    (tmp_path / "source.pdf").write_bytes(b"%PDF-1.7 fixture")
    kinds = ["main_text"] * 5 + ["references"] * 2 + ["appendix"] * 3
    monkeypatch.setattr(image_ocr.sandbox, "run", fake_run(10))
    monkeypatch.setattr(page_sort, "sort_pages", fake_sort(kinds))
    seen = []
    monkeypatch.setattr(
        image_ocr, "process_image", lambda service, image, n: seen.append(n) or artifact()
    )
    result = image_ocr.extract_images(tmp_path / "source.pdf", tmp_path, "unix:/x")
    # The first reference page is still read: the conclusion often ends on it.
    assert seen == [1, 2, 3, 4, 5, 6]
    assert result["pages"] == 10 and result["read_pages"] == [1, 2, 3, 4, 5, 6]
    assert [s["page"] for s in result["artifact"]["meta"]["segments"]] == seen
    assert result["artifact"]["meta"]["extra"]["skipped_pages"] == [7, 8, 9, 10]


def test_too_long_main_text_stops_before_reading(tmp_path, monkeypatch):
    import image_ocr

    (tmp_path / "source.pdf").write_bytes(b"%PDF-1.7 fixture")
    monkeypatch.setattr(image_ocr.sandbox, "run", fake_run(30, native_words=1500))
    monkeypatch.setattr(
        image_ocr, "process_image", lambda *a: pytest.fail("must not read pages")
    )
    with pytest.raises(ValueError, match="25,000 words"):
        image_ocr.extract_images(tmp_path / "source.pdf", tmp_path, "unix:/x")


def test_sorting_outage_reads_short_papers_and_stops_long_ones(tmp_path, monkeypatch):
    import image_ocr, page_sort

    def down(pictures, cache):
        raise RuntimeError("service down")

    monkeypatch.setattr(page_sort, "sort_pages", down)
    monkeypatch.setattr(image_ocr, "process_image", lambda *a: artifact())
    short = tmp_path / "short"
    short.mkdir()
    (short / "source.pdf").write_bytes(b"%PDF-1.7 a")
    monkeypatch.setattr(image_ocr.sandbox, "run", fake_run(3))
    assert image_ocr.extract_images(short / "source.pdf", short, "unix:/x")["read_pages"] == [1, 2, 3]
    long = tmp_path / "long"
    long.mkdir()
    (long / "source.pdf").write_bytes(b"%PDF-1.7 b")
    monkeypatch.setattr(image_ocr.sandbox, "run", fake_run(30))
    with pytest.raises(ValueError, match="retry later"):
        image_ocr.extract_images(long / "source.pdf", long, "unix:/x")
