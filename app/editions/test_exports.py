import base64
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reading_engine"))
import editions
from pipeline import verify_exports, finish


@pytest.fixture
def exported(tmp_path):
    run = tmp_path / "job"
    out = run / "out"
    out.mkdir(parents=True)
    pdf = b"%PDF-1.7\nTest fixture, not parsed by these controller tests."
    (run / "original.pdf").write_bytes(pdf)
    for name in ("original.pdf", "paper.pdf"):
        (out / name).write_bytes(pdf)
    (out / "pages").mkdir()
    (out / "pages/1.png").write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jgR8AAAAASUVORK5CYII="
        )
    )
    e = {
        "id": "a" * 64,
        "title": "Sample",
        "original_title": "Original",
        "doi": "10.1234/sample",
        "year": "2024",
        "license": "CC BY",
        "authors": ["Author"],
        "notice": "Check the original.",
        "source_page_count": 1,
        "visuals": [],
        "chapters": [
            {
                "id": "abstract",
                "title": "Abstract",
                "html": "<p>A safe paragraph.</p>",
                "speech": "A safe paragraph.",
                "source_pages": [1],
            }
        ],
    }
    (out / "edition.json").write_text(json.dumps(e))
    editions.make_epub(e, out)
    (out / "export-complete.json").write_text(json.dumps({"id": e["id"]}))
    return run, out, e


def test_valid_exports_and_original_identity(exported):
    run, out, e = exported
    assert (
        verify_exports(
            out, hashlib.sha256((run / "original.pdf").read_bytes()).hexdigest()
        )["id"]
        == e["id"]
    )
    with pytest.raises(ValueError, match="not the supplied PDF"):
        verify_exports(out, "wrong")


@pytest.mark.parametrize(
    "bad",
    [
        "<script>alert(1)</script>",
        '<p onclick="bad()">text</p>',
        '<img src="https://tracking.invalid/x.png"/>',
    ],
)
def test_unsafe_sandbox_markup_is_refused(exported, bad):
    run, out, e = exported
    e["chapters"][0]["html"] = bad
    (out / "edition.json").write_text(json.dumps(e))
    with pytest.raises(ValueError):
        verify_exports(out)


def test_chapter_path_injection_is_refused(exported):
    run, out, e = exported
    e["chapters"][0]["id"] = "../../other-job"
    (out / "edition.json").write_text(json.dumps(e))
    with pytest.raises(ValueError):
        verify_exports(out)


def test_epub_zip_slip_is_refused(exported):
    run, out, e = exported
    with zipfile.ZipFile(out / "paper.epub", "a") as z:
        z.writestr("../outside.xhtml", "<html/>")
    with pytest.raises(ValueError):
        verify_exports(out)


def test_audio_failure_does_not_discard_reading_formats(exported, monkeypatch):
    run, out, e = exported

    def failed(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(editions, "make_audio", failed)
    finish(run, {"audio": True})
    progress = json.loads((run / "progress.json").read_text())
    assert progress["state"] == "done"
    assert "audio" in progress["error"]
    assert (
        (out / "index.html").is_file()
        and (out / "paper.epub").is_file()
        and (out / "paper.pdf").is_file()
    )
    assert (out / "audio-warning.json").is_file()
