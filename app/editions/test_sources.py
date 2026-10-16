from pathlib import Path
import sys
import pytest
from defusedxml import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reading_engine"))
from jats_source import split_sections, license_ok
import pipeline


def xml(result="Original result"):
    return f"""<article><front><article-meta><article-id pub-id-type="doi">10.1234/source</article-id><title-group><article-title>Source title</article-title></title-group><permissions><license>Creative Commons Attribution</license></permissions><abstract><p>Summary</p></abstract></article-meta></front><body><sec><title>Introduction</title><p>Opening</p></sec><sec><title>Methods</title><p>Procedure</p></sec><sec><title>Results</title><p>{result}</p></sec></body></article>""".encode()


def test_source_revision_must_match_every_section(monkeypatch, tmp_path):
    prepared = {
        "doi": "10.1234/source",
        "sections": split_sections(ET.fromstring(xml())),
    }
    listing = b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><CommonPrefixes><Prefix>PMC1.1/</Prefix></CommonPrefixes><CommonPrefixes><Prefix>PMC1.2/</Prefix></CommonPrefixes></ListBucketResult>'

    def bounded(url, maximum):
        if "?list-type" in url:
            return listing
        return xml("Changed result") if ".2.xml" in url else xml()

    monkeypatch.setattr(pipeline, "bounded_get", bounded)
    calls = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield b"%PDF-1.7\npublic fixture"

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(pipeline.requests, "get", get)
    record = pipeline.download_pdf("PMC1", tmp_path / "source.pdf", prepared)
    assert record["version"] == 1 and record["section_match"] == "exact"
    assert len(calls) == 1 and calls[0].endswith("PMC1.1.pdf")


def test_mismatched_source_never_downloads_a_pdf(monkeypatch, tmp_path):
    prepared = {
        "doi": "10.1234/source",
        "sections": split_sections(ET.fromstring(xml())),
    }
    monkeypatch.setattr(
        pipeline,
        "bounded_get",
        lambda url, maximum: (
            b"<ListBucketResult/>" if "?list-type" in url else xml("Different findings")
        ),
    )

    def unexpected(*a, **k):
        raise AssertionError("PDF download must not happen")

    monkeypatch.setattr(pipeline.requests, "get", unexpected)
    with pytest.raises(ValueError, match="matches this saved rewrite exactly"):
        pipeline.download_pdf("PMC1", tmp_path / "source.pdf", prepared)


def test_canonical_licence_and_combined_sections():
    root = ET.fromstring(xml())
    assert license_ok(root)
    root.find(
        "./front/article-meta/permissions/license"
    ).text = "Creative Commons Attribution licenses/by-nc/"
    assert not license_ok(root)
    root.find("./body/sec[3]/title").text = "Results and Discussion"
    assert split_sections(root)["discussion"] == ""
    assert "Original result" in split_sections(root)["results"]
