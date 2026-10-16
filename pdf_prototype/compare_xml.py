"""Post-hoc diagnostic against publisher XML; never feeds text to the pipeline.

Run with ../.venv/bin/python. Five-word overlap locates reference prose paragraphs
in PDF-derived sections. This is not a factual-fidelity score or a table audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper_corpus"))
from harvest import split_sections, KINDS, license_ok


def tokens(text):
    text = html.unescape(re.sub(r"</?[A-Za-z][^>]*>", " ", text))
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\\(?:mathrm|text|operatorname)\s*", "", text)
    return re.findall(r"[^\W_]+", text)


def shingles(text):
    words = tokens(text)
    return {tuple(words[i:i + 5]) for i in range(len(words) - 4)}


def compare(run, xml):
    root = ET.parse(xml).getroot()
    reference = split_sections(root)
    if not reference or not license_ok(root):
        raise ValueError("Reference must be a standard CC BY research article.")
    source = json.loads((run / "sections.json").read_text())
    doi = root.findtext("./front/article-meta/article-id[@pub-id-type='doi']") or ""
    extracted = json.loads((run / "extraction.json").read_text())["artifact"]["text"]
    if not doi or re.sub(r"\s", "", doi.casefold()) not in re.sub(r"\s", "", extracted.casefold()):
        raise ValueError("Reference XML DOI was not found in the extracted PDF; do not attach unverified source metadata.")
    # Collapse the two introduction pieces for membership checking, not for rewrite input.
    predicted = {k: v for k, v in source.items() if not k.startswith("introduction_")}
    predicted["introduction"] = source["introduction_first"] + "\n" + source["introduction_rest"]
    sets = {k: shingles(v) for k, v in predicted.items()}
    paragraphs = []
    abstract = root.find("./front/article-meta/abstract")
    groups = [("abstract", abstract)] if abstract is not None else []
    for section in root.findall("./body/sec"):
        heading = (section.findtext("title") or "").lower()
        kind = next((k for k, names in KINDS if any(n in heading for n in names)), None)
        if kind:
            groups.append((kind, section))
    for expected, section in groups:
        # ElementTree's .//p includes captions; explicitly traverse prose while
        # excluding figure/table/reference wrappers for this prose-only check.
        def walk(node):
            if node.tag in {"table-wrap", "fig", "ref-list", "supplementary-material"}:
                return
            if node.tag == "p":
                yield node
                return
            for child in node:
                yield from walk(child)
        for p in walk(section):
            text = "".join(p.itertext())
            grams = shingles(text)
            if len(tokens(text)) < 20:
                continue
            scores = {k: len(grams & v) / len(grams) for k, v in sets.items()}
            best = max(scores, key=scores.get)
            paragraphs.append({"expected_section": expected, "best_section": best,
                "five_word_coverage_in_expected": scores.get(expected, 0),
                "located_in_expected": best == expected and scores[expected] >= .7,
                "reference_text": text})
    section_stats = []
    for k, v in reference.items():
        if k == "original_section_order" or not v:
            continue
        grams = shingles(v)
        section_stats.append({"section": k, "reference_words": len(tokens(v)), "pdf_words": len(tokens(source[k])),
                              "five_word_overlap": len(grams & shingles(source[k])) / max(1, len(grams))})
    meta = root.find("./front/article-meta")
    result = {"doi": root.findtext("./front/article-meta/article-id[@pub-id-type='doi']"),
        "authors": [" ".join(filter(None, [n.findtext("given-names"), n.findtext("surname")])) for n in meta.findall("./contrib-group/contrib[@contrib-type='author']/name")],
        "year": meta.findtext("./pub-date/year"),
        "xml_file": str(xml), "license": "CC BY", "paragraphs": paragraphs, "sections": section_stats,
        "paragraph_count": len(paragraphs), "located_in_expected": sum(p["located_in_expected"] for p in paragraphs),
        "method": "Prose paragraphs of at least 20 words: best five-word overlap must be the expected section, with at least 70% overlap. Figures and tables excluded. Diagnostic only; not an accuracy benchmark.",
        "independence": "Publisher XML was read only after PDF extraction and Jev labelling; no XML text supplied to either model."}
    (run / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Located {result['located_in_expected']}/{len(paragraphs)} reference prose paragraphs in expected sections")
    for p in paragraphs:
        if not p["located_in_expected"]:
            print(p["expected_section"], p["best_section"], round(p["five_word_coverage_in_expected"], 3), p["reference_text"][:140])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("xml", type=Path)
    a = p.parse_args()
    compare(a.run, a.xml)
