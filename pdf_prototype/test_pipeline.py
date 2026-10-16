"""Offline regression tests; no services or paid calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from label import assemble, blocks_from, questions_for, call_jev, label


def answer(choice, confidence=1):
    return {"choice": choice, "confidence": confidence, "probabilities": {choice: 1}}


def fixture():
    rows = [("Study title", "title", "title", 1),
            ("## Introduction", "heading", "introduction", 1),
            ("First paragraph begins and", "paragraph", "introduction", 1),
            ("continues here.", "paragraph", "introduction", 2),
            ("Second paragraph.", "paragraph", "introduction", 2),
            ("## Methods", "heading", "methods", 2),
            ("Method prose.", "paragraph", "methods", 2),
            ("## Results and Discussion", "heading", "results_discussion", 2),
            ("Finding and interpretation.", "paragraph", "results_discussion", 2),
            ("1. Bibliography entry", "references", "other", 2)]
    blocks, answers = [], {}
    for i, (text, role, section, page) in enumerate(rows):
        bid = f"b{i:04d}"
        blocks.append({"id": bid, "page": page, "text": text, "start": i * 100, "end": i * 100 + len(text)})
        answers[bid + "_role"] = answer(role)
        answers[bid + "_section"] = answer(section)
    answers["b0003_boundary"] = answer("join")
    return blocks, answers


class PipelineTests(unittest.TestCase):
    def test_split_preserves_offsets_and_html(self):
        text = "  ## Title\n\nParagraph one.\n\n<table>\n<tr><td>42</td></tr>\n</table>  "
        blocks = blocks_from({"text": text, "meta": {"segments": [{"page": 1, "start": 0, "end": len(text)}]}})
        self.assertEqual(len(blocks), 3)
        for b in blocks:
            self.assertEqual(text[b["start"]:b["end"]], b["text"])
        self.assertTrue(blocks[-1]["text"].endswith("</table>"))

    def test_assembled_structure_and_page_join(self):
        blocks, answers = fixture()
        result = assemble(blocks, answers)
        self.assertTrue(result["ready"])
        self.assertEqual(result["sections"]["introduction_first"], "First paragraph begins and\ncontinues here.")
        self.assertEqual(result["sections"]["introduction_rest"], "Second paragraph.")
        self.assertIn("Results and Discussion", result["sections"]["results"])
        self.assertEqual(result["sections"]["discussion"], "")
        self.assertEqual(result["sections"]["conclusion"], "")
        self.assertEqual(len(result["ledger"]), len(blocks))
        used = [i for group in result["section_block_ids"].values() for i in group]
        self.assertEqual(len(used), len(set(used)))
        self.assertIsNone(result["ledger"][-1]["destination"])

    def test_uncertainty_stops_rewrite(self):
        blocks, answers = fixture()
        answers["b0006_section"] = answer("methods", .3)
        self.assertFalse(assemble(blocks, answers)["ready"])

    def test_missing_required_stops_rewrite(self):
        blocks, answers = fixture()
        for bid in ("b0005", "b0006"):
            answers[bid + "_section"] = answer("other")
        result = assemble(blocks, answers)
        self.assertIn("methods", result["missing_required"])
        self.assertFalse(result["ready"])

    def test_speculative_prose_join_ignored_for_tables(self):
        blocks, answers = fixture()
        answers["b0003_role"] = answer("table")
        answers["b0003_boundary"] = answer("join", .1)
        result = assemble(blocks, answers)
        self.assertEqual(result["ledger"][3]["boundary"], "separate")
        self.assertFalse(result["ledger"][3]["review_reasons"])

    def test_cross_section_join_stops(self):
        blocks, answers = fixture()
        answers["b0003_section"] = answer("methods")
        self.assertIn("page-join conflicts with section or block role", assemble(blocks, answers)["ledger"][3]["review_reasons"])

    def test_overrides_require_reason_and_valid_block(self):
        blocks, answers = fixture()
        with self.assertRaises(ValueError):
            assemble(blocks, answers, overrides={"missing": {}})
        with self.assertRaises(ValueError):
            assemble(blocks, answers, overrides={"b0001": {"role": "heading", "section": "introduction"}})
        with self.assertRaises(ValueError):
            assemble(blocks, answers, overrides={"b0001": {"role": "heading", "section": "introduction", "reason": "test", "boundary": "whatever"}})

    def test_review_changes_labels_not_source(self):
        blocks, answers = fixture()
        answers["b0006_section"] = answer("methods", .2)
        result = assemble(blocks, answers, overrides={"b0006": {"role": "paragraph", "section": "methods", "reason": "Checked PDF"}})
        self.assertTrue(result["ready"])
        self.assertEqual(result["ledger"][6]["text"], blocks[6]["text"])
        self.assertEqual(result["ledger"][6]["section_answer"]["confidence"], .2)

    def test_questions_identify_target_in_instructions(self):
        blocks, _ = fixture()
        request = questions_for(blocks, blocks[3:4], "test-model")
        self.assertEqual(len(request["questions"]), 3)
        for q in request["questions"].values():
            self.assertIn("b0003", q["instructions"])
            self.assertIn("uncertain", q["criteria"])

    def test_review_manifest_cannot_target_different_pdf_or_text(self):
        artifact = {"text": "Title", "meta": {"segments": [{"page": 1, "start": 0, "end": 5}]}}
        with tempfile.TemporaryDirectory() as temp, patch("label.call_jev", side_effect=AssertionError("network")):
            path = Path(temp)
            (path / "extraction.json").write_text(json.dumps({"signature": {"pdf_sha256": "pdf-hash"}, "artifact": artifact}))
            with self.assertRaisesRegex(ValueError, "exact PDF"):
                label(path, overrides={"pdf_sha256": "different", "blocks": {}})
            with self.assertRaisesRegex(ValueError, "exact extracted block"):
                label(path, overrides={"pdf_sha256": "pdf-hash", "blocks": {"b0000": {"text_sha256": "wrong"}}})

    def test_cached_call_needs_no_key_or_network(self):
        import hashlib
        body = {"state": "test", "model": "test", "questions": {}}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as temp, patch("label.requests.post", side_effect=AssertionError("network")):
            path = Path(temp)
            (path / f"{key}.json").write_text(json.dumps({"request": body, "response": {"answers": {}}, "seconds": 0}))
            _, cached = call_jev(body, path)
            self.assertTrue(cached)


if __name__ == "__main__":
    unittest.main()
