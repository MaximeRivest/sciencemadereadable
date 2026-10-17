from caption_labels import label_of, key, source_captions, rewrite_paragraphs, choose_anchors
import json


def test_labels_with_and_without_a_full_stop():
    assert label_of("Fig. 3 Distance between the centroids") == ("Figure 3", False)
    assert label_of("**Fig. 12** Examples of trajectories") == ("Figure 12", False)
    assert label_of("Table 2. Health status") == ("Table 2", True)
    assert label_of("Figure 10: Architecture") == ("Figure 10", True)
    assert label_of("Fig. 3a shows") == (None, False)
    assert label_of("Figures 3 and 4 show") == (None, False)


def block(text, role, confidence=0.99, page=1, review=None):
    return {"text": text, "role": role, "page": page, "role_answer": {"confidence": confidence}, "manual_review": review}


def test_only_jev_captions_count_in_the_original():
    ledger = [
        block("Fig. 2 shows how the model traced crowns.", "paragraph"),
        block("Fig. 2 Example of validation images", "caption", page=6),
        block("Fig. 4 Map of crowns", "caption", confidence=0.4),
        block("Satellite map of California", "caption"),
        block("Fig. 5 Proportion of pixels", "paragraph", review={"role": "caption"}),
    ]
    found = source_captions(ledger)
    assert set(found) == {"Figure 2", "Figure 5"} and found["Figure 2"]["page"] == 6


def test_jev_decides_between_a_caption_and_a_sentence(tmp_path):
    ledger = [block("Fig. 2 Example of validation images for giant sequoia trees", "caption"),
              block("Table 1. Ground observations", "caption")]
    parts = {"results": "Fig. 2 shows some examples. The model was close.\n\nFig. 2 Examples of giant sequoia crowns traced by hand.\n\nTable 1. Ground checks."}
    (tmp_path / "labelled.json").write_text(json.dumps({"ledger": ledger}))
    (tmp_path / "rewrite.json").write_text(json.dumps({"parts": parts}))
    asked = {}

    def fake_jev(body, cache):
        asked.update(body["questions"])
        state = body["state"]["captions"]
        answers = {}
        for q in body["questions"]:
            c = 0.1 if "shows" in state[q]["paragraph"] else 0.95
            answers[q] = {"choice": "caption" if c > 0.5 else "mention", "probabilities": {"caption": c, "mention": 1 - c}}
        return {"response": {"answers": answers, "usage": {}}}, False

    result = choose_anchors(tmp_path, call_jev=fake_jev, model="m")
    assert len(asked) == 2  # only Figure 2 needed Jev; Table 1 was unambiguous
    assert result["anchors"]["Figure 2"]["key"] == key("Fig. 2 Examples of giant sequoia crowns traced by hand.")
    assert result["anchors"]["Table 1"]["how"] == "unique"


def test_key_matches_markdown_and_rendered_html():
    assert key("**Fig. 8** A mosaic of $2020$ photos") == key("Fig. 8 A mosaic of 2020 photos")
