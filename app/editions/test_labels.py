from label import assemble, rewrite_size_problem


def answer(choice, confidence=0.95):
    return {"choice": choice, "confidence": confidence}


def run(rows):
    """rows: (page, text, role, section) -> assembled result."""
    blocks, answers = [], {}
    for i, (page, text, role, section) in enumerate(rows):
        bid = f"b{i:04d}"
        blocks.append({"id": bid, "page": page, "text": text, "start": 0, "end": 0})
        answers[bid + "_role"] = answer(role)
        answers[bid + "_section"] = answer(section)
    return assemble(blocks, answers)


BASE = [
    (1, "# A study", "title", "title"),
    (1, "We study things.", "paragraph", "introduction"),
    (2, "## 2 Methods", "heading", "methods"),
    (2, "We measured.", "paragraph", "methods"),
    (3, "## 3 Results", "heading", "results"),
    (3, "It worked.", "paragraph", "results"),
]


def test_lettered_appendix_after_references_is_never_rewritten():
    result = run(
        BASE
        + [
            (4, "## References", "heading", "other"),
            (4, "[1] A. Author. A paper. 2020.", "references", "other"),
            (5, "## G DERIVING THE OPTIMAL TIME SAMPLING", "heading", "methods"),
            (5, "We derive the sampling rule.", "paragraph", "methods"),
            (6, "## H EXPERIMENTAL DETAILS", "heading", "results"),
            (6, "Extra runs on more data.", "paragraph", "results"),
        ]
    )
    assert "derive" not in result["sections"]["methods"]
    assert "Extra runs" not in result["sections"]["results"]
    assert result["sections"]["results"].endswith("It worked.")
    assert not [r for r in result["review"] if r["page"] >= 5]
    assert all(b["appendix"] for b in result["ledger"] if b["page"] >= 5)


def test_named_appendix_before_references_is_never_rewritten():
    result = run(BASE + [(4, "# Appendix A. Proofs", "heading", "other"), (4, "Proof of the lemma.", "paragraph", "results")])
    assert "Proof" not in result["sections"]["results"]


def test_digit_numbered_methods_after_references_are_still_rewritten():
    # Nature-style papers place Methods after the reference list.
    result = run(
        BASE
        + [
            (4, "## References", "heading", "other"),
            (4, "[1] A. Author. A paper. 2020.", "references", "other"),
            (5, "## 4 Methods", "heading", "methods"),
            (5, "Samples were frozen.", "paragraph", "methods"),
        ]
    )
    assert "frozen" in result["sections"]["methods"]


def test_size_problem_names_the_reason():
    ok = {"title": "A", "abstract": "word " * 200, "methods": "word " * 3000}
    assert rewrite_size_problem(ok) is None
    one_huge_passage = {**ok, "results": "measurement " * 11000}
    assert "results section" in rewrite_size_problem(one_huge_passage)
    too_long = {k: "word " * 6000 for k in ("introduction_rest", "methods", "results", "discussion")}
    too_long["methods"] += "x" * 60000
    assert "25,000 words" in rewrite_size_problem(too_long)


def test_long_sections_split_at_subheadings_and_rebuild_exactly():
    from label import split_section, SECTION_TOKENS

    short = "## Methods\n\nWe measured."
    assert split_section(short) == [short]
    long = "\n\n".join(
        f"### 4.{i} Step {i}\n\n" + "\n\n".join(["The samples were measured again with care. " * 40] * 6)
        for i in range(1, 8)
    )
    pieces = split_section(long)
    assert len(pieces) > 1
    assert "\n\n".join(pieces) == long
    assert all(p.startswith("### 4.") for p in pieces)
    assert rewrite_size_problem({"title": "A", "methods": long}) is None


def test_tables_are_kept_separately_and_their_captions_rewritten():
    result = run(
        BASE
        + [
            (3, "<table><tr><td>0.85</td><td>400</td></tr></table>", "table", "results"),
            (3, "Table 1. Counts per site.", "caption", "results"),
        ]
    )
    assert "<table>" not in result["sections"]["results"]
    assert "Table 1. Counts per site." in result["sections"]["results"]
    table = next(b for b in result["ledger"] if b["role"] == "table")
    assert table["disposition"] == "retained_separately" and not table["review_reasons"]
