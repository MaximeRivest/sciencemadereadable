import pytest
from page_sort import choose_pages


def plan(kinds, confidence=None):
    confidence = confidence or [0.95] * len(kinds)
    answers = {n: {"choice": k, "confidence": c} for n, (k, c) in enumerate(zip(kinds, confidence), 1)}
    return choose_pages(answers, len(kinds))


M, R, A, F, T = "main_text", "references", "appendix", "front_or_back_matter", "main_text_then_references"


def test_main_text_end_page_is_read_and_the_rest_skipped():
    p = plan([M, M, M, T, R, R, A, A])
    assert p["read"] == [1, 2, 3, 4, 5] and p["skipped"] == [6, 7, 8] and p["main_text_ends"] == 5


def test_title_pages_are_always_read():
    assert plan([F, R, M, M, R])["read"] == [1, 2, 3, 4, 5]


def test_unsure_pages_before_the_end_are_read():
    p = plan([M, M, A, M, R, A], [0.95, 0.95, 0.6, 0.95, 0.9, 0.9])
    assert p["read"] == [1, 2, 3, 4, 5]


def test_unsure_end_does_not_end_the_main_text():
    p = plan([M, M, M, R, R], [0.95, 0.95, 0.95, 0.7, 0.95])
    assert p["read"] == [1, 2, 3, 4, 5]


def test_confident_main_text_after_references_is_read():
    # e.g. figure pages placed after the reference list
    assert plan([M, M, M, R, R, M, A])["read"] == [1, 2, 3, 4, 6]


def test_no_end_found_reads_everything():
    assert plan([M] * 6)["read"] == list(range(1, 7))


def test_every_page_must_be_answered():
    with pytest.raises(ValueError):
        choose_pages({1: {"choice": M, "confidence": 1}}, 2)
