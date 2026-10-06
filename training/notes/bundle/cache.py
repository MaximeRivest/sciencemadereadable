"""
The shared glossary cache
=========================

A plain meaning is written once and reused by every later paper: p-value, ANOVA, salinity...
Only words the cache does not have yet go to the model. The meaning is general (what the word
means), so it holds across papers; what is central in a paper, and its picture, stay per paper.

    glossary_cache.json   {"p-value": {"term": "p-value", "meaning": "...", "paper": "PMC..."}}

The model may replace a cached meaning that is wrong for a paper (another sense of the word);
that replacement stays in the paper's own glossary and does not overwrite the cache.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PATH = Path(__file__).resolve().parent / "glossary_cache.json"


def load() -> dict:
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def save(cache: dict) -> None:
    PATH.write_text(json.dumps(dict(sorted(cache.items())), indent=1, ensure_ascii=False))


def known_in(cache: dict, text: str) -> list[dict]:
    """Cached entries whose term appears in text (whole words, any case)."""
    low = text.lower()
    return [e for k, e in cache.items() if re.search(rf"(?<![\w-]){re.escape(k)}(?![\w-])", low)]


def add(cache: dict, entries: list[dict], paper_id: str) -> int:
    """Add the new entries of one paper; returns how many were new."""
    n = 0
    for e in entries:
        key = e["term"].strip().lower()
        if key and key not in cache:
            cache[key] = {"term": e["term"].strip(), "meaning": e["meaning"], "paper": paper_id}
            n += 1
    return n
