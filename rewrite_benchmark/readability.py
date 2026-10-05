"""Free, automatic readability metrics for a text, to iterate on pleasant and clear rewriting.

    from readability import measure, histogram
    m = measure(text)          # dict of numbers (see FIELDS)
    histogram(text, "sentence") / histogram(text, "word_age")   # for distribution plots

Sentence length: words per sentence (sentences of 3+ words; headings, table rows and
captions-like fragments under 3 words are ignored).
Paragraph length: words per paragraph.
Word age: for every word of 4+ letters found in the Kuperman et al. (2012)
age-of-acquisition list, the age at which it is usually learned. Names, abbreviations
and numbers are not counted. The reader's age (kid.READER_AGE) marks "late" words.
Grade: Flesch-Kincaid grade level (syllables estimated from vowel groups): a rough,
widely used readability index, useful for comparing versions of the same text.
"""
from __future__ import annotations

import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "glossary"))
import glossary as G                                              # noqa: E402
from kid import READER_AGE                                        # noqa: E402

FIELDS = {
    "words": "words",
    "sent_mean": "words per sentence (mean)",
    "sent_p90": "words per sentence (90th percentile)",
    "sent_over_30": "% sentences over 30 words",
    "para_mean": "words per paragraph (mean)",
    "para_over_150": "% paragraphs over 150 words",
    "age_mean": "word age, mean (years)",
    "age_over_12": "% words usually learned at 12+",
    "age_over_reader": f"% words usually learned at {READER_AGE}+",
    "fk_grade": "Flesch-Kincaid grade",
}


def _clean(text: str) -> str:
    text = re.sub(r"⟦([^⟧]+)⟧", r"\1", text or "")
    text = re.sub(r"^\s*##.*$", "", text, flags=re.M)               # section labels
    text = re.sub(r"^\s*\|.*$", "", text, flags=re.M)               # table rows
    return re.sub(r"\[[\d,\s–-]+\]", "", text)                       # citation brackets


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", _clean(text)) if len(s.split()) >= 3]


def paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", _clean(text)) if len(p.split()) >= 8]


def word_ages(text: str) -> list[float]:
    out = []
    for w in re.findall(r"(?<![A-Za-z])[a-z][a-z'-]*[a-z](?![A-Za-z])", _clean(text)):
        if len(w) < 4:
            continue
        for f in G.word_forms(w):
            if f in G.AGES:
                out.append(G.AGES[f])
                break
    return out


def _syllables(w: str) -> int:
    w = w.lower()
    n = len(re.findall(r"[aeiouy]+", w)) - (1 if w.endswith("e") and not w.endswith("le") else 0)
    return max(1, n)


def measure(text: str) -> dict:
    ss = [len(s.split()) for s in sentences(text)]
    ps = [len(p.split()) for p in paragraphs(text)]
    ages = word_ages(text)
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", _clean(text))
    syl = sum(_syllables(w) for w in words)
    pct = lambda xs, f: 100 * sum(f(x) for x in xs) / max(1, len(xs))
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    return {
        "words": len(words),
        "sent_mean": st.mean(ss) if ss else 0, "sent_p90": q(ss, 0.9), "sent_over_30": pct(ss, lambda x: x > 30),
        "para_mean": st.mean(ps) if ps else 0, "para_over_150": pct(ps, lambda x: x > 150),
        "age_mean": st.mean(ages) if ages else 0, "age_over_12": pct(ages, lambda a: a >= 12),
        "age_over_reader": pct(ages, lambda a: a >= READER_AGE),
        "fk_grade": 0.39 * len(words) / max(1, len(ss)) + 11.8 * syl / max(1, len(words)) - 15.59 if ss else 0,
    }


def histogram(text: str, kind: str) -> tuple[list[float], list[float]]:
    """(bin centres, share of items per bin) for "sentence" (words, bins of 3, to 72) or
    "word_age" (years, bins of 0.5, 2-18)."""
    if kind == "sentence":
        xs, lo, hi, w = [min(len(s.split()), 71) for s in sentences(text)], 0, 72, 3
    else:
        xs, lo, hi, w = [min(a, 17.9) for a in word_ages(text)], 2, 18, 0.5
    n = int((hi - lo) / w)
    counts = [0] * n
    for x in xs:
        counts[min(n - 1, int((x - lo) / w))] += 1
    total = max(1, len(xs))
    return [lo + w * (i + 0.5) for i in range(n)], [c / total for c in counts]
