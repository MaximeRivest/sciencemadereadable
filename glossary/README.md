# Reference glossaries

Goal: give the rewriter the *facts* about each hard term (from reference works), so a
small model's weights go into presentation (wording, rhythm, flow), not into knowledge.

    .venv/bin/python glossary/glossary.py               # first 3 benchmark test papers
    .venv/bin/python glossary/glossary.py PMC12995867   # any corpus paper

Output: `out/<paper_id>.json` (the glossary) and `out/review.md` (tables to read).

Which terms (see the docstring of glossary.py for details):
- hard for a 12–16-year-old: learned at 12 or later (Kuperman et al. 2012 age-of-acquisition
  ratings, `data/osf-vb9je.xlsx`), or not in that list and rare in everyday English (wordfreq);
- AND rarely seen by the model: used in at most 1% of the 5,000 corpus papers, and (single
  words) less common than Zipf 3.3 in everyday English;
- abbreviations take their meaning from the paper itself ("chemical oxygen demand (COD)").

Explanations: Simple English Wikipedia, else Wikipedia, else Wiktionary (first two
sentences), cached in `data/cache/`.

Known gaps (first version, 2026-10-01): some author names and software names slip
through; Wikipedia's first sentences are sometimes technical (correct facts, hard
words: the rewriter's job is to say them plainly); sense errors are flagged by a weak
word-overlap check only.

## First test, no retraining (2026-10-01, 3 benchmark test papers, 15 parts, Opus judge)

| | faithful | understandable | pleasant | structure | mean | serious faithfulness issues |
|---|---|---|---|---|---|---|
| Opus | 8.27 | 8.53 | 8.07 | 8.93 | 8.45 | |
| Opus + glossary (`opus_glossary.py`) | 8.40 | 8.47 | 8.07 | 8.67 | 8.40 | |
| 4B student, step 152 | 6.33 | 7.47 | 7.00 | 8.00 | 7.20 | 23 |
| 4B student, step 152 + glossary | 6.60 | 7.20 | 7.33 | 8.20 | 7.33 | 17 |

The student was never trained with a glossary (`eval_student.py generate --glossary`
adds it to the same prompt). Of the 17 serious issues left, about 4 are knowledge
(acetate "sugar-like", "endogenous", NMDS) and about 13 are reading precision
(which number belongs to which group, "respectively", hedges widened, a cited
study presented as the authors' own).
