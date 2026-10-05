# Scientific rewriting pilot

The runnable notebook is [`../../notebooks/scientific-rewriting.md`](../../notebooks/scientific-rewriting.md).

## Start

```sh
rat doctor notebooks/scientific-rewriting.md
rat ensure notebooks/scientific-rewriting.md
rat play notebooks/scientific-rewriting.md
```

The default run is local only: it prepares source data, verifies the reference export, displays examples, defines the programs, and checks configuration. It makes no inference calls. Set `RUN_WRITERS=True` to generate; review the rubric and set `RUBRIC_APPROVED=True` and `RUN_JUDGES=True` to evaluate. Model IDs are explicit: `openai-codex:gpt-6-astra` and `openai-codex:gpt-6-luna`. Live inference access has not been tested by creating this notebook; model-list discovery is optional and no fallback is made.

A full uncached experiment is 10 writing calls plus up to 156 judging calls. There are no automatic retries and no call limit. Dollar cost is unknown rather than assumed zero. Outputs, token usage, elapsed times, input fingerprints, program versions, and failures are saved in the ignored `runs/` directory. Identical completed calls are reused. Each model judges every available candidate, including itself; judge results remain uncalibrated until reviewed by people.

## Data

- `data/paper.json`: original JATS text, section boundaries, tables, references, source attribution/checksum.
- `data/source_evidence.json`: explicitly labelled Figure 1 transcription and source conflicts. It supplies the same evidence to both writers and judges, without supplying the prior rewrites as target answers.
- `data/reference_messages.jsonl`: the five complete, visible answer texts from the conversation.
- `data/reference_pairs.jsonl`: eight original sections paired with verbatim rewrite excerpts, message IDs, character spans, and checksums. Editorial notes stay separate.
- `data/reference_document.md`: the assembled previous rewrite, convenient to read.
- `data/provenance.json`: selected session/branch IDs, hashes, and extraction policy.

Source paper: Verónica Loewe-Muñoz, Claudia Delard, Rodrigo del Río, Mónica Balzarini, and Dusan Gomory (2024), *Recommendations for increasing yield of the edible Pinus pinea L. pine nuts*, PLOS ONE, https://doi.org/10.1371/journal.pone.0300008. CC BY 4.0. Plain-text formatting and explanatory adaptations are not the original publisher layout.

Conversation source: session `01a0e736-a851-75b8-8af7-9bab57c64275`, September 28, 2026. Recorded model: `openai-codex/gpt-6-astra`. This is a user-guided, tool-assisted reference, not a human-written answer or an independently verified gold standard. Only the selected visible answers and limited provenance were exported, never thinking, tools, system prompts, or unrelated content.

## Re-extract, if needed

```sh
python rewrite_benchmark/prepare.py \
  --session ~/.pi/agent/sessions/--home-maxime-Projects-scholarsreadinglist--/2026-09-28T08-51-54-066Z_01a0e736-a851-75b8-8af7-9bab57c64275.jsonl
```

The selected discussion answer pins the branch. Other papers require their own explicit section mapping and reference set: the preparation script refuses to apply this pilot's pairings to a different DOI.

## Offline checks

```sh
.venv/bin/python rewrite_benchmark/check.py
```

Runs every notebook cell with inference disabled, verifies hashes/exact spans, renders a real FunctAI request without sending it, then tests the whole pipeline with explicitly simulated responses in a temporary directory. Covers context handoffs, caching, all 156 assessment rows, anonymous judge inputs, reading context, rating-sheet expansion, and invalid-quotation detection. Simulated answers and scores never enter the experiment's real results.

## Scientific limits

This is a development pilot on one paper already used to shape the task. Scores are diagnostic, not an estimate across literature. Evaluate the evaluator before optimizing the translator. The notebook does not average dimensions into one score, does not turn judge failures into low-quality writer scores, and does not claim that two related model judges provide independent verification. Keep all sections and versions of a paper together when later splitting into development and held-out test sets.
