# PDF → readable research: off-site prototype

**Working end to end on one 13-page research PDF. Nothing deployed.**

**New: rewrite → Kokoro audio, e-ink web reader, PDF and EPUB.** Open
`reading/index.html`; see [READING.md](READING.md) for downloads, checks and the
one-command `make_readable.sh` pipeline.

Open `demo/index.html` for the original/extracted/re-written comparison, section
mapping, inspected decisions, original PDF pages, and limitations. It is an
offline report: local scripts/assets, no CDN or API calls from the page.

## Result — 2026-10-15

Public CC BY sample: *Multiclass Determination of Endocrine-Disrupting Chemicals
in Meconium: First Evidence of Perfluoroalkyl Substances in this Biological
Compartment*, DOI `10.3390/toxics12010075`, PMC10819471.

- Read **all 13 pages** through Attachments 1.0.1 + LightOnOCR-3-0.8B, with no
  fallback. Rasterization + successful OCR requests: **58.7 seconds**.
- Jev **1.14.0** labelled **149 blocks**, plus prose-continuation judgments at
  page boundaries. 19 requests, **188,682 input tokens**, about **$0.0079** at
  the documented $0.042/million input tokens. Sum of request times: **3.4 s**.
- Four blocks needed inspection before the confidence gate would pass: the
  publication-type label, a table footnote, bibliography entry 40, and the
  publisher disclaimer. Decisions are in `toxics-review.json`, tied to the PDF
  hash **and each exact block's text hash**. These were checked against the
  page images; no source text was changed. **Not an unattended success.**
- The split first introduction paragraph was joined across pages. Combined
  Results and Discussion went to `results`, matching the existing XML splitter.
  No separate discussion was fabricated.
- Generated **100 glossary entries from PDF text**, using the existing offline
  reference database, in **2.1 s**. No saved paper glossary or publisher XML was
  used to construct the input.
- The existing local **our-9b** rewriter completed the opening and all three
  nonempty remaining section calls, using the application's actual prompt
  definitions, `writer="opus"`, temperature 0, and existing reply-limit rules.
  All calls finished normally. **347.7 s** for rewriting, sequentially.
- Roughly **6 min 52 s** of measured successful stage work, excluding development,
  the initial capped OCR attempt, and inspection time. Not a latency benchmark.
- Post-hoc publisher XML check: **32/33 prose paragraphs** met the rule of
  at least 70% five-word overlap in the expected section. The remaining chemical
  reagents paragraph still matched Methods best, but scored 68.0% overlap.
  This is a text-placement diagnostic, **not 97% scientific accuracy**.
- Repeat run: extraction, all Jev requests, glossary, and all rewrite calls came
  from cache. It ran without a TypeSafe key. A local model-inventory GET remains.

This is an existing corpus paper, **not a held-out generalization benchmark**.
No full factual-fidelity, number, equation, figure, or table audit was performed.

## Boundaries and deliberate trade-offs

- Only files in this folder are written. The live queue, paper cache, website,
  workers, and deployment configuration are not touched. The prototype does
  share the existing OCR and rewrite services, so it consumes their GPU time.
  Sequential requests limit, but do not eliminate, contention.
- Every page is rendered to an image-only PDF at up to 400 dpi / 5 megapixels,
  then submitted to the isolated Attachments service. This is slower than native
  text extraction but exercises OCR on text-bearing pages too, without changing
  Attachments. The original PDF is kept. Native text is saved **only for diagnosis**.
- The Attachments service caps OCR at ten pages per request. The prototype uses
  five-page windows and checks `ocr_pages`, page segments, warnings, and fallback
  metadata. The first attempted whole-document request hit this cap; the gate
  rejected it rather than feeding three empty pages to the rewriter.
- Jev receives original blocks and nearby text, plus candidate headings from the
  whole document. It labels structural role and section **membership**, not just
  topic. Narrow choices include `other`, `uncertain`, and `results_discussion`.
  Its parallel questions do not depend on one another's answers.
- Page-continuation questions are speculative: only used for prose/prose pairs.
  The first iteration also gated on these answers for tables; that orchestration
  bug was fixed. Table fragments remain separate and intact in the ledger.
- The 0.65 confidence gate is provisional and uncalibrated on research PDFs.
  Unknown body material, missing required sections, or uncertain labels stop
  input preparation. Overrides must record an inspection reason and cannot
  change the text. There is no silent “force through” option.
- Every non-whitespace character **returned in the OCR page segments** is
  accounted for exactly once in the block ledger. This does not establish that
  OCR recovered every character in the PDF. References and metadata remain
  available separately; they are not fed to the rewriter.
- Tables are included in the raw model input to keep the existing prompt format;
  this costs output tokens and is not a recommendation for production. Generated
  tables are hidden in the report. OCR tables are labelled as extraction, not as
  authoritative source numbers. Original PDF pages are always available.
- Rendering is not figure interpretation. Formula markup is not fully typeset
  in this report. Text that failed OCR cannot be repaired by section labels.
- Prototype limits: English standard research papers, 1–20 pages, original and
  rendered PDF each at most 25 MiB. Password-protected PDFs, empty pages, OCR
  warnings/fallbacks and unsupported structures fail closed. This is a local
  development tool, **not a public upload endpoint or security boundary**.
- Text is sent to TypeSafe. Do not use confidential/unpublished PDFs without
  permission. Keys stay in the environment and are never written to run outputs.
  Run files stay on disk until you delete them; there is no automatic retention
  policy in this local prototype.

## Reproduce the sample

From this directory, on lambda:

```bash
uv sync --frozen
mkdir -p runs/toxics
curl -fL \
  'https://mdpi-res.com/d_attachment/toxics/toxics-12-00075/article_deploy/toxics-12-00075.pdf' \
  -o runs/toxics/source.pdf

# Existing credential file, found in the project configuration.
set -a
source ~/.config/scholarsreadinglist/typesafe.env
set +a

# First pass deliberately stops at four uncertain blocks.
./run.sh runs/toxics/source.pdf runs/toxics

# Resume with the documented inspections for this exact PDF and these blocks.
./run.sh runs/toxics/source.pdf runs/toxics toxics-review.json

# Independent diagnostic, AFTER the PDF-derived rewrite. XML is not required
# for extraction/labelling/glossary/rewrite, only for this paired report.
../.venv/bin/python compare_xml.py runs/toxics ../paper_corpus/xml/PMC10819471.xml
.venv/bin/python report.py runs/toxics
```

For another PDF: `./run.sh /path/paper.pdf runs/another-paper`. Use a **new run
folder** for another input/settings. If the gate stops, inspect `labelled.json`
and the page images. Create a reviewed decisions file in the same format as
`toxics-review.json`; do not reuse this sample's labels.

Dependencies are isolated in this folder's `.venv` and `uv.lock`. Glossary
preparation deliberately uses `../.venv/bin/python` and the existing read-only
reference database. Rewriting imports `../app/web/src/programs.ts` and the app's
installed lm15 client; run `npm install` in `../app` if those are missing. This is
an in-checkout prototype, not yet a portable package.

Endpoints: `extract.py --service URL` overrides Attachments (default
`http://127.0.0.1:8820`). `REWRITER_URL` and `REWRITER_MODEL` override the local
rewriter (defaults `http://127.0.0.1:8015/v1`, `our-9b`).

## Files and tests

| File | Responsibility |
|---|---|
| `extract.py` | Bounded rendering, five-page OCR windows, complete-page/fallback checks |
| `label.py` | Verbatim source spans, Jev questions/cache, review gate, section assembly |
| `prepare_input.py` | Existing glossary and reply limits, exclusively from PDF sections |
| `rewrite.ts` | Actual application prompts → local model; checks finish reason and fields |
| `compare_xml.py` | Independent, post-hoc prose placement diagnostic |
| `report.py`, `report.html` | Offline interactive report with sanitized Markdown |
| `run.sh` | Extraction → labelling → glossary → rewrite, no public queue |
| `toxics-review.json` | Four explicit, hash-bound inspections for this sample |
| `runs/toxics/` | Full extraction, block ledger, raw requests/replies, logs, cache, rewrite |
| `demo/` | Reader-facing report with sample PDF, pages and rewrite |

```bash
.venv/bin/python -m unittest -v test_pipeline
node --conditions=functai-source ../app/tools/check_prompts.ts

# NixOS: use a separate stock Chromium, not the household's shared-profile launcher.
CHROMIUM_BIN="$(nix build nixpkgs#chromium --no-link --print-out-paths)/bin/chromium" \
  .venv/bin/python check_report.py demo
```

Verified: **11 offline tests**, **16 existing prompt-parity checks**, and a
Chromium walkthrough of desktop/mobile layouts, all four tabs, page images,
generated-table hiding and dark theme, with no script errors. The browser
check is intentionally sample-specific (page 13 and the 32/33 diagnostic).

## Before considering a site integration

1. Test a diverse held-out PDF/XML set: two-column layouts, scans, unusual section
   names, missing conclusions, long tables, formulas, combined sections.
2. Measure extraction omissions separately from section classification mistakes,
   review rate, downstream rewrite fidelity, latency and actual billed usage.
3. Add reliable table/figure handling and table continuation, with originals as
   the displayed authority; avoid spending rewrite tokens regenerating cells.
4. Add an actual upload sandbox, job isolation, privacy/retention controls,
   reviewed failure UI, context-size limits and fair scheduling.
5. Choose review thresholds from measured errors rather than this one sample.
