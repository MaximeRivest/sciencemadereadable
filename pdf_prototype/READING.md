# Rewrite → audio and comfortable reading

The off-site pipeline now produces an e-ink web reader, EPUB, PDF, and Kokoro MP3.
Open **`reading/index.html`**. The downloads and chapter player are on its first page.
Nothing was added to or published on the live ScienceMadeReadable site.

**Original visuals are now integrated:** all **two figures and four tables** in
the sample PDF appear beside their rewritten captions. Tables 1 and 2 each retain
both source-page parts, for eight original crops in total. The web reader has a
zoom/pan viewer and a figures-and-tables gallery. PDF placements preserve original
vector text; EPUB and web use direct 300-dpi source images, never OCR cells.

## What was made for the sample

| Output | File | Result |
|---|---|---|
| E-ink web reader | `reading/index.html` | Paged text, inline originals, tap-to-enlarge zoom/pan, chapter navigation, font/line spacing controls, saved position; no animations |
| EPUB 3 | `reading/paper.epub` | Reflowable text with inline original crops and full-size image views, plus the 13-page source appendix; about 7.9 MiB |
| Reading PDF | `reading/paper.pdf` | 45 A5 reading pages + eight larger A4 original-visual pages at the relevant passages + the 13-page source appendix; about 896 KiB |
| Full Kokoro audio | `reading/paper.mp3` | 44 min 33 s, British male `bm_george`, 1× synthesis speed, 64 kbps mono MP3; about 21 MiB |
| Chapter audio | `reading/audio/*.mp3` | About this recording, abstract, introduction, methods, results/discussion, conclusion |
| Speech transcript | `reading/narration.txt` | Exact prepared text supplied to TTS, in playback order |
| Chapter times | `reading/chapters.txt` | Start times in the full recording |
| Offline bundle | `reading/reading.zip` | Reader, audio, source pages and downloads together; about 58 MiB |

The audio build took about **55 seconds** on the existing local Kokoro service.
There were 119 bounded narration chunks; repeated identical chunks reused their
cache. This is one measured run, not a latency guarantee. Local speech and rewrite
services are shared with other applications; no service was restarted or reconfigured.

## For an Android e-paper tablet

- **Web:** open the reading artifact, then “Read on this screen”. Use Previous/Next
  or keyboard page keys. Font size and line spacing can be changed without smooth
  scrolling. Reading and audio progress are stored locally when the browser allows
  storage—not synced between devices.
- **EPUB:** download and open in the tablet's EPUB reading app. That app controls
  fonts, margins and pagination. This is the most portable reflowable format.
- **PDF:** use the large-print edition for fixed pages. Original tables/figures are
  in the appended source, with bookmarks. A separate byte-for-byte original PDF
  is included too; it is the authority for exact source appearance.
- **Offline web/audio:** download and extract the ZIP while online. Keep its files
  together. The HTML embeds the reading text and needs no external scripts; audio
  and source pages are relative files. Whether Android permits opening local HTML
  with sibling media depends on its browser/file manager. EPUB and MP3 downloads
  avoid that limitation. This is **not** a service-worker/PWA offline install.

Tested in Chromium at phone, tablet and desktop sizes, **not on the physical
Android tablet**. Hardware ghosting, refresh modes, touch latency, EPUB-reader
behaviour and lock-screen/background audio still need testing on that device.

## What changes—and what does not

The reading edition uses the existing rewrite, not another summarization call.
A single prepared chapter representation supplies the web, EPUB and PDF.

- Eight generated table/plot-data blocks are removed. The original two figures
  and four tables replace them at their reviewed caption locations. Multi-page
  tables retain their separate parts and all source footnotes. Original captions
  inside the crops remain distinct from the rewritten caption text.
- Web and EPUB originals are rendered directly from the source PDF at 300 dpi.
  The web viewer offers zoom/pan, next/previous part, and a sharp vector crop PDF.
  The print pipeline first lays out A5 text and A4 visual pages, then replaces
  the image placements with source PDF content: table text is still selectable
  and sharp when zoomed. The complete original paper remains appended.
- For this prototype, crop coordinates, continuations, and caption anchors were
  explicitly inspected. This is **not automatic detection for arbitrary PDFs**.
  `toxics-visuals.json` records the source PDF hash, rewrite hash, reviewed boxes,
  source-text checks, and exact caption anchors. It accounts for every numbered
  native caption and all generated-table placeholders in this sample. Missing or
  ambiguous mappings fail instead of silently attaching the wrong image.
- Supplementary figures/tables are not present in the supplied PDF and were not
  downloaded or invented. Only the originals actually present are integrated.
- Narration is prepared before visual insertion and is unchanged by this update;
  the existing MP3 is reused without new rewriting or speech synthesis.
- Narration removes Markdown/HTML and numbered citation markers. It announces
  tables and mathematical expressions instead of reading raw cells or TeX.
- A small explicit pronunciation map expands units such as `ng g⁻¹` to “nanograms
  per gram”, and spells out a few initialisms. Numerical values and inequalities
  are not rounded or changed. The transcript makes these adaptations inspectable.
- The speech stream is split into bounded chunks without losing or duplicating
  non-whitespace text. Every completed PCM chunk is cached by text, voice, speed,
  endpoint and format contract, with a content hash. Failed/partial responses are
  not used as successful cache entries.
- The current Kokoro service provides no model-version identifier or word timing.
  Clear `runs/…/audio-cache/` if its model changes and you want fresh speech.
  There is **chapter-level** navigation, not word-level text/audio synchronization.
- MP3 chunks are checked for valid audio format/duration, not pronunciation or
  scientific accuracy. **The 45-minute recording has not been listened through.**
  Long chemical names and mathematical language especially need listening review.
- This remains a prototype based on a rewrite with four inspected section labels,
  not a fully fact-checked or fully unattended publishing pipeline.

The full MP3 and separate chapter tracks duplicate encoded audio deliberately:
convenient downloads and chapter playback cost extra disk/ZIP space. They do not
require duplicate synthesis. EPUB/web use raster crops for compatibility, so their
labels do not reflow like prose. PDF retains vector source content for sharp zoom.
Dense visuals get separate larger pages rather than being squeezed into A5 text;
this deliberately produces a mixed-page-size PDF.

## Run the whole pipeline

From `sciencemadereadable/pdf_prototype`:

```bash
# Existing TypeSafe credential; never written into any output.
set -a
source ~/.config/scholarsreadinglist/typesafe.env
set +a

# Use a separate stock browser for PDF export, not the household browser launcher.
export CHROMIUM_BIN="$(nix build nixpkgs#chromium --no-link --print-out-paths)/bin/chromium"

./make_readable.sh runs/toxics/source.pdf runs/toxics \
  --out reading --audio --pdf \
  --review toxics-review.json \
  --visuals toxics-visuals.json \
  --reference-xml ../paper_corpus/xml/PMC10819471.xml
```

For a new paper, use fresh run/output folders and **omit the sample review file**.
The pipeline stops if labels need inspection; review them, save the hash-bound
inspection file, and resume. It will not bypass the review gate just to produce
an audiobook. A new paper also needs its own reviewed visual manifest—never reuse
the sample's crop boxes. The first successful `--visuals` export saves that mapping
as `runs/…/original-visuals.json`, so subsequent exports retain the originals.
Changing visuals in a folder with an existing PDF requires `--pdf` to rebuild it.

Web + EPUB are always built. `--audio` and `--pdf` are optional.
`--reference-xml` is optional; it supplies a post-hoc check and author/licence
metadata. Its DOI must be found in the PDF's extracted text. No reference XML is
used for rewriting. Without it, the edition clearly marks source licence metadata
as unestablished rather than inventing it. Do not distribute such an edition
without checking its permissions and attribution.

For an **already completed rewrite**, skip extraction/rewriting:

```bash
uv sync --frozen
.venv/bin/python editions.py runs/toxics --out reading --audio --pdf
# Optional: --voice af_heart --speed 1.1 --tts-url http://127.0.0.1:8880
```

`--audio` resumes cached chunks, then re-encodes files. Omitting it preserves an
existing matching audio edition, if present. Reuse an output folder only for the
same rewrite. All files remain local until explicitly shared; no automatic
retention policy or public upload endpoint was added.

## Checks

```bash
.venv/bin/python -m unittest -v test_pipeline test_editions test_visual_assets
CHROMIUM_BIN="$CHROMIUM_BIN" .venv/bin/python check_reader.py reading
nix run nixpkgs#epubcheck -- reading/paper.epub
```

Verified:

- 27 offline tests: source accounting, review gates, safe markup, generated-table
  removal, numeric/citation handling, lossless chunking, valid EPUB XML, exact
  source-crop pixels, continuation parts, placement anchors and mismatch rejection.
- EPUBCheck 5.3.0: **no errors or warnings**, EPUB 3.3 rules.
- Chromium: page turns, resume, changed fonts/line spacing, phone/tablet/desktop
  layouts, inline images fully inside their pages, gallery, zoom/pan, continuation
  navigation, vector downloads, chapter audio loading/seeking and download URLs.
- All eight web/EPUB crops match direct source rendering pixel for pixel. The
  reading PDF contains all eight inline original placements on A4 pages, with
  source table values verified as selectable PDF text, not raster-only pictures.
- Every MP3's duration agrees with its PCM within 0.2 seconds.
- PDF appendix text matches the source exactly; rendered page geometry matches,
  allowing tiny document-colour-profile differences. The separate original PDF
  is unchanged.
- ZIP integrity and expected contents checked.
- Full pipeline rerun reused extraction, labels, glossary and rewrite; audio reused
  all 119 chunks, with **zero new synthesis requests**.
- The actual artifact preview loaded successfully, including audio metadata,
  paged reading and HTTP byte-range audio requests (206); no script errors.
