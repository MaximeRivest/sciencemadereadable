# Shared reading engine

This is the common implementation used by the off-site prototype and the online
reading service. Prototype scripts remain as symlinks so fixes do not drift.

- `jats_source.py`: canonical JATS text, section and licence rules, shared with
  corpus/benchmark preparation and archived-source verification.
- `extract.py`: bounded PDF rendering and the original local extraction adapter.
  The public service renders in a sandbox and uses raw RGB → controller-generated
  PNG → Attachments image OCR (`app/editions/image_ocr.py`), not a forwarded PDF.
- `label.py`: Jev classification, source-span accounting, section assembly and
  review gating. The HTTP API uses only its non-parsing assembly functions.
- `prepare_input.py`: the existing offline glossary and model reply limits; no
  private-document online dictionary fallback.
- `rewrite.ts`: the application's actual rewrite programs. Production bundles
  this with `--conditions=functai-source`; prompt parity is a release check.
- `editions.py`: reading representation, EPUB, mixed-size/vector-source PDF,
  Kokoro narration, chapter audio and downloads.
- `visual_assets.py`: reviewed source crop/placement manifests and format adapters.
- `automatic_visuals.py`: conservative automatic figure pairing and source-page
  panels. It does not infer table cells or assert uncertain crop boundaries.
- `reader.html`: dependency-free paged web reader, original-image zoom/pan,
  chapter audio, saved place and offline-friendly downloads.

File parsing/export must not be invoked on public uploads outside the service's
sandbox. The local prototype is a development tool, not a security boundary.
See `app/editions/README.md` for deployment and the private-job contract.
