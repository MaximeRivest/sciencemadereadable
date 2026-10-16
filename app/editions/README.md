# Private reading editions

PDF → local Attachments/LightOnOCR → Jev document/section classification → reader
review → existing 9B rewrite and offline glossary → e-ink web reader, PDF, EPUB,
optional Kokoro audio. Existing server-saved rewrites skip OCR/classification and
rewriting, using their verified CC BY source record and PMC's original PDF. The
archived XML must reproduce every original section exactly before its matching
PDF version is accepted; a matching DOI alone is not enough.

The rendering code is shared in `../../reading_engine`; the former prototype
entry points are symlinks to it. No production code imports a prototype output.

## User contract

- English original-research PDFs, 1–20 pages, at most 25 MiB.
- Explicit permission and TypeSafe processing consent before upload. PDF text
  goes to TypeSafe; OCR, glossary, rewriting and speech are local. Offline
  glossary lookup refuses an online fallback for private documents. The offline
  snapshot has no Wiktionary rows, so private documents skip that dictionary
  source (public papers still query it); a few one-word definitions may be absent.
- PDF sections are always reviewed before rewriting. Low-confidence decisions
  require explicit block approvals; unknown/missing required sections stop.
- Private jobs are not inserted in the public library or public rewrite queue.
- Access is a random 256-bit capability, saved in the reader's browser or an
  explicitly downloaded recovery file. The server stores only its hash. No
  capability appears in a URL. Opening a reader exchanges the bearer for a
  Secure, HttpOnly, SameSite=Lax cookie scoped to that one job's path.
- Files expire after 48 hours, and can be deleted sooner. Expiry is checked on
  every access; cleanup runs in the worker loop and on restart. An offline worker
  removes expired files when it returns. Downloaded copies do not expire.
- Two unfinished jobs and eight creations/day/connection, bounded queue, one
  private worker, 12 GiB total state cap, 900 MiB/job, 40-minute processing limit.
  Saved-paper exports have two reserved queue slots. These are anonymous abuse
  limits, not an account/authentication system.
- Audio is optional and capped at 90 minutes. Failure leaves the reading formats
  available and offers an audio retry. There are at most two manual retries.
- Originals are never reconstructed from generated cells. Clearly paired embedded
  figures can be cropped; uncertain boundaries use explicitly labelled original
  source pages. Their position uses matching captions, otherwise a clearly
  labelled source section. This is not perfect arbitrary-layout detection.
- The combined PDF keeps original vector content and visible annotations/form
  appearances, without interactive source actions. The separately downloaded
  original remains byte-for-byte unchanged. Crops are for reading, **not redaction**.
- Complex TeX is retained in its written notation rather than silently simplified;
  complete mathematical typesetting is not part of this release. Source/format
  notes identify unplaced visual references and retained notation. Audio announces
  formulas instead of reading raw symbols.
- No guarantee of factual correctness, pronunciation, or physical-tablet behaviour.
  The website says to check the original before relying on a detail.

## Trust boundaries

`service.py` is a small HTTP API; `worker_main.py` is a separate system service.
A parser/renderer OOM cannot take down the HTTP process. A queue lock prevents two
workers owning the same state; a persistent lease coordinates cancellation and
deletion. Interrupted work resumes from atomic caches, with bounded restart
recovery rather than an endless retry loop.

Untrusted PDF operations run in bubblewrap with a private PID/mount/network/IPC
namespace, no keys or home, immutable code, CPU/file/process limits, read-only
inputs and one writable output directory. Export cannot write the controller's
job status or another job. Symlinks, hard links, special files and excessive
output are refused. Parser diagnostics are bounded temporary files, not unbounded
captured output or writes through parser-chosen paths.

For upload OCR, only bounded **raw RGB pixels** leave the PDF renderer. The
controller encodes a minimal PNG itself and sends that to Attachments. It never
forwards a parser-produced PDF/image container to another decoder. Production
uses Attachments' private Unix socket, not its public per-IP upload quota.
Jev receives text as data, not executable code. Its keys never enter the PDF
sandbox or browser.

Exported HTML/metadata and EPUB contents are validated outside the sandbox.
Only the trusted template supplies scripts. Private HTML has hashed-script CSP;
files have `no-store`, `noindex`, `nosniff`, and no-referrer headers. PDFs/EPUBs/ZIPs
are downloads. SVG/scripts/remote images are not accepted as generated assets.
File access is owner-checked and traversal/symlink guarded; audio supports ranges.

The main site's `editions_proxy.py` forwards only the declared routes, with bounded
bodies/concurrency/timeouts. It replaces client-identity headers, forwards private
cookies/ranges, and does not apply the legacy API's wildcard CORS to these routes.
Only the site's two HTTPS origins receive credentialed CORS. The document API is
loopback-only. Do not include `/var/lib/smr-editions` in long-lived backups.

## Layout

- `service.py`, `store.py`: private API, SQLite state, worker controller, retention.
- `worker_main.py`: separate worker entry point, graceful stop and queue ownership.
- `pipeline.py`: stage orchestration, source verification, export verification.
- `sandbox.py`, `pdf_ops.py`: networkless file processing and isolation probe.
- `image_ocr.py`: raw-pixel boundary and bounded Attachments image OCR calls.
- `../../reading_engine/`: common extraction, section assembly, glossary, rewrite,
  rendering, visual placement, EPUB/PDF, TTS and e-ink reader.
- `../web/src/editions.ts`: upload/consent, private jobs, section review, recovery,
  download opening, cancellation, retry and deletion.
- `../editions_proxy.py`: production and development reverse proxy.

## Develop

```sh
# Nix Python needs the shared foreign-library directory for PyMuPDF.
uv sync --frozen --python /path/to/nix/python3.13
export LD_LIBRARY_PATH=/run/current-system/sw/share/nix-ld/lib
export CHROMIUM_BIN="$(nix build nixpkgs#chromium --no-link --print-out-paths)/bin/chromium"
./dev.sh up
# Main dev site: ../tools/dev.sh up (or restart smr-dev-site after proxy edits).
# https://lambda.tail69222b.ts.net:18940/?reading

.venv/bin/python -m pytest -q test_service.py test_exports.py test_proxy.py test_image_ocr.py test_sources.py \
  ../../pdf_prototype/test_pipeline.py ../../pdf_prototype/test_editions.py ../../pdf_prototype/test_visual_assets.py
.venv/bin/python check_browser.py --upload       # public test sample only; stops at review
.venv/bin/python check_browser.py --confirm      # approves only the explicitly inspected sample blocks
```

Browser test state in `/tmp/smr-editions-browser-state.json` contains a capability:
mode 0600, never commit or publish it. Tests create real work; do not rerun them
unnecessarily. Screenshots go to `/tmp/smr-editions-browser`.

The API accepts raw `application/pdf`, not multipart archives or arbitrary URLs.
Main routes (all job routes require its bearer or cookie):

```
GET    /api/editions/health
POST   /api/editions/jobs?audio=true     X-SMR-Consent: typesafe; X-SMR-Rights: permitted
POST   /api/editions/papers             {doi, model, audio}: an existing saved rewrite
GET    /api/editions/jobs/ID
GET    /api/editions/jobs/ID/review
POST   /api/editions/jobs/ID/confirm     {confirmed:true, changes:{block:{role,section,boundary}}}
GET    /api/editions/jobs/ID/pages/N
POST   /api/editions/jobs/ID/open        set the path-scoped cookie, return reader URL
POST   /api/editions/jobs/ID/retry
POST   /api/editions/jobs/ID/cancel
DELETE /api/editions/jobs/ID
GET    /api/editions/jobs/ID/files/NAME  authenticated exports, including byte-range audio
```

There is no public job listing. Recovery files contain the capability, not an
administrator token. Rate-limit identities are HMACs, not stored IP addresses.

## Deploy and roll back

The declarative units live in
`~/Projects/os/machines/hosts/lambda/smr-editions.nix`. Follow `~/Projects/os/README.md`:
commit/push the machine configuration and deploy it to both machines. The units
skip cleanly until a code release exists. Only the worker receives the TypeSafe
credential and the read-only glossary reference mount.

```sh
# From a clean, committed service/engine checkout:
app/editions/release.sh prepare
app/editions/release.sh activate COMMIT
app/editions/release.sh status
```

`prepare` archives committed source, checks prompt parity, builds a self-contained
Node worker with `functai-source` resolution, installs the locked Python runtime,
roots its Nix interpreter, and writes an immutable root-owned release under
`/opt/smr-editions/releases/COMMIT`. The original source, locked dependencies,
compiled-worker checksum and commit are recorded. Secrets/state are never copied.

`activate` sets a maintenance marker under a queue transaction, refuses a busy
worker, switches the `current` symlink, restarts API + worker, and waits for a
**fresh worker heartbeat**. A failed health check restores the preceding release.
State is separate and kept across releases. Keep previous code/runtime roots for
rollback; do not delete a release still in use.

The production **launch gate starts closed**. Local operator smoke requests can
create jobs without the proxy's client header; public proxy requests cannot.
After sandbox, real-job, private-access, and browser checks:

```sh
app/tools/publish.sh queue    # only when the legacy rewrite queue is empty
# Check the public proxy/private downloads while new public jobs are still gated.
app/editions/release.sh enable
app/tools/publish.sh          # staged and live browser checks, committed/pushed source
```

Backend must be deployed before the front end. Test the new proxy on the public
API before publishing the page. Never set `SMR_READING_DEV=1` in production.

Rollback:

```sh
app/editions/release.sh disable   # stop new public jobs; existing downloads remain usable
app/tools/publish.sh rollback    # restore the previous public page
app/editions/release.sh rollback  # preceding worker/API release, only while idle
```

On the very first release there is no earlier backend: leave it running with the
creation gate closed so existing owners can retrieve/delete their files. Never
roll back by deleting private state. A full NixOS rollback is not the normal app
rollback mechanism.

Operations: `journalctl -u smr-editions -u smr-editions-worker`, localhost `:8825`
health, and `release.sh status`. The public proxy is the existing `smr-server`
user service on :8799; its private paper service stays local on :8795. No new relay,
DNS record, public port, or user profile is required.
