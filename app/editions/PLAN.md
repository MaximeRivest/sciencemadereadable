# Production reading editions

Authorized: build, validate and release the full reading pipeline. Existing public
site must remain usable during development. No unrelated files are included.

## Release gates

- [x] Shared rendering engine, not a second copy of prototype logic.
- [x] Persistent bounded queue; one document worker; restart, retry and cancellation.
- [x] Private uploads: unguessable owner capability, no public listing, short retention,
      delete control, explicit TypeSafe consent, size/page/work/disk limits.
- [x] Untrusted PDF parsing and print/export run without network, keys or other jobs
      in a filesystem/process sandbox. Never parse an upload in the public API.
- [x] PDF section review with original pages; uncertain labels cannot silently pass.
- [x] Original visuals: structured publisher assets for existing papers; conservative
      original-page fallback for uncertain PDF crops, clearly labelled (not guessed).
- [x] Main website entry points for uploads and saved-paper reading editions.
- [x] Browser, API/auth, traversal/XSS, expiry/deletion, restart and real-job tests.
- [x] Declarative service, reproducible release, health checks, rollback instructions.
- [x] Commit/push; stage and live browser checks; release only after gates pass.

No promise of perfect extraction, fully checked scientific accuracy, or universal
PDF structure. Unsupported inputs must fail clearly. The first public release is
bounded to English research papers (20 pages / 25 MiB) with a reader review step.
