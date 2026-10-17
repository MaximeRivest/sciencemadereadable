"""Private reading jobs. Public traffic arrives through the site's allowlisted proxy."""

from __future__ import annotations
import base64
import fcntl
from contextlib import asynccontextmanager
import hashlib
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, Response
from starlette.middleware.cors import CORSMiddleware
from store import Store, Denied, MAX_UPLOAD, MAX_JOB_BYTES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ID = re.compile(r"^[a-f0-9]{32}$")
DOI = re.compile(r'^10\.\d{4,9}/[^\s<>"?#]{1,220}$', re.I)
COOKIE_PREFIX = "smr_read_"
STAGES = {
    "extracting",
    "classifying",
    "rewriting",
    "exporting",
    "audio",
    "review",
    "waiting_model",
    "done",
}
EXTENSIONS = {
    ".html",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".epub",
    ".mp3",
    ".json",
    ".txt",
    ".zip",
}
ORIGINS = os.environ.get(
    "SMR_READING_ORIGINS",
    "https://sciencemadereadable.com,https://www.sciencemadereadable.com",
).split(",")


def csp_for_html(path):
    text = path.read_text()
    hashes = [
        "'sha256-"
        + base64.b64encode(hashlib.sha256(x.encode()).digest()).decode()
        + "'"
        for x in re.findall(r"<script[^>]*>([\s\S]*?)</script>", text, re.I)
    ]
    return (
        "default-src 'none'; script-src "
        + " ".join(hashes)
        + "; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; object-src 'none'; frame-ancestors 'none'"
    )


def time_limit(folder):
    """40 minutes for ordinary papers; long papers get time for each extra page
    read and for rewriting a longer main text (at most two hours)."""
    extra = 0
    try:
        read = json.loads((folder / "page-sort.json").read_text()).get("read", [])
        extra += 30 * max(0, min(100, len(read)) - 20)
    except (OSError, ValueError, AttributeError):
        pass
    try:
        chars = sum(
            len(v) for v in json.loads((folder / "sections.json").read_text()).values()
        )
        extra += max(0, min(160_000, chars) - 60_000) / 4 * 1.3 / 25
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return int(min(7200, 2400 + extra))


class Worker:
    def __init__(self, store):
        self.store, self.stop = store, threading.Event()
        self.active = None
        self.process = None
        self.lock = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def terminate(self):
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
            except ProcessLookupError:
                pass

    def initialize(self):
        self.lock = (self.store.root / "worker.lock").open("a")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            self.lock = None
            raise RuntimeError("A reading worker already owns this queue.")
        self.store.recover()

    def loop(self):
        if self.lock is None:
            self.initialize()
        try:
            self.run_loop()
        finally:
            self.lock.close()
            self.lock = None

    def run_loop(self):
        while not self.stop.is_set():
            self.store.heartbeat(self.active)
            self.store.cleanup(self.active)
            job = self.store.claim()
            if not job:
                self.stop.wait(2)
                continue
            self.active = job["id"]
            folder = self.store.jobs / job["id"]
            (folder / "job.json").write_text(
                json.dumps(
                    {
                        "id": job["id"],
                        "kind": job["kind"],
                        "config": job["config"],
                        "expires": job["expires"],
                    }
                )
            )
            (folder / "progress.json").unlink(missing_ok=True)
            start = time.monotonic()
            try:
                with (folder / "worker.log").open("ab") as log:
                    self.process = subprocess.Popen(
                        [sys.executable, str(HERE / "pipeline.py"), str(folder)],
                        stdout=log,
                        stderr=log,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                        close_fds=True,
                    )
                    while self.process.poll() is None:
                        self.store.heartbeat(job["id"])
                        current = self.store.get(job["id"])
                        if (
                            self.stop.is_set()
                            or current["state"] in {"deleted", "cancelled"}
                            or current["expires"] < time.time()
                        ):
                            self.terminate()
                            break
                        limit = time_limit(folder)
                        if time.monotonic() - start > limit:
                            self.terminate()
                            self.store.update(
                                job["id"],
                                state="failed",
                                error=f"Processing exceeded {limit // 60} minutes. Try a shorter paper.",
                                stage="Stopped at time limit",
                            )
                            break
                        size = sum(
                            f.stat().st_size
                            for f in folder.rglob("*")
                            if f.is_file() and not f.is_symlink()
                        )
                        if size > MAX_JOB_BYTES:
                            self.terminate()
                            self.store.update(
                                job["id"],
                                state="failed",
                                error="This paper exceeded the per-document storage limit.",
                                stage="Stopped at size limit",
                            )
                            break
                        self.read_progress(job["id"], folder)
                        self.stop.wait(1)
                    if not self.stop.is_set():
                        current = self.store.get(job["id"])
                        if current["state"] not in {"deleted", "cancelled", "failed"}:
                            self.read_progress(job["id"], folder)
                            current = self.store.get(job["id"])
                            if self.process.returncode != 0 or current["state"] not in {
                                "review",
                                "waiting_model",
                                "done",
                            }:
                                failure = folder / "failure.json"
                                message = "We could not finish this document. You can retry or delete it."
                                if failure.is_file():
                                    message = str(
                                        json.loads(failure.read_text()).get(
                                            "error", message
                                        )
                                    )[:500]
                                self.store.update(
                                    job["id"],
                                    state="failed",
                                    error=message,
                                    stage="Needs attention",
                                )
            except Exception:
                self.terminate()
                self.store.update(
                    job["id"],
                    state="failed",
                    error="The worker was interrupted. Please retry.",
                    stage="Needs attention",
                )
            finally:
                self.active, self.process = None, None
                self.store.heartbeat()
                self.store.cleanup()

    def read_progress(self, jid, folder):
        path = folder / "progress.json"
        if path.is_file():
            p = json.loads(path.read_text())
            if p.get("state") in STAGES:
                fields = {
                    "state": p["state"],
                    "stage": str(p.get("stage", ""))[:180],
                    "error": str(p.get("error", ""))[:500],
                }
                if p["state"] == "waiting_model":
                    fields["next_run"] = time.time() + 60
                self.store.update(jid, **fields)


def create_app(root=None, start_worker=None, secure_cookies=None):
    if start_worker is None:
        start_worker = os.environ.get("SMR_READING_NO_WORKER") != "1"
    if secure_cookies is None:
        secure_cookies = os.environ.get("SMR_READING_DEV") != "1"
    store = Store(root or os.environ.get("SMR_READING_STATE", "/var/lib/smr-editions"))
    worker = Worker(store)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            worker.initialize()
            worker.thread.start()
        yield
        worker.stop.set()
        worker.terminate()
        if start_worker:
            worker.thread.join(timeout=10)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store, app.state.worker = store, worker
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-SMR-Consent",
            "X-SMR-Rights",
        ],
        max_age=600,
    )

    @app.middleware("http")
    async def security(request, call_next):
        if request.method in {"POST", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and origin not in ORIGINS:
                return JSONResponse(
                    {"error": "This origin is not allowed."}, status_code=403
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive"
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
        return response

    @app.exception_handler(Denied)
    async def denied(request, error):
        return JSONResponse({"error": error.message}, status_code=error.status)

    def owner(request, jid):
        if not ID.fullmatch(jid):
            raise Denied(404, "This private job is unavailable.")
        auth = request.headers.get("authorization", "")
        token = (
            auth[7:]
            if auth.startswith("Bearer ")
            else request.cookies.get(COOKIE_PREFIX + jid, "")
        )
        return store.authorize(jid, token), token

    def address(request):
        # Only the loopback front proxy may set this header. uvicorn itself has
        # forwarded-IP processing disabled; direct internet access is not exposed.
        if request.client.host in {"127.0.0.1", "::1"}:
            return request.headers.get("x-smr-client", request.client.host)[:100]
        return request.client.host

    def accepting(request):
        gated = os.environ.get("SMR_READING_LAUNCH_GATE") == "1"
        internal = request.client.host in {
            "127.0.0.1",
            "::1",
        } and not request.headers.get("x-smr-client")
        if gated and not (store.root / "public-enabled").exists() and not internal:
            raise Denied(
                503, "Reading editions are being prepared. Please try again later."
            )

    async def small_json(request, limit=150000):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > limit:
                raise Denied(413, "Request too large.")
        try:
            value = json.loads(data)
            if not isinstance(value, dict):
                raise ValueError()
            return value
        except (ValueError, UnicodeDecodeError, RecursionError):
            raise Denied(400, "Send a JSON object.")

    @app.get("/api/editions/health")
    def health():
        release = (
            json.loads((ROOT / "RELEASE.json").read_text()).get("commit", "unknown")
            if (ROOT / "RELEASE.json").exists()
            else "development"
        )
        return {
            "ok": True,
            "worker_online": store.lease()["touched"] > time.time() - 30,
            "worker_seen": store.lease()["touched"],
            "version": 1,
            "release": release,
            "max_bytes": MAX_UPLOAD,
            "max_pages": 200,
            "retention_hours": 48,
            "formats": ["web", "epub", "pdf", "mp3"],
            "accepting_jobs": os.environ.get("SMR_READING_LAUNCH_GATE") != "1"
            or (store.root / "public-enabled").exists(),
        }

    @app.post("/api/editions/jobs")
    async def upload(request: Request):
        accepting(request)
        if (
            request.headers.get("x-smr-consent") != "typesafe"
            or request.headers.get("x-smr-rights") != "permitted"
        ):
            raise Denied(
                400,
                "Confirm processing permission and consent to sending extracted text to TypeSafe.",
            )
        if request.headers.get("content-type", "").split(";")[0] != "application/pdf":
            raise Denied(415, "Upload a PDF file, not a link or an archive.")
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            raise Denied(400, "Invalid upload length.")
        if length > MAX_UPLOAD or length < 0:
            raise Denied(413, "The PDF must be under 25 MiB.")
        jid, token = store.create(
            address(request),
            "pdf",
            {"audio": request.query_params.get("audio") == "true", "confirmed": False},
        )
        folder = store.jobs / jid
        temp = folder / "upload.part"
        size = 0
        try:
            with temp.open("wb") as f:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise Denied(413, "The PDF must be under 25 MiB.")
                    f.write(chunk)
            with temp.open("rb") as f:
                if size < 8 or f.read(5) != b"%PDF-":
                    raise Denied(415, "This file does not look like a PDF.")
            temp.replace(folder / "source.pdf")
            store.update(jid, state="queued", stage="Waiting for PDF extraction")
        except BaseException:
            store.delete(jid)
            store.cleanup()
            raise
        return {"id": jid, "token": token, "expires": store.get(jid)["expires"]}

    @app.post("/api/editions/papers")
    async def paper(request: Request):
        accepting(request)
        data = await small_json(request, 2000)
        doi = str(data.get("doi", "")).strip().lower()
        model = str(data.get("model", "our-9b"))
        if not DOI.fullmatch(doi) or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,60}", model):
            raise Denied(400, "Choose a paper and a saved rewrite from the library.")
        jid, token = store.create(
            address(request),
            "paper",
            {"doi": doi, "model": model, "audio": data.get("audio") is True},
        )
        store.update(jid, state="queued", stage="Preparing the saved paper")
        return {"id": jid, "token": token, "expires": store.get(jid)["expires"]}

    @app.get("/api/editions/jobs/{jid}")
    def status(jid: str, request: Request):
        job, _ = owner(request, jid)
        return store.view(job)

    @app.get("/api/editions/jobs/{jid}/review")
    def review(jid: str, request: Request):
        job, _ = owner(request, jid)
        if job["state"] != "review":
            raise Denied(409, "This job is not waiting for a section review.")
        data = json.loads((store.jobs / jid / "labelled.json").read_text())
        return {
            k: data[k]
            for k in ("sections", "ledger", "review", "missing_required", "accounting")
        }

    @app.post("/api/editions/jobs/{jid}/confirm")
    async def confirm(jid: str, request: Request):
        job, _ = owner(request, jid)
        if job["state"] != "review":
            raise Denied(409, "This job is not waiting for review.")
        data = await small_json(request)
        if data.get("confirmed") is not True:
            raise Denied(400, "Check the extracted sections before continuing.")
        changes = data.get("changes", {})
        if not isinstance(changes, dict) or len(changes) > 500:
            raise Denied(400, "Invalid review changes.")
        sys.path.insert(0, str(ROOT / "reading_engine"))
        from label import assemble, rewrite_size_problem, ROLES, SECTIONS

        folder = store.jobs / jid
        current = json.loads((folder / "labelled.json").read_text())
        ledger = {b["id"]: b for b in current["ledger"]}
        overrides = {}
        for bid, change in changes.items():
            if (
                bid not in ledger
                or not isinstance(change, dict)
                or not all(
                    isinstance(
                        change.get(k, "separate" if k == "boundary" else None), str
                    )
                    for k in ("role", "section", "boundary")
                )
                or change.get("role") not in ROLES
                or change.get("section") not in SECTIONS
                or change.get("boundary", "separate") not in {"join", "separate"}
            ):
                raise Denied(400, "One reviewed block has an invalid label.")
            overrides[bid] = {k: change[k] for k in ("role", "section")}
            overrides[bid].update(
                boundary=change.get(
                    "boundary", ledger[bid].get("boundary", "separate")
                ),
                reason="Confirmed by the reader against the extracted text and source page",
            )
        unchecked = [
            b["id"]
            for b in current["ledger"]
            if b["page"] in current.get("fallback_pages", []) and b["id"] not in changes
        ]
        if unchecked:
            raise Denied(
                422,
                "Check every block from the backup-OCR page(s) against the original before continuing.",
            )
        blocks = [
            {k: b[k] for k in ("id", "page", "text", "start", "end")}
            for b in current["ledger"]
        ]
        answers = {}
        for b in current["ledger"]:
            for kind in ("role", "section", "boundary"):
                if b.get(kind + "_answer"):
                    answers[b["id"] + "_" + kind] = b[kind + "_answer"]
        updated = assemble(blocks, answers, current["threshold"], overrides)
        if not updated["ready"]:
            raise Denied(
                422,
                "Some blocks still need a clear label, or a required research section is missing.",
            )
        problem = rewrite_size_problem(updated["sections"])
        if problem:
            raise Denied(422, problem)
        current.update(updated)
        from io_utils import save

        save(folder / "labelled.json", current)
        save(folder / "sections.json", current["sections"])
        config = {**job["config"], "confirmed": True}
        store.update(
            jid,
            config=config,
            state="queued",
            stage="Sections confirmed; waiting for rewrite",
        )
        return {"ok": True}

    @app.post("/api/editions/jobs/{jid}/open")
    def open_reader(jid: str, request: Request):
        job, token = owner(request, jid)
        if job["state"] != "done":
            raise Denied(409, "The reading edition is not ready yet.")
        response = JSONResponse({"url": f"/api/editions/jobs/{jid}/files/index.html"})
        response.set_cookie(
            COOKIE_PREFIX + jid,
            token,
            max_age=max(0, int(job["expires"] - time.time())),
            httponly=True,
            secure=secure_cookies,
            samesite="lax",
            path=f"/api/editions/jobs/{jid}/",
        )
        return response

    @app.post("/api/editions/jobs/{jid}/retry")
    def retry(jid: str, request: Request):
        job, _ = owner(request, jid)
        audio_retry = (
            job["state"] == "done"
            and job["config"].get("audio")
            and (store.jobs / jid / "out" / "audio-warning.json").is_file()
        )
        if (job["state"] != "failed" and not audio_retry) or job["retries"] >= 2:
            raise Denied(
                409,
                "This job cannot be retried. You can delete it and try a different PDF.",
            )
        (store.jobs / jid / "failure.json").unlink(missing_ok=True)
        store.update(
            jid, state="queued", error="", stage="Retrying", retries=job["retries"] + 1
        )
        return {"ok": True}

    @app.post("/api/editions/jobs/{jid}/cancel")
    def cancel(jid: str, request: Request):
        job, _ = owner(request, jid)
        if job["state"] not in {"done", "failed", "cancelled"}:
            store.update(jid, state="cancelled", stage="Cancelled")
        return {"ok": True}

    @app.delete("/api/editions/jobs/{jid}")
    def delete(jid: str, request: Request):
        owner(request, jid)
        store.delete(jid)
        store.cleanup(worker.active)
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE_PREFIX + jid, path=f"/api/editions/jobs/{jid}/")
        return response

    @app.get("/api/editions/jobs/{jid}/pages/{number}")
    def review_page(jid: str, number: int, request: Request):
        owner(request, jid)
        if not 1 <= number <= 200:
            raise Denied(404, "Page not found.")
        path = store.jobs / jid / "pages" / f"{number}.png"
        if not path.is_file() or path.is_symlink():
            raise Denied(404, "Page not found.")
        return FileResponse(path, media_type="image/png")

    @app.api_route(
        "/api/editions/jobs/{jid}/files/{name:path}", methods=["GET", "HEAD"]
    )
    def files(jid: str, name: str, request: Request):
        job, _ = owner(request, jid)
        if job["state"] != "done":
            raise Denied(404, "File not available.")
        relative = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or "%" in name
            or relative.is_absolute()
            or any(x in {".", ".."} for x in name.split("/"))
            or not re.fullmatch(r"[a-zA-Z0-9_./-]{1,180}", name)
        ):
            raise Denied(404, "File not available.")
        base = store.jobs / jid / "out"
        path = base / relative
        if path.suffix not in EXTENSIONS or name in {
            "print.html",
            "edition.json",
            "audio.json",
            "visuals.json",
        }:
            raise Denied(404, "File not available.")
        if name != "index.html" and path.suffix == ".html":
            raise Denied(404, "File not available.")
        if (
            base.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(base.resolve())
            or any(
                base.joinpath(*relative.parts[:i]).is_symlink()
                for i in range(1, len(relative.parts) + 1)
            )
        ):
            raise Denied(404, "File not available.")
        headers = {}
        if path.suffix == ".html":
            headers["Content-Security-Policy"] = csp_for_html(path)
        attachment = path.suffix in {".epub", ".zip", ".pdf"}
        return FileResponse(
            path,
            filename=path.name if attachment else None,
            media_type={".epub": "application/epub+zip", ".mp3": "audio/mpeg"}.get(
                path.suffix
            ),
            content_disposition_type="attachment" if attachment else "inline",
            headers=headers,
        )

    return app
