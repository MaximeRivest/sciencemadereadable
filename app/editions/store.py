"""Persistent private jobs. Tokens and addresses are never stored in clear text."""

from __future__ import annotations
import contextlib
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import time

ACTIVE = (
    "uploading",
    "queued",
    "extracting",
    "classifying",
    "rewriting",
    "exporting",
    "audio",
    "waiting_model",
)
BUSY = ("extracting", "classifying", "rewriting", "exporting", "audio")
TTL = 48 * 3600
MAX_UPLOAD = 25 * 1024 * 1024
MAX_JOB_BYTES = 900 * 1024 * 1024
MAX_TOTAL_BYTES = 12 * 1024**3


class Denied(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.jobs = self.root / "jobs"
        self.jobs.mkdir(exist_ok=True, mode=0o700)
        secret_file = self.root / "service.key"
        if not secret_file.exists():
            temporary = self.root / (".key-" + secrets.token_hex(12))
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(secrets.token_bytes(32))
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(
                    temporary, secret_file
                )  # publish complete bytes atomically; another process may win
            except FileExistsError:
                pass
            finally:
                temporary.unlink(missing_ok=True)
        self.secret = secret_file.read_bytes()
        if len(self.secret) != 32:
            raise RuntimeError("Invalid reading-service secret file.")
        self.db = self.root / "jobs.sqlite"
        with self.connect() as c:
            c.executescript("""
              PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, who TEXT NOT NULL,
                kind TEXT NOT NULL, state TEXT NOT NULL, stage TEXT NOT NULL DEFAULT '',
                created REAL NOT NULL, updated REAL NOT NULL, expires REAL NOT NULL,
                retries INTEGER NOT NULL DEFAULT 0, config TEXT NOT NULL, error TEXT NOT NULL DEFAULT '',
                next_run REAL NOT NULL DEFAULT 0
              );
              CREATE TABLE IF NOT EXISTS worker_lease (slot INTEGER PRIMARY KEY CHECK(slot=1), id TEXT, touched REAL NOT NULL);
              CREATE INDEX IF NOT EXISTS jobs_owner_created ON jobs(who, created);
              CREATE INDEX IF NOT EXISTS jobs_state_created ON jobs(state, created);
            """)

    def connect(self):
        c = sqlite3.connect(self.db, timeout=20)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=20000")
        return c

    def identity(self, address):
        return hmac.new(self.secret, address.encode(), hashlib.sha256).hexdigest()

    def create(self, address, kind, config):
        now = time.time()
        who = self.identity(address)
        if shutil.disk_usage(self.root).free < 3 * 1024**3:
            raise Denied(503, "Storage is temporarily full. Please try later.")
        if self.total_bytes() > MAX_TOTAL_BYTES:
            raise Denied(
                503, "The reading service is at its storage limit. Please try later."
            )
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            if (
                c.execute(
                    "SELECT count(*) FROM jobs WHERE who=? AND created>?",
                    (who, now - 86400),
                ).fetchone()[0]
                >= 8
            ):
                raise Denied(
                    429,
                    "Limit reached: eight reading jobs per day from this connection.",
                )
            if (
                c.execute(
                    "SELECT count(*) FROM jobs WHERE who=? AND state NOT IN ('done','failed','cancelled','deleted') AND expires>?",
                    (who, now),
                ).fetchone()[0]
                >= 2
            ):
                raise Denied(
                    429,
                    "You already have two unfinished reading jobs. Finish or delete one first.",
                )
            if c.execute(
                "SELECT count(*) FROM jobs WHERE state NOT IN ('done','failed','cancelled','deleted','review') AND expires>?",
                (now,),
            ).fetchone()[0] >= (10 if kind == "paper" else 8):
                raise Denied(503, "The reading queue is full. Please try later.")
            jid, token = secrets.token_hex(16), secrets.token_urlsafe(32)
            c.execute(
                "INSERT INTO jobs(id,token_hash,who,kind,state,created,updated,expires,config) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    jid,
                    hashlib.sha256(token.encode()).hexdigest(),
                    who,
                    kind,
                    "uploading",
                    now,
                    now,
                    now + TTL,
                    json.dumps(config),
                ),
            )
        (self.jobs / jid).mkdir(mode=0o700)
        return jid, token

    def total_bytes(self):
        return sum(
            f.stat().st_size
            for f in self.jobs.rglob("*")
            if f.is_file() and not f.is_symlink()
        )

    def get(self, jid):
        with self.connect() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["config"] = json.loads(result["config"])
        return result

    def authorize(self, jid, token):
        row = self.get(jid)
        supplied = hashlib.sha256(str(token or "").encode()).hexdigest()
        correct = row["token_hash"] if row else "0" * 64
        good = hmac.compare_digest(correct, supplied)
        if (
            not row
            or not good
            or row["state"] == "deleted"
            or row["expires"] <= time.time()
        ):
            raise Denied(
                404,
                "This private job is unavailable, expired, or its access key is missing.",
            )
        return row

    def update(self, jid, **fields):
        allowed = {"state", "stage", "error", "config", "retries", "next_run"}
        if not fields.keys() <= allowed:
            raise ValueError("Unknown job field")
        if "config" in fields:
            fields["config"] = json.dumps(fields["config"])
        fields["updated"] = time.time()
        with self.connect() as c:
            c.execute(
                "UPDATE jobs SET "
                + ",".join(f"{k}=?" for k in fields)
                + " WHERE id=? AND state NOT IN ('deleted','cancelled')",
                (*fields.values(), jid),
            )

    def heartbeat(self, active=None):
        with self.connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO worker_lease VALUES(1,?,?)",
                (active, time.time()),
            )

    def lease(self):
        with self.connect() as c:
            row = c.execute("SELECT * FROM worker_lease WHERE slot=1").fetchone()
        return dict(row) if row else {"id": None, "touched": 0}

    def recover(self):
        with self.connect() as c:
            c.execute(
                "UPDATE jobs SET state=CASE WHEN retries<2 THEN 'queued' ELSE 'failed' END, retries=retries+1, stage='Worker restarted', error=CASE WHEN retries<2 THEN '' ELSE 'Repeated worker interruptions; try a smaller paper.' END, updated=? WHERE state IN ('extracting','classifying','rewriting','exporting','audio')",
                (time.time(),),
            )
            c.execute(
                "UPDATE jobs SET state='failed', error='The upload was interrupted. Upload the PDF again.' WHERE state='uploading' AND updated<?",
                (time.time() - 180,),
            )
        self.heartbeat()

    def claim(self):
        now = time.time()
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            if (self.root / "maintenance").exists():
                return None
            row = c.execute(
                "SELECT id FROM jobs WHERE (state='queued' OR (state='waiting_model' AND next_run<=?)) AND expires>? ORDER BY created LIMIT 1",
                (now, now),
            ).fetchone()
            if not row:
                return None
            c.execute(
                "UPDATE jobs SET state='extracting', stage='Starting', updated=? WHERE id=?",
                (now, row["id"]),
            )
            c.execute(
                "INSERT OR REPLACE INTO worker_lease VALUES(1,?,?)", (row["id"], now)
            )
        return self.get(row["id"])

    def view(self, job):
        with self.connect() as c:
            ahead = c.execute(
                "SELECT count(*) FROM jobs WHERE state IN ('queued','waiting_model') AND created<? AND expires>?",
                (job["created"], time.time()),
            ).fetchone()[0]
        return {
            k: job[k]
            for k in (
                "id",
                "kind",
                "state",
                "stage",
                "created",
                "updated",
                "expires",
                "error",
                "retries",
            )
        } | {
            "ahead": ahead,
            "audio": bool(job["config"].get("audio")),
            "retention_hours": 48,
            "worker_online": self.lease()["touched"] > time.time() - 30,
        }

    def delete(self, jid):
        # Worker sees the tombstone before removing private files. Its process is
        # stopped by the controller before the directory is finally erased.
        with self.connect() as c:
            c.execute(
                "UPDATE jobs SET state='deleted', stage='', error='', config='{}', updated=? WHERE id=?",
                (time.time(), jid),
            )

    def cleanup(self, active=None):
        now = time.time()
        lease = self.lease()
        active = active or (lease["id"] if lease["touched"] > now - 120 else None)
        with self.connect() as c:
            rows = c.execute(
                "SELECT id FROM jobs WHERE expires<=? OR state='deleted'", (now,)
            ).fetchall()
            for row in rows:
                if row["id"] == active:
                    continue
                path = self.jobs / row["id"]
                if path.exists():
                    try:
                        shutil.rmtree(path)
                    except FileNotFoundError:
                        pass  # another cleanup finished the same deletion
                c.execute(
                    "UPDATE jobs SET state='deleted', config='{}', stage='', error='' WHERE id=?",
                    (row["id"],),
                )
            c.execute(
                "DELETE FROM jobs WHERE state='deleted' AND created<?",
                (now - 3 * 86400,),
            )
