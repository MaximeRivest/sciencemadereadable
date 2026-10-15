"""Paying to make a whole-field map now (the search service's whole-field maps: search/srl_search/field.py).

Whole-field maps wait in a free line (about 10 a day: the AI that judges papers costs about $1 a map). A reader can
pay (one Stripe payment link, made by tools/stripe_field_setup.py) to move their map to the paid line, which goes
first: ready in about 20 minutes. The map is public for everyone either way.

How a payment reaches the line, without any key that can write to Stripe on this machine:
  1. the page sends the reader to the payment link with ?client_reference_id=<the map's key> (24 hex characters);
  2. Stripe sends them back to /?paid=<checkout session id>; the page asks GET /api/fields/paid?session=...;
  3. this module reads that session with the read-only key (stripe_read_key: Checkout Sessions read), checks it
     was paid through the field link, and writes FIELDS/paid/<key>.json, which the search service's worker reads;
  4. every minute it also lists the link's recent payments, for readers who closed the page before coming back.
A payment counts only if the map was waiting when it arrived; otherwise (already being made, made, unknown, or
paid twice) it is owed back, as is a paid map that cannot be made. Refunds go out by themselves when
stripe_refund_key exists (a restricted key: Refunds write, Checkout Sessions read); else
tools/field_payments.py lists them. A name to show on the map is kept until approved there.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = Path(os.environ.get("SRL_FIELDS", "/mnt/fast/scholarsreadinglist/search/fields"))
PAID = FIELDS / "paid"
READ_KEY = HERE / "stripe_read_key"
REFUND_KEY = HERE / "stripe_refund_key"
SUPPORT = HERE / "support.json"
ASKER_SECRET = HERE / "asker_secret"
KEY_RE = re.compile(r"^[0-9a-f]{24}$")
LOCK = threading.Lock()


def config() -> dict | None:
    """The field payment link: {url, id, amount (cents), mode}, from support.json (tools/stripe_field_setup.py)."""
    try:
        return json.loads(SUPPORT.read_text()).get("field_payment") or None
    except (OSError, ValueError):
        return None


def asker(who: str) -> str:
    """An anonymous code for one reader (the door's per-address id), stable while their maps wait: the search
    service counts a reader's waiting free maps with it. The address itself is never stored."""
    if not ASKER_SECRET.exists():
        ASKER_SECRET.write_text(secrets.token_hex(32))
        ASKER_SECRET.chmod(0o600)
    return hashlib.sha256((ASKER_SECRET.read_text().strip() + who).encode()).hexdigest()[:16]


def _stripe(key: str, method: str, path: str, params: dict | None = None) -> dict:
    data = urllib.parse.urlencode(params or {}).encode() if method == "POST" else None
    url = f"https://api.stripe.com/v1/{path}" + ("?" + urllib.parse.urlencode(params) if method == "GET" and params else "")
    req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, p)


def record(s: dict) -> dict:
    """One completed checkout session of the field link → its map's marker (idempotent). Returns the marker."""
    key = s.get("client_reference_id") or ""
    name = next(((f.get("text") or {}).get("value") for f in s.get("custom_fields") or [] if f.get("key") == "credit"), None)
    entry = {"session": s["id"], "payment_intent": s.get("payment_intent"), "amount": s.get("amount_total"),
             "currency": s.get("currency"), "livemode": s.get("livemode"),
             "paid_at": datetime.fromtimestamp(s["created"], timezone.utc).isoformat(timespec="seconds"),
             "recorded": _now(), "name": (name or "").strip()[:60] or None, "name_ok": False}
    with LOCK:
        if not KEY_RE.match(key):
            path = PAID / f"unknown-{s['id']}.json"
            if not path.exists():
                _write(path, {**entry, "key": key, "accepted": False, "refund": "due",
                              "why": "the payment named no map"})
            return _read(path)
        path = PAID / f"{key}.json"
        old = _read(path)
        if old and old.get("session") == s["id"]:
            return old
        if old:                                   # the same map paid twice: the second is owed back
            extra = PAID / f"extra-{s['id']}.json"
            if not extra.exists():
                _write(extra, {**entry, "key": key, "accepted": False, "refund": "due", "why": "this map was already paid for"})
            return old
        job = _read(FIELDS / key / "job.json") or {}
        ok = job.get("status") == "queued"
        why = None if ok else {"running": "the map was already being made", "done": "the map was already made",
                               "failed": "the map could not be made"}.get(job.get("status"), "no such map was waiting")
        marker = {**entry, "key": key, "topic": (job.get("request") or {}).get("q"), "accepted": ok,
                  "refund": None if ok else "due", "why": why}
        _write(path, marker)
        if ok:   # wake the worker's plan: the queue file's change starts it if it was idle
            q = FIELDS / "queue" / f"{key}.job"
            if q.exists():
                q.touch()
        print(f"field map paid: {key} {marker['topic']!r} ({'moved to the paid line' if ok else 'owed back: ' + why})",
              flush=True)
        return marker


def verify(session_id: str) -> tuple[int, dict]:
    """GET /api/fields/paid?session=cs_...: the page's call when Stripe sends the reader back."""
    cfg = config()
    if not cfg or not READ_KEY.exists():
        return 503, {"error": "payments are not set up"}
    if not re.match(r"^cs_(test|live)_[A-Za-z0-9]{10,200}$", session_id or ""):
        return 400, {"error": "not a checkout session"}
    try:
        s = _stripe(READ_KEY.read_text().strip(), "GET", f"checkout/sessions/{session_id}")
    except urllib.error.HTTPError as e:
        return (404 if e.code == 404 else 502), {"error": "Stripe doesn't know this payment" if e.code == 404
                                                  else f"Stripe didn't answer ({e.code})"}
    except OSError as e:
        return 502, {"error": f"Stripe didn't answer ({type(e).__name__})"}
    if s.get("payment_link") != cfg.get("id"):
        return 400, {"error": "this payment was not for a whole-field map"}
    if s.get("status") != "complete" or s.get("payment_status") != "paid":
        return 402, {"error": "this payment has not gone through (yet)"}
    m = record(s)
    job = _read(FIELDS / (m.get("key") or "-") / "job.json") or {}
    req = job.get("request") or {}
    return 200, {"key": m.get("key"), "topic": req.get("q"), "definition": req.get("definition"),
                 "accepted": m.get("accepted"), "why": m.get("why"), "status": job.get("status"),
                 "name": m.get("name")}


def owed() -> list[tuple[Path, dict]]:
    """Payments to give back: refused at arrival, or paid maps that could not be made (the topic itself, or twice
    a failure of ours: the worker tries a paid map once more)."""
    out = []
    for p in sorted(PAID.glob("*.json")):
        m = _read(p) or {}
        if m.get("refund") in ("done", "failed"):
            continue
        if m.get("accepted") and not m.get("refund"):
            job = _read(FIELDS / m["key"] / "job.json") or {}
            if job.get("status") == "failed" and (job.get("final") or job.get("retried")):
                m.update({"refund": "due", "why": "the map could not be made: " + (job.get("error") or "")})
                _write(p, m)
        if m.get("refund") == "due":
            out.append((p, m))
    return out


def refund_due() -> None:
    if not REFUND_KEY.exists():
        return
    key = REFUND_KEY.read_text().strip()
    for p, m in owed():
        try:
            r = _stripe(key, "POST", "refunds", {"payment_intent": m["payment_intent"],
                                                 "metadata[smr]": "field", "metadata[why]": (m.get("why") or "")[:400]})
            m.update({"refund": "done", "refund_id": r.get("id"), "refunded": _now()})
            print(f"field map payment refunded: {m.get('key')} ({m.get('why')})", flush=True)
        except (urllib.error.HTTPError, OSError) as e:
            m.update({"refund": "failed", "refund_error": str(e)})
            print(f"field map refund FAILED, do it by hand: {m.get('session')} ({e})", flush=True)
        _write(p, m)


def poll_forever() -> None:
    """Every minute: the field link's payments of the last 3 days (a reader may have closed the page before Stripe
    sent them back), then refunds owed."""
    while True:
        cfg = config()
        if cfg and READ_KEY.exists():
            try:
                params = {"payment_link": cfg["id"], "status": "complete", "created[gte]": int(time.time()) - 3 * 86400,
                          "limit": 100}
                for s in _stripe(READ_KEY.read_text().strip(), "GET", "checkout/sessions", params)["data"]:
                    if s.get("payment_status") == "paid":
                        record(s)
                refund_due()
            except Exception as e:  # noqa: BLE001  (network, a key without the right access: try again later)
                print(f"field payments not read: {e}", flush=True)
        time.sleep(60)


def offer(private: bool = False) -> dict | None:
    """What the page needs to show the offer: the link and the price (live links only through the public door)."""
    cfg = config()
    if not cfg or (cfg.get("mode") != "live" and not private):
        return None
    return {"url": cfg["url"], "amount": cfg["amount"], "currency": "usd", "mode": cfg.get("mode")}
