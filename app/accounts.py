"""Accounts, API keys and the paid plan, for the public door (server.py). Why it is shaped this way: ACCOUNTS_PLAN.md.

Three levels for the data API and the MCP server:
  - no key:        30 requests a minute per address (unchanged: no sign-up is ever needed);
  - free account:  60 a minute per account, wherever its programs run (shared cloud addresses no longer collide);
  - paid:          300 a minute, 100,000 a month (past that, the free account's 60 until the month ends).

Sign-in with GitHub or Google (OAuth, with PKCE and a state cookie bound to the browser): no passwords here, no
email to send. Credentials in app/oauth.json ({"github": {"client_id", "client_secret"}, "google": {...}});
a provider without credentials is not offered. Two logins with the same verified email are one account.

Keys look like smr_<32 characters>, are shown once and kept only as their SHA-256 (a key is 192 random bits, so
a fast hash is the right one: nothing to brute-force). Sent as `Authorization: Bearer smr_...` or `?key=smr_...`
(for addresses typed in a chat, and AI apps that can't set headers). The door strips the key before passing a
request on, and from its log lines.

Paying: one Stripe subscription payment link (tools/stripe_plan_setup.py) opened with
client_reference_id=<account id>. As for whole-field maps (fieldpay.py), no key that can write to Stripe lives
here: the read-only key (stripe_read_key) reads the checkout when the payer comes back, lists the link's recent
checkouts every minute (a payer who closed the page), and every 5 minutes the plan's subscriptions (cancelled or
unpaid: back to free). Customers manage and cancel in Stripe's own customer portal (support.json plan_payment.portal).

Stored (accounts.db, SQLite): for each account its email, name, the logins (provider, their id, login name),
key fingerprints, sessions (fingerprints), Stripe subscription ids and statuses, requests counted per day.
Never what was searched. Deleting an account deletes all of it (a Stripe subscription must be cancelled first).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("SMR_ACCOUNTS_DB", HERE / "accounts.db"))
OAUTH_FILE = Path(os.environ.get("SMR_OAUTH_FILE", HERE / "oauth.json"))
SUPPORT = Path(os.environ.get("SMR_SUPPORT_FILE", HERE / "support.json"))
READ_KEY = Path(os.environ.get("SMR_STRIPE_READ_KEY", HERE / "stripe_read_key"))
SITE = os.environ.get("SMR_SITE", "https://sciencemadereadable.com")
API_BASE = os.environ.get("SMR_ACCOUNT_API", "https://api.sciencemadereadable.com")
DEV_SITE = "https://lambda.tail69222b.ts.net:18940"   # app/tools/dev.sh: tailnet only
# where a signed-in reader may be sent back to, and the pages allowed to call the account routes with the cookie
RETURN_TO = {"site": SITE, "dev": DEV_SITE}
ORIGINS = {SITE, "https://www.sciencemadereadable.com", DEV_SITE}
ORIGINS |= {o for o in os.environ.get("SMR_EXTRA_ORIGINS", "").split() if o}   # tests on another address

PLANS = {"free": {"name": "Free account", "per_minute": 60, "per_month": None},
         "paid": {"name": "Paid", "per_minute": 300, "per_month": 100_000}}
PAID_STATUSES = {"active", "trialing", "past_due"}   # past_due: Stripe is retrying the card (it cancels after)
MAX_KEYS = 10
SESSION_S = 30 * 86400
LOGIN_S = 900                                        # a sign-in must come back within 15 minutes
KEY_RE = re.compile(r"^smr_[A-Za-z0-9_-]{32}$")
KEY_IN_TEXT = re.compile(r"smr_[A-Za-z0-9_-]{6,}")   # for scrubbing logs: anything that looks like a key
SESSION_COOKIE = "smr_session"
LOGIN_COOKIE = "smr_login"

PROVIDERS = {
    "github": {"label": "GitHub", "authorize": "https://github.com/login/oauth/authorize",
               "token": "https://github.com/login/oauth/access_token", "scope": "read:user user:email"},
    "google": {"label": "Google", "authorize": "https://accounts.google.com/o/oauth2/v2/auth",
               "token": "https://oauth2.googleapis.com/token", "scope": "openid email profile"},
}


class AccountError(Exception):
    def __init__(self, status: int, message: str, code: str = "error"):
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


# ---------------------------------------------------------------- storage
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, email TEXT, name TEXT, created TEXT NOT NULL,
  grant_until TEXT, grant_note TEXT);
CREATE TABLE IF NOT EXISTS identities (provider TEXT NOT NULL, subject TEXT NOT NULL,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, email TEXT, email_verified INTEGER NOT NULL DEFAULT 0,
  login TEXT, created TEXT NOT NULL, last_login TEXT, PRIMARY KEY (provider, subject));
CREATE INDEX IF NOT EXISTS identities_email ON identities(email);
CREATE TABLE IF NOT EXISTS keys (id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  hash TEXT UNIQUE NOT NULL, hint TEXT NOT NULL, name TEXT NOT NULL, created TEXT NOT NULL, last_used TEXT, revoked TEXT);
CREATE INDEX IF NOT EXISTS keys_user ON keys(user_id);
CREATE TABLE IF NOT EXISTS sessions (hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created REAL NOT NULL, expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS subscriptions (id TEXT PRIMARY KEY, user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
  customer TEXT, status TEXT NOT NULL, period_end INTEGER, livemode INTEGER, session TEXT, updated TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS subscriptions_user ON subscriptions(user_id);
CREATE TABLE IF NOT EXISTS usage (user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, day TEXT NOT NULL,
  requests INTEGER NOT NULL, PRIMARY KEY (user_id, day));
"""
_LOCK = threading.RLock()
_CONN: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        new = not DB_PATH.exists()
        c = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None, timeout=10)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=5000")
        c.executescript(SCHEMA)
        if new:
            DB_PATH.chmod(0o600)
        _CONN = c
    return _CONN


def q(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    with _LOCK:
        return _db().execute(sql, args).fetchall()


def q1(sql: str, args: tuple = ()) -> sqlite3.Row | None:
    rows = q(sql, args)
    return rows[0] if rows else None


def reset_for_tests(path: Path) -> None:
    global _CONN, DB_PATH
    with _LOCK:
        if _CONN is not None:
            _CONN.close()
        _CONN, DB_PATH = None, path
        PENDING.clear(), MONTH.clear(), PLAN_CACHE.clear(), TOUCHED.clear()


def fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


SIGNIN_SECRET = re.compile(r"(?<=[?&])(code|state)=[^&\s\"]+")


def scrub(text: str) -> str:
    """A log line without any API key in it, nor a sign-in's one-time code."""
    text = KEY_IN_TEXT.sub("smr_…", text)
    if "/api/account/callback/" in text:
        text = SIGNIN_SECRET.sub(r"\1=…", text)
    return text


# ---------------------------------------------------------------- sign-in (OAuth 2, authorization code + PKCE)
def oauth_config() -> dict:
    try:
        cfg = json.loads(OAUTH_FILE.read_text())
    except (OSError, ValueError):
        return {}
    return {p: v for p, v in cfg.items() if p in PROVIDERS and v.get("client_id") and v.get("client_secret")}


def providers() -> list[dict]:
    return [{"id": p, "label": PROVIDERS[p]["label"]} for p in PROVIDERS if p in oauth_config()]


def redirect_uri(provider: str) -> str:
    return f"{API_BASE}/api/account/callback/{provider}"


def login_start(provider: str, to: str = "site") -> tuple[str, str]:
    """The provider's sign-in address, and the cookie that binds the answer to this browser."""
    cfg = oauth_config().get(provider)
    if not cfg:
        raise AccountError(404, "This sign-in is not available.", "unavailable")
    state, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    params = {"client_id": cfg["client_id"], "redirect_uri": redirect_uri(provider), "scope": PROVIDERS[provider]["scope"],
              "state": state, "code_challenge": challenge, "code_challenge_method": "S256"}
    if provider == "google":
        params.update(response_type="code", nonce=state, prompt="select_account")
    else:
        params["allow_signup"] = "true"
    to = to if to in RETURN_TO else "site"
    return PROVIDERS[provider]["authorize"] + "?" + urllib.parse.urlencode(params), f"{provider}.{state}.{verifier}.{to}"


def _http(method: str, url: str, data: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
    """JSON over HTTPS (replaced in the tests)."""
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Accept": "application/json", "User-Agent": "sciencemadereadable", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def _jwt_claims(token: str) -> dict:
    part = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def _identity(provider: str, code: str, verifier: str) -> dict:
    """The provider's answer → {subject, email, email_verified, name, login}."""
    cfg = oauth_config()[provider]
    form = {"client_id": cfg["client_id"], "client_secret": cfg["client_secret"], "code": code,
            "redirect_uri": redirect_uri(provider), "code_verifier": verifier}
    if provider == "google":
        form["grant_type"] = "authorization_code"
    status, tok = _http("POST", PROVIDERS[provider]["token"], form)
    if status != 200 or not (tok.get("access_token") or tok.get("id_token")):
        raise AccountError(502, f"{PROVIDERS[provider]['label']} did not confirm the sign-in.", "provider")
    if provider == "google":
        # the id token came straight from Google over TLS in exchange for our secret: its claims can be read as
        # they are (Google's OpenID guide), after checking it is ours and fresh
        c = _jwt_claims(tok.get("id_token") or "")
        if c.get("aud") != cfg["client_id"] or c.get("iss") not in ("https://accounts.google.com", "accounts.google.com") \
                or float(c.get("exp", 0)) < time.time() or not c.get("sub"):
            raise AccountError(502, "Google's answer could not be checked.", "provider")
        return {"subject": str(c["sub"]), "email": (c.get("email") or "").lower() or None,
                "email_verified": bool(c.get("email_verified")), "name": c.get("name"), "login": c.get("email"),
                "nonce": c.get("nonce")}
    auth = {"Authorization": f"Bearer {tok['access_token']}", "Accept": "application/vnd.github+json"}
    status, user = _http("GET", "https://api.github.com/user", headers=auth)
    if status != 200 or not user.get("id"):
        raise AccountError(502, "GitHub did not share who you are.", "provider")
    status, emails = _http("GET", "https://api.github.com/user/emails", headers=auth)
    emails = emails if status == 200 and isinstance(emails, list) else []
    best = next((e for e in emails if e.get("primary") and e.get("verified")), None) \
        or next((e for e in emails if e.get("verified")), None)
    return {"subject": str(user["id"]), "email": (best or {}).get("email", "").lower() or None,
            "email_verified": bool(best), "name": user.get("name") or user.get("login"), "login": user.get("login")}


def login_finish(provider: str, params: dict, cookie: str | None) -> tuple[str, str, str]:
    """The provider sent the reader back: (session token, account id, where to send them)."""
    parts = (cookie or "").split(".")
    to = parts[3] if len(parts) == 4 and parts[3] in RETURN_TO else "site"
    if params.get("error"):
        raise AccountError(400, "Sign-in was cancelled.", "cancelled")
    if len(parts) != 4 or parts[0] != provider or not params.get("state") \
            or not secrets.compare_digest(parts[1], params["state"]) or not params.get("code"):
        raise AccountError(400, "This sign-in expired or was started in another browser. Please try again.", "expired")
    if provider not in oauth_config():
        raise AccountError(404, "This sign-in is not available.", "unavailable")
    ident = _identity(provider, params["code"], parts[2])
    if provider == "google" and ident.pop("nonce", None) != parts[1]:
        raise AccountError(400, "This sign-in could not be checked. Please try again.", "expired")
    uid = upsert_user(provider, ident)
    return new_session(uid), uid, to


def upsert_user(provider: str, ident: dict) -> str:
    now = _now()
    with _LOCK:
        row = q1("SELECT user_id FROM identities WHERE provider=? AND subject=?", (provider, ident["subject"]))
        if row:
            uid = row["user_id"]
            q("UPDATE identities SET email=?, email_verified=?, login=?, last_login=? WHERE provider=? AND subject=?",
              (ident["email"], int(ident["email_verified"]), ident["login"], now, provider, ident["subject"]))
        else:
            # the same person through another provider: the same verified email on both sides
            same = q1("SELECT user_id FROM identities WHERE email=? AND email_verified=1", (ident["email"],)) \
                if ident["email"] and ident["email_verified"] else None
            if same:
                uid = same["user_id"]
            else:
                uid = "u" + secrets.token_hex(10)
                q("INSERT INTO users (id, email, name, created) VALUES (?, ?, ?, ?)",
                  (uid, ident["email"] if ident["email_verified"] else None, ident["name"], now))
            q("INSERT INTO identities (provider, subject, user_id, email, email_verified, login, created, last_login) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (provider, ident["subject"], uid, ident["email"],
                                                  int(ident["email_verified"]), ident["login"], now, now))
        if ident["email"] and ident["email_verified"]:
            q("UPDATE users SET email=? WHERE id=?", (ident["email"], uid))
        if ident.get("name"):
            q("UPDATE users SET name=COALESCE(name, ?) WHERE id=?", (ident["name"], uid))
    return uid


def new_session(uid: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _LOCK:
        q("DELETE FROM sessions WHERE expires < ?", (now,))
        q("INSERT INTO sessions (hash, user_id, created, expires) VALUES (?, ?, ?, ?)",
          (fingerprint(token), uid, now, now + SESSION_S))
    return token


def session_user(token: str | None) -> str | None:
    if not token or len(token) > 100:
        return None
    row = q1("SELECT user_id FROM sessions WHERE hash=? AND expires > ?", (fingerprint(token), time.time()))
    return row["user_id"] if row else None


def end_session(token: str | None) -> None:
    if token:
        q("DELETE FROM sessions WHERE hash=?", (fingerprint(token),))


# ---------------------------------------------------------------- keys
def make_key(uid: str, name: str) -> dict:
    name = re.sub(r"\s+", " ", str(name or "")).strip()[:60] or "My key"
    with _LOCK:
        if q1("SELECT COUNT(*) n FROM keys WHERE user_id=? AND revoked IS NULL", (uid,))["n"] >= MAX_KEYS:
            raise AccountError(409, f"An account can have {MAX_KEYS} keys. Delete one you no longer use first.", "too_many")
        key = "smr_" + secrets.token_urlsafe(24)
        kid = "k" + secrets.token_hex(6)
        q("INSERT INTO keys (id, user_id, hash, hint, name, created) VALUES (?, ?, ?, ?, ?, ?)",
          (kid, uid, fingerprint(key), key[:8] + "…" + key[-4:], name, _now()))
    return {"id": kid, "key": key, "name": name, "hint": key[:8] + "…" + key[-4:]}


def revoke_key(uid: str, kid: str) -> bool:
    with _LOCK:
        n = _db().execute("UPDATE keys SET revoked=? WHERE id=? AND user_id=? AND revoked IS NULL",
                          (_now(), kid, uid)).rowcount
    return n > 0


TOUCHED: dict[str, float] = {}


def key_account(key: str) -> dict | None:
    """The account behind an API key, or None (not a key of ours, revoked, or its account deleted)."""
    if not KEY_RE.match(key or ""):
        return None
    row = q1("SELECT id, user_id FROM keys WHERE hash=? AND revoked IS NULL", (fingerprint(key),))
    if not row:
        return None
    now = time.time()
    if now - TOUCHED.get(row["id"], 0) > 300:   # "last used", to the nearest 5 minutes
        TOUCHED[row["id"]] = now
        q("UPDATE keys SET last_used=? WHERE id=?", (_now(), row["id"]))
    per_minute, plan, note = allowance(row["user_id"])
    return {"user": row["user_id"], "key": row["id"], "plan": plan, "per_minute": per_minute, "note": note}


# ---------------------------------------------------------------- plans and usage
PLAN_CACHE: dict[str, tuple[float, dict]] = {}


def plan_of(uid: str) -> dict:
    """{"plan": "paid"|"free", "source": "stripe"|"grant"|None, "status", "until"}, cached 20 seconds."""
    hit = PLAN_CACHE.get(uid)
    if hit and time.time() - hit[0] < 20:
        return hit[1]
    sub = q1("SELECT status, period_end FROM subscriptions WHERE user_id=? ORDER BY (status IN ('active','trialing','past_due')) "
             "DESC, updated DESC LIMIT 1", (uid,))
    user = q1("SELECT grant_until FROM users WHERE id=?", (uid,))
    if sub and sub["status"] in PAID_STATUSES:
        out = {"plan": "paid", "source": "stripe", "status": sub["status"], "until": sub["period_end"]}
    elif user and user["grant_until"] and user["grant_until"] >= _today():
        out = {"plan": "paid", "source": "grant", "status": "active", "until": user["grant_until"]}
    else:
        out = {"plan": "free", "source": None, "status": sub["status"] if sub else None, "until": None}
    PLAN_CACHE[uid] = (time.time(), out)
    return out


PENDING: dict[tuple[str, str], int] = {}   # requests counted since the last write, by (account, day)
MONTH: dict[str, tuple[str, int]] = {}     # this month's requests so far, by account
_USAGE_LOCK = threading.Lock()


def month_used(uid: str) -> int:
    month = _today()[:7]
    with _USAGE_LOCK:
        hit = MONTH.get(uid)
        if hit and hit[0] == month:
            return hit[1]
    row = q1("SELECT COALESCE(SUM(requests), 0) n FROM usage WHERE user_id=? AND day LIKE ?", (uid, month + "%"))
    with _USAGE_LOCK:
        pending = sum(n for (u, d), n in PENDING.items() if u == uid and d.startswith(month))
        MONTH[uid] = (month, row["n"] + pending)
        return MONTH[uid][1]


def allowance(uid: str) -> tuple[int, str, str | None]:
    """(requests a minute, plan name, why it is less than the plan's) for an account right now."""
    plan = plan_of(uid)["plan"]
    cap = PLANS[plan]["per_month"]
    if cap and month_used(uid) >= cap:
        return PLANS["free"]["per_minute"], "free", (f"this month's {cap:,} paid requests are used: the free account's "
                                                    f"{PLANS['free']['per_minute']} a minute until the month ends")
    return PLANS[plan]["per_minute"], plan, None


def count(uid: str, cost: int = 1) -> None:
    day = _today()
    month_used(uid)
    with _USAGE_LOCK:
        PENDING[(uid, day)] = PENDING.get((uid, day), 0) + cost
        m, n = MONTH[uid]
        MONTH[uid] = (m, n + cost)


def flush() -> None:
    with _USAGE_LOCK:
        rows = list(PENDING.items())
        PENDING.clear()
    for (uid, day), n in rows:
        try:
            q("INSERT INTO usage (user_id, day, requests) VALUES (?, ?, ?) "
              "ON CONFLICT(user_id, day) DO UPDATE SET requests = requests + excluded.requests", (uid, day, n))
        except sqlite3.IntegrityError:   # the account was deleted meanwhile
            pass


def usage(uid: str) -> dict:
    today = _today()
    row = q1("SELECT requests FROM usage WHERE user_id=? AND day=?", (uid, today))
    with _USAGE_LOCK:
        pending = PENDING.get((uid, today), 0)
    return {"today": (row["requests"] if row else 0) + pending, "month": month_used(uid)}


# ---------------------------------------------------------------- the account page's view, deleting
def me(uid: str, private: bool = False) -> dict | None:
    user = q1("SELECT * FROM users WHERE id=?", (uid,))
    if not user:
        return None
    plan = plan_of(uid)
    per_minute, effective, note = allowance(uid)
    return {
        "signed_in": True, "id": uid, "email": user["email"], "name": user["name"], "created": user["created"],
        "logins": [{"provider": r["provider"], "label": PROVIDERS.get(r["provider"], {}).get("label", r["provider"]),
                    "login": r["login"], "email": r["email"]}
                   for r in q("SELECT * FROM identities WHERE user_id=? ORDER BY created", (uid,))],
        "plan": {**plan, "name": PLANS[plan["plan"]]["name"], "per_minute": per_minute, "effective": effective,
                 "per_month": PLANS[plan["plan"]]["per_month"], "note": note},
        "plans": PLANS,
        "usage": usage(uid),
        "keys": [dict(r) for r in q("SELECT id, name, hint, created, last_used FROM keys WHERE user_id=? AND revoked IS NULL "
                                     "ORDER BY created", (uid,))],
        "max_keys": MAX_KEYS,
        "offer": offer(private),
    }


def delete_user(uid: str) -> None:
    if plan_of(uid)["source"] == "stripe":
        raise AccountError(409, "Cancel your paid plan first (Manage your plan), then delete the account.", "subscribed")
    with _LOCK:
        q("DELETE FROM users WHERE id=?", (uid,))   # identities, keys, sessions and usage go with it
    PLAN_CACHE.pop(uid, None)
    with _USAGE_LOCK:
        MONTH.pop(uid, None)
        for k in [k for k in PENDING if k[0] == uid]:
            del PENDING[k]


# ---------------------------------------------------------------- Stripe (read-only)
def plan_config() -> dict | None:
    """The subscription link: {url, id, price, amount (cents), mode, portal}, from support.json (stripe_plan_setup.py)."""
    try:
        return json.loads(SUPPORT.read_text()).get("plan_payment") or None
    except (OSError, ValueError):
        return None


def offer(private: bool = False) -> dict | None:
    """What the page needs to offer the paid plan (live links only through the public door)."""
    cfg = plan_config()
    if not cfg or (cfg.get("mode") != "live" and not private):
        return None
    return {"url": cfg["url"], "amount": cfg["amount"], "currency": "usd", "mode": cfg.get("mode"),
            "portal": cfg.get("portal"), "tax": cfg.get("tax", False)}


def _stripe(path: str, params: dict | None = None) -> dict:
    key = READ_KEY.read_text().strip()
    url = f"https://api.stripe.com/v1/{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
    status, out = _http("GET", url, headers={"Authorization": f"Bearer {key}"})
    if status != 200:
        raise AccountError(502 if status != 404 else 404, (out.get("error") or {}).get("message") or f"Stripe answered {status}",
                           "stripe")
    return out


def _period_end(sub: dict) -> int | None:
    return sub.get("current_period_end") or next((i.get("current_period_end") for i in (sub.get("items") or {}).get("data") or []), None)


def _save_sub(sub_id: str, uid: str | None, customer: str | None, status: str, period_end, livemode, session=None) -> None:
    q("INSERT INTO subscriptions (id, user_id, customer, status, period_end, livemode, session, updated) VALUES (?,?,?,?,?,?,?,?) "
      "ON CONFLICT(id) DO UPDATE SET status=excluded.status, period_end=COALESCE(excluded.period_end, period_end), "
      "customer=COALESCE(excluded.customer, customer), user_id=COALESCE(subscriptions.user_id, excluded.user_id), "
      "session=COALESCE(subscriptions.session, excluded.session), updated=excluded.updated",
      (sub_id, uid, customer, status, period_end, int(bool(livemode)), session, _now()))
    if uid:
        PLAN_CACHE.pop(uid, None)


def record_checkout(s: dict) -> dict:
    """One completed checkout of the plan link → its subscription, on the account it names (idempotent)."""
    ref = s.get("client_reference_id") or ""
    uid = ref if re.match(r"^u[0-9a-f]{20}$", ref) and q1("SELECT 1 FROM users WHERE id=?", (ref,)) else None
    sub_id = s.get("subscription")
    if isinstance(sub_id, dict):
        sub_id = sub_id.get("id")
    if not sub_id:
        return {"accepted": False, "why": "this payment started no subscription"}
    status, period_end = "active", None
    try:
        sub = _stripe(f"subscriptions/{sub_id}")
        status, period_end = sub.get("status", "active"), _period_end(sub)
    except (AccountError, OSError):
        pass   # the read key may not read subscriptions yet: a paid checkout means its first month is paid
    known = q1("SELECT user_id FROM subscriptions WHERE id=?", (sub_id,))
    _save_sub(sub_id, uid, s.get("customer"), status, period_end, s.get("livemode"), s.get("id"))
    if not known:
        print(f"plan subscribed: {sub_id} for {uid or 'NO ACCOUNT (' + ref[:30] + ')'} ({status})", flush=True)
    return {"accepted": bool(uid), "user": uid, "status": status,
            "why": None if uid else "the payment named no account: write to us and we will connect it"}


def verify_checkout(session_id: str) -> tuple[int, dict]:
    """GET /api/account/subscribed?session=cs_...: the page's call when Stripe sends the payer back."""
    cfg = plan_config()
    if not cfg or not READ_KEY.exists():
        return 503, {"error": "payments are not set up"}
    if not re.match(r"^cs_(test|live)_[A-Za-z0-9]{10,200}$", session_id or ""):
        return 400, {"error": "not a checkout session"}
    try:
        s = _stripe(f"checkout/sessions/{session_id}")
    except AccountError as e:
        return e.status, {"error": "Stripe doesn't know this payment" if e.status == 404 else f"Stripe didn't answer ({e.message})"}
    except OSError as e:
        return 502, {"error": f"Stripe didn't answer ({type(e).__name__})"}
    if s.get("payment_link") != cfg.get("id"):
        return 400, {"error": "this payment was not for the paid plan"}
    if s.get("status") != "complete":
        return 402, {"error": "this payment has not gone through (yet)"}
    return 200, record_checkout(s)


def refresh_subscriptions() -> int:
    """Every subscription of the plan's price, with its status now (cancelled, unpaid → back to free)."""
    cfg = plan_config() or {}
    if not cfg.get("price"):
        return 0
    params, n = {"price": cfg["price"], "status": "all", "limit": 100}, 0
    while True:
        page = _stripe("subscriptions", params)
        for sub in page.get("data") or []:
            if q1("SELECT 1 FROM subscriptions WHERE id=?", (sub["id"],)):
                row = q1("SELECT user_id, status FROM subscriptions WHERE id=?", (sub["id"],))
                if row["status"] != sub["status"]:
                    print(f"plan subscription {sub['id']} of {row['user_id']}: {row['status']} -> {sub['status']}", flush=True)
                _save_sub(sub["id"], None, sub.get("customer"), sub["status"], _period_end(sub), sub.get("livemode"))
                n += 1
        if not page.get("has_more") or not page.get("data"):
            return n
        params["starting_after"] = page["data"][-1]["id"]


def sync_checkouts(days: int = 3) -> int:
    cfg = plan_config()
    if not cfg:
        return 0
    params, n = {"payment_link": cfg["id"], "status": "complete", "created[gte]": int(time.time()) - days * 86400,
                 "limit": 100}, 0
    while True:
        page = _stripe("checkout/sessions", params)
        for s in page.get("data") or []:
            record_checkout(s)
            n += 1
        if not page.get("has_more") or not page.get("data"):
            return n
        params["starting_after"] = page["data"][-1]["id"]


def background_forever() -> None:
    """Usage counts written every 15 s; the plan link's checkouts every minute; its subscriptions every 5."""
    last_checkouts = last_subs = 0.0
    while True:
        time.sleep(15)
        try:
            flush()
        except Exception as e:  # noqa: BLE001
            print(f"account usage not written: {e}", flush=True)
        if not plan_config() or not READ_KEY.exists():
            continue
        now = time.time()
        try:
            if now - last_checkouts >= 60:
                last_checkouts = now
                sync_checkouts()
            if now - last_subs >= 300:
                last_subs = now
                refresh_subscriptions()
        except Exception as e:  # noqa: BLE001  (network, a key without the right access: try again later)
            print(f"plan payments not read: {e}", flush=True)
