"""Accounts, keys and limits through the real door (server.py's PublicHandler on a spare port), with a fake search
service behind it and fake GitHub, Google and Stripe answers (accounts._http).

    .venv/bin/python -m pytest app/test_accounts.py -q
"""
from __future__ import annotations

import base64
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accounts  # noqa: E402
import server  # noqa: E402

SITE = "https://sciencemadereadable.com"
SEEN: list[str] = []   # what the fake search service was asked (path + query, and the headers' names)


class FakeSearch(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def answer(self, body: bytes):
        SEEN.append(self.path + " " + ",".join(k.lower() for k in self.headers.keys()))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.answer(b'{"results": []}')

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.answer(json.dumps({"jsonrpc": "2.0", "id": json.loads(raw).get("id"), "result": {"content": []}}).encode())


def jwt(claims: dict) -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    return f"{enc({'alg': 'RS256'})}.{enc(claims)}.sig"


class Fake:
    """Stands for GitHub, Google and Stripe."""
    def __init__(self):
        self.github_user = {"id": 7, "login": "ada", "name": "Ada"}
        self.github_emails = [{"email": "ada@example.org", "primary": True, "verified": True}]
        self.google = {"sub": "g-1", "email": "ada@example.org", "email_verified": True, "name": "Ada L."}
        self.nonce = None
        self.sessions: dict[str, dict] = {}
        self.subs: dict[str, dict] = {}
        self.calls: list[str] = []

    def __call__(self, method, url, data=None, headers=None):
        self.calls.append(f"{method} {url.split('?')[0]}")
        if url == accounts.PROVIDERS["github"]["token"]:
            assert data["code"] == "good" and data["code_verifier"]
            return 200, {"access_token": "gho_x"}
        if url == "https://api.github.com/user":
            return 200, self.github_user
        if url == "https://api.github.com/user/emails":
            return 200, self.github_emails
        if url == accounts.PROVIDERS["google"]["token"]:
            return 200, {"id_token": jwt({**self.google, "aud": "gid", "iss": "https://accounts.google.com",
                                          "exp": time.time() + 60, "nonce": self.nonce})}
        if "api.stripe.com/v1/checkout/sessions/" in url:
            sid = url.rsplit("/", 1)[-1]
            return (200, self.sessions[sid]) if sid in self.sessions else (404, {"error": {"message": "no such"}})
        if "api.stripe.com/v1/checkout/sessions" in url:
            return 200, {"data": list(self.sessions.values()), "has_more": False}
        if "api.stripe.com/v1/subscriptions/" in url:
            return 200, self.subs[url.rsplit("/", 1)[-1]]
        if "api.stripe.com/v1/subscriptions" in url:
            return 200, {"data": list(self.subs.values()), "has_more": False}
        raise AssertionError(f"unexpected call {method} {url}")


@pytest.fixture()
def door(tmp_path, monkeypatch):
    search = ThreadingHTTPServer(("127.0.0.1", 0), FakeSearch)
    threading.Thread(target=search.serve_forever, daemon=True).start()
    monkeypatch.setattr(server, "SEARCH_API", f"http://127.0.0.1:{search.server_port}")
    accounts.reset_for_tests(tmp_path / "accounts.db")
    (tmp_path / "oauth.json").write_text(json.dumps({"github": {"client_id": "ghid", "client_secret": "ghs"},
                                                     "google": {"client_id": "gid", "client_secret": "gs"}}))
    (tmp_path / "support.json").write_text(json.dumps({"plan_payment": {
        "url": "https://buy.stripe.com/plan", "id": "plink_1", "price": "price_1", "amount": 1000, "mode": "live",
        "portal": "https://billing.stripe.com/p/login/x"}}))
    (tmp_path / "stripe_read_key").write_text("rk_live_read")
    monkeypatch.setattr(accounts, "OAUTH_FILE", tmp_path / "oauth.json")
    monkeypatch.setattr(accounts, "SUPPORT", tmp_path / "support.json")
    monkeypatch.setattr(accounts, "READ_KEY", tmp_path / "stripe_read_key")
    fake = Fake()
    monkeypatch.setattr(accounts, "_http", fake)
    server.SEARCHES.clear()
    pub = ThreadingHTTPServer(("127.0.0.1", 0), server.PublicHandler)
    threading.Thread(target=pub.serve_forever, daemon=True).start()
    SEEN.clear()
    yield f"http://127.0.0.1:{pub.server_port}", fake
    pub.shutdown(), search.shutdown()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def call(base, path, method="GET", body=None, headers=None, cookie=None):
    h = {"X-Forwarded-For": "203.0.113.5", **(headers or {})}
    if cookie:
        h["Cookie"] = cookie
    if body is not None:
        h.setdefault("Content-Type", "application/json")
        h.setdefault("Origin", SITE)
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 method=method, headers=h)
    try:
        r = OPENER.open(req, timeout=10)
    except urllib.error.HTTPError as e:
        r = e
    raw = r.read()
    try:
        out = json.loads(raw)
    except ValueError:
        out = raw.decode()
    return r.status if hasattr(r, "status") else r.code, out, r.headers


def cookies(headers) -> dict:
    out = {}
    for c in headers.get_all("Set-Cookie") or []:
        k, _, rest = c.partition("=")
        out[k] = rest.split(";")[0]
    return out


def sign_in(base, fake, provider="github") -> str:
    st, _, h = call(base, f"/api/account/login/{provider}")
    assert st == 302
    loc = urllib.parse.urlparse(h["Location"])
    p = dict(urllib.parse.parse_qsl(loc.query))
    assert p["redirect_uri"] == f"https://api.sciencemadereadable.com/api/account/callback/{provider}"
    assert p["code_challenge_method"] == "S256"
    fake.nonce = p.get("nonce")
    login = cookies(h)[accounts.LOGIN_COOKIE]
    assert "HttpOnly" in h["Set-Cookie"] and "Secure" in h["Set-Cookie"] and "SameSite=Lax" in h["Set-Cookie"]
    st, _, h = call(base, f"/api/account/callback/{provider}?code=good&state={p['state']}",
                    cookie=f"{accounts.LOGIN_COOKIE}={login}")
    assert st == 302 and h["Location"] == SITE + "/?account", h["Location"]
    return f"{accounts.SESSION_COOKIE}={cookies(h)[accounts.SESSION_COOKIE]}"


def test_signed_out_view_and_cors(door):
    base, _ = door
    st, d, h = call(base, "/api/account", headers={"Origin": SITE})
    assert st == 200 and d["signed_in"] is False
    assert {p["id"] for p in d["providers"]} == {"github", "google"}
    assert h["Access-Control-Allow-Origin"] == SITE and h["Access-Control-Allow-Credentials"] == "true"
    st, d, h = call(base, "/api/account", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in h   # another site can't read an account
    st, _, h = call(base, "/api/data/search?q=x", headers={"Origin": "https://evil.example"})
    assert h["Access-Control-Allow-Origin"] == "*"   # the open API stays open


def test_sign_in_refused_without_the_browsers_cookie(door):
    base, fake = door
    st, _, h = call(base, "/api/account/login/github")
    state = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(h["Location"]).query))["state"]
    st, _, h = call(base, f"/api/account/callback/github?code=good&state={state}")   # another browser
    assert st == 302 and h["Location"] == SITE + "/?account&signin=expired"
    assert accounts.SESSION_COOKIE not in cookies(h)
    st, _, h = call(base, "/api/account/callback/github?error=access_denied&state=x",
                    cookie=f"{accounts.LOGIN_COOKIE}=github.x.y.site")
    assert h["Location"].endswith("signin=cancelled")


def test_keys_limits_and_scrubbing(door, capsys):
    base, fake = door
    me = sign_in(base, fake)
    st, d, _ = call(base, "/api/account", cookie=me)
    assert d["signed_in"] and d["email"] == "ada@example.org" and d["plan"]["plan"] == "free"
    assert d["offer"]["url"] == "https://buy.stripe.com/plan"

    # changes only from the site, as JSON
    st, _, _ = call(base, "/api/account/keys", "POST", {"name": "x"}, cookie=me, headers={"Origin": "https://evil.example"})
    assert st == 403
    st, k, _ = call(base, "/api/account/keys", "POST", {"name": "laptop"}, cookie=me)
    assert st == 200 and accounts.KEY_RE.match(k["key"])
    key = k["key"]
    assert accounts.q1("SELECT hash FROM keys")["hash"] == accounts.fingerprint(key)   # only the fingerprint is kept
    assert key not in (Path(accounts.DB_PATH).read_bytes().decode("latin-1"))

    # the key in a header or in the address; neither reaches the search service nor the log
    st, _, _ = call(base, "/api/data/search?q=coral&k=3", headers={"Authorization": f"Bearer {key}"})
    assert st == 200
    st, _, _ = call(base, f"/api/data/search?q=coral&key={key}&k=3")
    assert st == 200
    assert all("smr_" not in s and "authorization" not in s for s in SEEN), SEEN
    assert SEEN[-1].startswith("/search?q=coral&k=3 ")
    err = capsys.readouterr().err
    assert key not in err and "smr_…" in err

    # 60 a minute for the account, from any address; the address's own 30 untouched
    for i in range(58):
        assert call(base, f"/api/data/works/W{i}", headers={"Authorization": f"Bearer {key}",
                                                              "X-Forwarded-For": f"198.51.100.{i}"})[0] == 200
    st, d, h = call(base, "/api/data/works/W1", headers={"Authorization": f"Bearer {key}"})
    assert st == 429 and "60 a minute for this key" in d["error"] and "paid plan allows 300" in d["error"]
    assert call(base, "/api/data/works/W1")[0] == 200   # no key: the address's own limit
    accounts.flush()
    assert accounts.usage(accounts.q1("SELECT id FROM users")["id"])["today"] == 60

    # a wrong or deleted key is refused, in JSON, in plain text, and as an MCP tool error
    bad = "smr_" + "x" * 32
    st, d, _ = call(base, "/api/data/search?q=x", headers={"Authorization": f"Bearer {bad}"})
    assert st == 401 and "isn't valid" in d["error"]
    st, t, _ = call(base, f"/api/data/search/why%20do%20cats%20purr?key={bad}")
    assert st == 401 and "isn't valid" in t
    rpc = {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "search_papers", "arguments": {}}}
    st, d, _ = call(base, f"/mcp?key={bad}", "POST", rpc, headers={"Accept": "application/json"})
    assert st == 200 and d["result"]["isError"] and "isn't valid" in d["result"]["content"][0]["text"]
    st, d, _ = call(base, "/mcp", "POST", {**rpc, "method": "initialize"}, headers={"Authorization": f"Bearer {bad}"})
    assert "result" in d and not d["result"].get("isError")   # connecting still works: only calls are refused
    other = {"Authorization": "Bearer some-app-token"}          # not our key: ignored, as before keys existed
    assert call(base, "/api/data/works/W2", headers={**other, "X-Forwarded-For": "192.0.2.9"})[0] == 200

    st, d, _ = call(base, "/api/account/keys/revoke", "POST", {"id": k["id"]}, cookie=me)
    assert st == 200
    time.sleep(0.01)
    st, d, _ = call(base, "/api/data/works/W1", headers={"Authorization": f"Bearer {key}"})
    assert st == 401


def test_paid_plan_through_stripe(door):
    base, fake = door
    me = sign_in(base, fake)
    uid = accounts.q1("SELECT id FROM users")["id"]
    st, k, _ = call(base, "/api/account/keys", "POST", {"name": "agent"}, cookie=me)
    fake.subs["sub_1"] = {"id": "sub_1", "status": "active", "customer": "cus_1", "livemode": True,
                          "items": {"data": [{"current_period_end": 1900000000}]}}
    fake.sessions["cs_live_abcdefghijk"] = {"id": "cs_live_abcdefghijk", "payment_link": "plink_1", "status": "complete",
                                            "client_reference_id": uid, "subscription": "sub_1", "customer": "cus_1",
                                            "livemode": True}
    st, d, _ = call(base, "/api/account/subscribed?session=cs_live_abcdefghijk")
    assert st == 200 and d["accepted"] and d["status"] == "active"
    st, d, _ = call(base, "/api/account", cookie=me)
    assert d["plan"]["plan"] == "paid" and d["plan"]["per_minute"] == 300 and d["plan"]["until"] == 1900000000
    hdr = {"Authorization": f"Bearer {k['key']}"}
    for i in range(70):   # past the free 60
        assert call(base, f"/api/data/works/W{i}", headers=hdr)[0] == 200
    # deleting is refused while a plan is paid for (the door can't cancel it: no key that writes to Stripe)
    st, d, _ = call(base, "/api/account/delete", "POST", {"confirm": "delete"}, cookie=me)
    assert st == 409
    # cancelled at Stripe: back to free at the next read
    fake.subs["sub_1"]["status"] = "canceled"
    assert accounts.refresh_subscriptions() == 1
    accounts.PLAN_CACHE.clear()
    assert accounts.plan_of(uid)["plan"] == "free"
    st, d, _ = call(base, "/api/account/subscribed?session=cs_live_nopenopenope")
    assert st == 404


def test_monthly_cap_falls_back_to_free(door):
    base, fake = door
    me = sign_in(base, fake)
    uid = accounts.q1("SELECT id FROM users")["id"]
    accounts.q("UPDATE users SET grant_until='2999-01-01' WHERE id=?", (uid,))
    accounts.q("INSERT INTO usage VALUES (?, ?, ?)", (uid, accounts._today(), 100_000))
    accounts.PLAN_CACHE.clear(), accounts.MONTH.clear()
    per_minute, plan, note = accounts.allowance(uid)
    assert (per_minute, plan) == (60, "free") and "100,000" in note


def test_google_links_to_the_same_account_and_delete(door):
    base, fake = door
    sign_in(base, fake, "github")
    me = sign_in(base, fake, "google")   # same verified email: same account
    assert accounts.q1("SELECT COUNT(*) n FROM users")["n"] == 1
    st, d, _ = call(base, "/api/account", cookie=me)
    assert [x["provider"] for x in d["logins"]] == ["github", "google"]
    fake.google = {**fake.google, "sub": "g-2", "email": "someone@else.org"}
    sign_in(base, fake, "google")
    assert accounts.q1("SELECT COUNT(*) n FROM users")["n"] == 2
    st, d, h = call(base, "/api/account/delete", "POST", {"confirm": "delete"}, cookie=me)
    assert st == 200 and "Max-Age=0" in h["Set-Cookie"]
    assert accounts.q1("SELECT COUNT(*) n FROM users")["n"] == 1
    assert accounts.q1("SELECT COUNT(*) n FROM identities")["n"] == 1   # the other person's login stays
    st, d, _ = call(base, "/api/account", cookie=me)
    assert d["signed_in"] is False


def test_paid_traffic_waits_its_turn(door, monkeypatch):
    monkeypatch.setattr(server, "PAID_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(server, "PAID_WAIT_S", 0.2)
    base, fake = door
    me = sign_in(base, fake)
    uid = accounts.q1("SELECT id FROM users")["id"]
    accounts.q("UPDATE users SET grant_until='2999-01-01' WHERE id=?", (uid,))
    accounts.PLAN_CACHE.clear()
    k = call(base, "/api/account/keys", "POST", {"name": "a"}, cookie=me)[1]["key"]
    server.PAID_SLOTS.acquire()   # all paid slots busy
    st, d, h = call(base, "/api/data/works/W1", headers={"Authorization": f"Bearer {k}"})
    assert st == 503 and h["Retry-After"] == "5"
    assert call(base, "/api/data/works/W1")[0] == 200   # the free door is not held up by paid traffic
    server.PAID_SLOTS.release()
    assert call(base, "/api/data/works/W1", headers={"Authorization": f"Bearer {k}"})[0] == 200


def test_admin_dashboard_only_for_the_admin_account(door, monkeypatch):
    base, fake = door
    monkeypatch.setattr(server.admin, "data", lambda rng, health: {"range": rng, "secret": 1})
    st, d, h = call(base, "/api/account/admin/data")                       # signed out
    assert st == 401 and "secret" not in d and ["GitHub", "/api/account/login/github?to=admin"] in d["signin"]
    assert "Access-Control-Allow-Origin" not in h and "frame-ancestors 'none'" in h["Content-Security-Policy"]
    me = sign_in(base, fake)                                                 # ada@example.org: not an admin
    st, d, _ = call(base, "/api/account/admin/data", cookie=me)
    assert st == 403 and "secret" not in d
    monkeypatch.setattr(accounts, "ADMINS", {"ada@example.org"})
    st, d, h = call(base, "/api/account/admin/data?range=7d", cookie=me, headers={"Origin": SITE})
    assert st == 200 and d == {"range": "7d", "secret": 1}
    assert "Access-Control-Allow-Origin" not in h                           # not even the site's pages may read it
    st, page, _ = call(base, "/api/account/admin/")
    assert st == 200 and "app.js" in page
    st, _, _ = call(base, "/admin/data", cookie=me)                          # the tailnet address stays closed here
    assert st == 404
    # signing in from the dashboard comes back to it
    st, _, h = call(base, "/api/account/login/github?to=admin")
    assert cookies(h)[accounts.LOGIN_COOKIE].endswith(".admin")
