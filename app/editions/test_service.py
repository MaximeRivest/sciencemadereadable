import hashlib
import json
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient
from service import create_app
from store import Store, Denied


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "state", start_worker=False, secure_cookies=True)
    with TestClient(app, base_url="https://api.sciencemadereadable.com") as c:
        yield c


def new(client, audio=False):
    r = client.post(
        "/api/editions/jobs?audio=" + str(audio).lower(),
        content=b"%PDF-1.7\nfixture",
        headers={
            "Content-Type": "application/pdf",
            "X-SMR-Consent": "typesafe",
            "X-SMR-Rights": "permitted",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def auth(j):
    return {"Authorization": "Bearer " + j["token"]}


def test_consent_content_type_and_size(client):
    assert client.post("/api/editions/jobs", content=b"%PDF-1.7").status_code == 400
    headers = {
        "X-SMR-Consent": "typesafe",
        "X-SMR-Rights": "permitted",
        "Content-Type": "application/zip",
    }
    assert (
        client.post("/api/editions/jobs", content=b"zip", headers=headers).status_code
        == 415
    )
    headers["Content-Type"] = "application/pdf"
    assert (
        client.post(
            "/api/editions/jobs", content=b"not a pdf", headers=headers
        ).status_code
        == 415
    )
    headers["Content-Length"] = str(26 * 1024 * 1024)
    assert (
        client.post("/api/editions/jobs", content=b"", headers=headers).status_code
        == 413
    )


def test_private_capability_and_no_listing(client):
    j = new(client)
    other = new(client)
    url = "/api/editions/jobs/" + j["id"]
    assert client.get(url).status_code == 404
    assert client.get(url, headers=auth(other)).status_code == 404
    assert client.get(url, headers=auth(j)).json()["state"] == "queued"
    assert client.get("/api/editions/jobs").status_code in (404, 405)
    row = client.app.state.store.get(j["id"])
    assert row["token_hash"] != j["token"]
    assert row["token_hash"] == hashlib.sha256(j["token"].encode()).hexdigest()
    assert row["who"] != "testclient"


def test_open_cookie_csp_and_range(client):
    j = new(client)
    store = client.app.state.store
    out = store.jobs / j["id"] / "out"
    out.mkdir()
    (out / "index.html").write_text("<html><script>window.test=1</script></html>")
    (out / "paper.mp3").write_bytes(b"1234567890")
    assert (
        client.post(f"/api/editions/jobs/{j['id']}/open", headers=auth(j)).status_code
        == 409
    )
    store.update(j["id"], state="done")
    r = client.post(f"/api/editions/jobs/{j['id']}/open", headers=auth(j))
    cookie = r.headers["set-cookie"]
    assert "HttpOnly" in cookie and "Secure" in cookie and "SameSite=lax" in cookie
    r = client.get(r.json()["url"])
    assert r.status_code == 200
    assert "script-src 'sha256-" in r.headers["content-security-policy"]
    assert "no-store" in r.headers["cache-control"]
    assert "noindex" in r.headers["x-robots-tag"]
    r = client.get(
        f"/api/editions/jobs/{j['id']}/files/paper.mp3", headers={"Range": "bytes=2-5"}
    )
    assert r.status_code == 206 and r.content == b"3456"


def test_traversal_symlink_and_internal_files(client, tmp_path):
    j = new(client)
    store = client.app.state.store
    out = store.jobs / j["id"] / "out"
    out.mkdir()
    (out / "edition.json").write_text('{"private":"internal"}')
    (out / "print.html").write_text("not a served page")
    secret = tmp_path / "secret.txt"
    secret.write_text("do not serve")
    (out / "leak.txt").symlink_to(secret)
    store.update(j["id"], state="done")
    for name in (
        "../source.pdf",
        "%2e%2e/source.pdf",
        "leak.txt",
        "edition.json",
        "print.html",
        "a\\b.txt",
    ):
        assert (
            client.get(
                f"/api/editions/jobs/{j['id']}/files/{name}", headers=auth(j)
            ).status_code
            == 404
        )


def test_delete_removes_files_and_revokes_access(client):
    j = new(client)
    store = client.app.state.store
    folder = store.jobs / j["id"]
    assert folder.exists()
    assert (
        client.delete("/api/editions/jobs/" + j["id"], headers=auth(j)).status_code
        == 200
    )
    assert not folder.exists()
    assert (
        client.get("/api/editions/jobs/" + j["id"], headers=auth(j)).status_code == 404
    )
    assert store.get(j["id"])["config"] == {}


def test_expiry_and_restart(client):
    j = new(client)
    store = client.app.state.store
    store.update(j["id"], state="rewriting")
    store.recover()
    assert store.get(j["id"])["state"] == "queued"
    with store.connect() as db:
        db.execute("UPDATE jobs SET expires=0 WHERE id=?", (j["id"],))
    assert (
        client.get("/api/editions/jobs/" + j["id"], headers=auth(j)).status_code == 404
    )
    store.cleanup()
    assert not (store.jobs / j["id"]).exists()


def test_limits_cancel_and_retry(client):
    j = new(client)
    new(client)
    r = client.post(
        "/api/editions/papers", json={"doi": "10.3390/test", "model": "our-9b"}
    )
    assert r.status_code == 429
    assert (
        client.post(f"/api/editions/jobs/{j['id']}/retry", headers=auth(j)).status_code
        == 409
    )
    client.post(f"/api/editions/jobs/{j['id']}/cancel", headers=auth(j))
    client.app.state.store.update(j["id"], state="rewriting")
    assert client.app.state.store.get(j["id"])["state"] == "cancelled"


def test_cors_rejects_untrusted_origin(client):
    response = client.post(
        "/api/editions/papers",
        json={"doi": "10.3390/test"},
        headers={"Origin": "https://evil.invalid"},
    )
    assert response.status_code == 403
    good = client.options(
        "/api/editions/jobs",
        headers={
            "Origin": "https://sciencemadereadable.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type,x-smr-consent,x-smr-rights",
        },
    )
    assert good.status_code == 200
    assert (
        good.headers["access-control-allow-origin"] == "https://sciencemadereadable.com"
    )
    assert good.headers["access-control-allow-credentials"] == "true"


def test_active_worker_lease_defers_file_removal(client):
    j = new(client)
    store = client.app.state.store
    claimed = store.claim()
    assert claimed["id"] == j["id"]
    client.delete("/api/editions/jobs/" + j["id"], headers=auth(j))
    assert (store.jobs / j["id"]).exists()
    store.heartbeat()
    store.cleanup()
    assert not (store.jobs / j["id"]).exists()


def test_maintenance_and_bounded_restart_recovery(client):
    j = new(client)
    store = client.app.state.store
    marker = store.root / "maintenance"
    marker.touch()
    assert store.claim() is None
    marker.unlink()
    for _ in range(3):
        assert store.claim() is not None
        store.recover()
    assert store.get(j["id"])["state"] == "failed"
    assert store.claim() is None


def test_launch_gate_blocks_public_creation(client, monkeypatch):
    monkeypatch.setenv("SMR_READING_LAUNCH_GATE", "1")
    assert (
        client.post(
            "/api/editions/papers",
            json={"doi": "10.3390/test"},
            headers={"X-SMR-Client": "public-visitor"},
        ).status_code
        == 503
    )
    (client.app.state.store.root / "public-enabled").touch()
    assert (
        client.post(
            "/api/editions/papers",
            json={"doi": "10.3390/test"},
            headers={"X-SMR-Client": "public-visitor"},
        ).status_code
        == 200
    )


def test_bad_json_and_doi(client):
    assert client.post("/api/editions/papers", content="not json").status_code == 400
    for doi in ("http://127.0.0.1:8015", "10.1234/../../bad?x=1", ""):
        assert client.post("/api/editions/papers", json={"doi": doi}).status_code == 400


def test_review_requires_owner_and_resolved_blocks(client):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reading_engine"))
    from label import assemble

    j = new(client)
    store = client.app.state.store
    folder = store.jobs / j["id"]
    blocks, answers = [], {}
    for i, (role, section, text) in enumerate(
        [
            ("title", "title", "Title"),
            ("paragraph", "introduction", "Introduction"),
            ("paragraph", "methods", "Methods"),
            ("paragraph", "results", "Results"),
        ]
    ):
        bid = f"b{i:04d}"
        blocks.append(
            {
                "id": bid,
                "page": 1,
                "start": i * 20,
                "end": i * 20 + len(text),
                "text": text,
            }
        )
        for kind, choice in [("role", role), ("section", section)]:
            answers[bid + "_" + kind] = {
                "choice": choice,
                "confidence": 0.2 if i == 2 else 1,
                "probabilities": {choice: 1},
            }
    data = assemble(blocks, answers)
    data["accounting"] = {}
    (folder / "labelled.json").write_text(json.dumps(data))
    store.update(j["id"], state="review")
    url = f"/api/editions/jobs/{j['id']}"
    assert client.get(url + "/review").status_code == 404
    assert client.get(url + "/review", headers=auth(j)).status_code == 200
    assert (
        client.post(
            url + "/confirm", headers=auth(j), json={"confirmed": True}
        ).status_code
        == 422
    )
    assert (
        client.post(
            url + "/confirm",
            headers=auth(j),
            json={
                "confirmed": True,
                "changes": {"b0002": {"role": "paragraph", "section": "methods"}},
            },
        ).status_code
        == 200
    )
    assert store.get(j["id"])["config"]["confirmed"] is True
    assert json.loads((folder / "labelled.json").read_text())["ready"] is True
