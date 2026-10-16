import io
from email.message import Message
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import editions_proxy as proxy


class Handler:
    def __init__(self, path, method="GET", headers=None, body=b""):
        self.path, self.command = path, method
        self.headers = Message()
        for k, v in (headers or {}).items():
            self.headers[k] = v
        self.rfile = io.BufferedReader(io.BytesIO(body))
        self.wfile = io.BytesIO()
        self.connection = type("Connection", (), {"settimeout": lambda *a: None})()
        self.status = None
        self.out_headers = {}
        self.client_address = ("untrusted-address", 1)

    def client(self):
        return "trusted-address-hash"

    def send_response(self, status):
        self.status = status

    def send_header(self, k, v):
        self.out_headers[k.lower()] = v

    def end_headers(self):
        pass


def test_allowlist_and_oversize_never_contact_upstream():
    for path, headers, status in [
        ("/api/editions/jobs/../../secret", {}, 404),
        ("/api/editions/jobs", {"Content-Length": str(26 * 1024 * 1024)}, 413),
        ("/api/editions/jobs", {"Transfer-Encoding": "chunked"}, 411),
    ]:
        h = Handler(path, "POST", headers)
        with patch.object(proxy.http.client, "HTTPConnection") as connect:
            proxy.proxy(h)
            assert h.status == status
            connect.assert_not_called()


def test_proxy_preserves_private_headers_but_overwrites_client_identity():
    h = Handler(
        "/api/editions/jobs/" + "a" * 32 + "/files/paper.mp3",
        headers={
            "Authorization": "Bearer private",
            "Cookie": "private=cookie",
            "Range": "bytes=0-9",
            "X-SMR-Client": "spoof",
        },
    )

    class Upstream:
        status = 206

        def getheaders(self):
            return [
                ("Content-Length", "10"),
                ("Content-Type", "audio/mpeg"),
                ("Content-Range", "bytes 0-9/100"),
                ("Access-Control-Allow-Origin", "https://sciencemadereadable.com"),
                ("Set-Cookie", "private=issued; HttpOnly"),
            ]

        def read(self, n):
            value = getattr(self, "body", b"0123456789")
            self.body = b""
            return value

    with patch.object(proxy.http.client, "HTTPConnection") as connect:
        connect.return_value.getresponse.return_value = Upstream()
        proxy.proxy(h)
        sent = connect.return_value.request.call_args.kwargs["headers"]
        assert sent["X-SMR-Client"] == "trusted-address-hash"
        assert (
            sent["Authorization"] == "Bearer private"
            and sent["Cookie"] == "private=cookie"
        )
        assert h.status == 206 and h.wfile.getvalue() == b"0123456789"
        assert h.out_headers["content-range"] == "bytes 0-9/100"
        assert "HttpOnly" in h.out_headers["set-cookie"]
        assert h.out_headers["cache-control"] == "no-store, private"


def test_failure_has_safe_cors_and_no_internal_address():
    h = Handler(
        "/api/editions/health", headers={"Origin": "https://sciencemadereadable.com"}
    )
    with patch.object(
        proxy.http.client,
        "HTTPConnection",
        side_effect=OSError("private internal diagnostic"),
    ):
        proxy.proxy(h)
    assert h.status == 503
    assert b"private internal diagnostic" not in h.wfile.getvalue()
    assert (
        h.out_headers["access-control-allow-origin"]
        == "https://sciencemadereadable.com"
    )
