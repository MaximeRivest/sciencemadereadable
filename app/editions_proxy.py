"""Bounded, allowlisted proxy for the private document service. No token in URLs."""

import http.client
import json
import os
import re
import socket
import threading
import time
from urllib.parse import urlsplit, parse_qs

ROUTE = re.compile(
    r"^/api/editions/(?:health|papers|jobs(?:/[a-f0-9]{32}(?:/(?:review|confirm|open|retry|cancel|pages/\d{1,2}|files/[a-zA-Z0-9_./-]{1,180}))?)?)$"
)
UPLOADS = threading.BoundedSemaphore(4)
STREAMS = threading.BoundedSemaphore(16)
RESPONSE_HEADERS = {
    "content-type",
    "content-length",
    "content-disposition",
    "content-range",
    "accept-ranges",
    "set-cookie",
    "content-security-policy",
    "access-control-allow-origin",
    "access-control-allow-credentials",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-max-age",
    "vary",
    "retry-after",
}


def applies(path):
    return path.split("?", 1)[0].startswith("/api/editions/")


def proxy(handler, backend=None):
    """Called before the legacy server's wildcard CORS handling."""
    u = urlsplit(handler.path)
    sent = False
    semaphore = UPLOADS if handler.command == "POST" else STREAMS
    acquired = False
    conn = None

    def fail(code, message):
        data = json.dumps({"error": message}).encode()
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", "no-store")
        origin = handler.headers.get("Origin")
        if origin in {
            "https://sciencemadereadable.com",
            "https://www.sciencemadereadable.com",
        }:
            handler.send_header("Access-Control-Allow-Origin", origin)
            handler.send_header("Access-Control-Allow-Credentials", "true")
            handler.send_header("Vary", "Origin")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        if handler.command != "HEAD":
            handler.wfile.write(data)

    try:
        if (
            not ROUTE.fullmatch(u.path)
            or ".." in u.path.split("/")
            or len(u.query) > 30
            or set(parse_qs(u.query)) - {"audio"}
        ):
            return fail(404, "Reading route not found.")
        acquired = semaphore.acquire(blocking=False)
        if not acquired:
            return fail(503, "The reading service is busy. Please retry shortly.")
        if handler.headers.get("Transfer-Encoding"):
            return fail(411, "Send a fixed-length request body.")
        values = handler.headers.get_all("Content-Length", [])
        if len(values) > 1:
            return fail(400, "Ambiguous request length.")
        length = int(values[0]) if values else 0
        limit = 25 * 1024 * 1024 if u.path == "/api/editions/jobs" else 150000
        if length < 0 or length > limit:
            return fail(413, "Request exceeds the document limit.")
        body = bytearray()
        if handler.command in {"POST", "DELETE"}:
            deadline = time.monotonic() + 90
            while len(body) < length:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return fail(408, "Upload timed out. Please try again.")
                handler.connection.settimeout(min(10, remaining))
                part = handler.rfile.read1(min(65536, length - len(body)))
                if not part:
                    return fail(400, "Upload ended before the declared length.")
                body.extend(part)
        target = urlsplit(
            backend or os.environ.get("SMR_READING_API", "http://127.0.0.1:8825")
        )
        if target.scheme != "http" or target.hostname not in {"127.0.0.1", "localhost"}:
            return fail(503, "Reading service configuration is invalid.")
        headers = {
            k: handler.headers[k]
            for k in (
                "Content-Type",
                "Authorization",
                "Cookie",
                "Origin",
                "Range",
                "X-SMR-Consent",
                "X-SMR-Rights",
                "Access-Control-Request-Method",
                "Access-Control-Request-Headers",
            )
            if handler.headers.get(k)
        }
        headers["X-SMR-Client"] = (
            handler.client()
            if hasattr(handler, "client")
            else handler.client_address[0]
        )
        conn = http.client.HTTPConnection(target.hostname, target.port, timeout=30)
        conn.request(
            handler.command,
            handler.path,
            body=bytes(body) if body else None,
            headers=headers,
        )
        response = conn.getresponse()
        handler.connection.settimeout(20)
        handler.send_response(response.status)
        for k, v in response.getheaders():
            if k.lower() in RESPONSE_HEADERS:
                handler.send_header(k, v)
        handler.send_header("Cache-Control", "no-store, private")
        handler.send_header("Referrer-Policy", "no-referrer")
        handler.send_header("X-Content-Type-Options", "nosniff")
        handler.send_header("X-Robots-Tag", "noindex, nofollow, noarchive")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True
        sent = True
        if handler.command != "HEAD":
            while chunk := response.read(65536):
                handler.wfile.write(chunk)
    except (ValueError, socket.timeout, OSError, http.client.HTTPException):
        if not sent:
            fail(
                503,
                "Reading editions are temporarily unavailable. Please retry shortly.",
            )
    finally:
        if conn:
            conn.close()
        if acquired:
            semaphore.release()
    return True
