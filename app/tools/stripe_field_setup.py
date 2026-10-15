"""Create the Stripe payment link for "make this whole-field map now" and save it in app/support.json
(field_payment), apart from the support links (tools/stripe_setup.py) so neither run touches the other's.

    STRIPE_SETUP_KEY=rk_test_... .venv/bin/python app/tools/stripe_field_setup.py   # test mode first
    STRIPE_SETUP_KEY=rk_live_... .venv/bin/python app/tools/stripe_field_setup.py   # then for real
    ... stripe_field_setup.py --price 1200                                           # another price (cents)

A restricted key with write access to Products, Prices and Payment Links; it is not saved: delete it afterwards.
The link sends the payer back to the site with the session (?paid={CHECKOUT_SESSION_ID}); the page names the map
with ?client_reference_id=<its key>; app/fieldpay.py checks payments with the read-only key (stripe_read_key).
The read-only key must be of the same mode (test or live) as the link. Run again safely: it reuses what matches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
SUPPORT = APP / "support.json"
SITE = "https://sciencemadereadable.com"
VERSION = "1"
KEY = os.environ.get("STRIPE_SETUP_KEY", "").strip()
if not KEY.startswith(("sk_", "rk_")):
    sys.exit("Set STRIPE_SETUP_KEY to a Stripe secret or restricted key (sk_... or rk_...).")
MODE = "test" if "_test_" in KEY else "live"
PRODUCT = {"name": "Make a whole-field map now",
           "description": "Moves your whole-field map on sciencemadereadable.com to the front of the line: ready in "
                          "about 20 minutes instead of waiting in the free line. The map is public for everyone. If it "
                          "can't be made, you get your money back.",
           "metadata[smr]": "field"}


def stripe(method: str, path: str, params: dict | None = None) -> dict:
    data = urllib.parse.urlencode(params or {}).encode() if method == "POST" else None
    url = f"https://api.stripe.com/v1/{path}" + ("?" + urllib.parse.urlencode(params) if method == "GET" and params else "")
    req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"Stripe refused {method} /{path}: {json.loads(e.read()).get('error', {}).get('message')}")


def every(path: str, params: dict | None = None):
    params = {"limit": 100, **(params or {})}
    while True:
        page = stripe("GET", path, params)
        yield from page["data"]
        if not page.get("has_more"):
            return
        params["starting_after"] = page["data"][-1]["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", type=int, default=1000, help="in cents (default $10: ~10x a map's AI and GPU cost)")
    a = ap.parse_args()
    product = next((p for p in every("products", {"active": "true"}) if p["metadata"].get("smr") == "field"), None) \
        or stripe("POST", "products", PRODUCT)
    price = next((p for p in every("prices", {"product": product["id"], "active": "true"})
                  if p.get("unit_amount") == a.price and p.get("currency") == "usd"), None) \
        or stripe("POST", "prices", {"product": product["id"], "currency": "usd", "unit_amount": a.price,
                                     "metadata[smr]": "field"})
    redirect = f"{SITE}/?paid={{CHECKOUT_SESSION_ID}}"
    link = None
    for l in every("payment_links", {"active": "true"}):
        if l["metadata"].get("smr") != "field":
            continue
        items = stripe("GET", f"payment_links/{l['id']}/line_items")["data"]
        same = (l["metadata"].get("smr_version") == VERSION and items and items[0]["price"]["id"] == price["id"]
                and (l.get("after_completion") or {}).get("redirect", {}).get("url") == redirect)
        if same and link is None:
            link = l
        else:
            stripe("POST", f"payment_links/{l['id']}", {"active": "false"})
    if link is None:
        link = stripe("POST", "payment_links", {
            "line_items[0][price]": price["id"], "line_items[0][quantity]": "1", "submit_type": "pay",
            "after_completion[type]": "redirect", "after_completion[redirect][url]": redirect,
            "custom_fields[0][key]": "credit", "custom_fields[0][label][type]": "custom",
            "custom_fields[0][label][custom]": "Name to show on the map (optional)",
            "custom_fields[0][type]": "text", "custom_fields[0][optional]": "true",
            "custom_text[submit][message]": "The map becomes public for everyone, like every map on the site. If it "
                                            "can't be made, or was made before your payment arrived, you get your "
                                            "money back. A name is shown after a quick check.",
            "metadata[smr]": "field", "metadata[smr_version]": VERSION})
        print(f"made the link {link['id']}")
    cfg = json.loads(SUPPORT.read_text()) if SUPPORT.exists() else {}
    cfg["field_payment"] = {"url": link["url"], "id": link["id"], "amount": a.price, "mode": MODE}
    SUPPORT.write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"{MODE} mode: {link['url']} (${a.price / 100:.2f}) saved in {SUPPORT.relative_to(APP.parent)}")
    if MODE == "test":
        print("Test card: 4242 4242 4242 4242. Test links show only on the private port; the read key must be a test "
              "key meanwhile. Then run again with the live key.")


main()
