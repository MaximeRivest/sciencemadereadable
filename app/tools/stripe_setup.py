"""Create the Stripe payment links the support window uses, and save them in app/support.json.

    STRIPE_SETUP_KEY=rk_test_... .venv/bin/python app/tools/stripe_setup.py     # test mode first
    STRIPE_SETUP_KEY=rk_live_... .venv/bin/python app/tools/stripe_setup.py     # then for real

The key is read from the environment and never saved. A restricted key is enough: Products, Prices and
Payment Links with write access (Dashboard → Developers → API keys → Create restricted key). Delete it
afterwards: the queue only needs a read-only key (app/stripe_read_key, see app/README.md).

What it makes, in US dollars (the GPU is billed in dollars), all on one product, "Support Science made
readable": one link per amount the window offers ($5, $10, $25, $50), one where the supporter chooses
(from $2), $3 a month, and "sponsor a day" ($110, asking for the name to show on the site and the day).
After paying, Stripe sends the supporter back to the site, which says thank you (?thanks).
It finds what an earlier run made (metadata smr=...), so it can run again safely: it only adds what is
missing, and it makes new links if the site address or a text below changed.
"""
from __future__ import annotations

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
KEY = os.environ.get("STRIPE_SETUP_KEY", "").strip()
if not KEY.startswith(("sk_", "rk_")):
    sys.exit("Set STRIPE_SETUP_KEY to a Stripe secret or restricted key (sk_... or rk_...).")
MODE = "test" if "_test_" in KEY else "live"

PRODUCT = {"name": "Support Science made readable",
           "description": "Keeps the GPU running that rewrites research papers in plain words on sciencemadereadable.com. "
                          "Free for everyone; supported by its readers. Not a donation to a charity.",
           "metadata[smr]": "support"}
LINKS = {   # key in support.json → the price, and what the link asks
    "5": {"unit_amount": 500}, "10": {"unit_amount": 1000}, "25": {"unit_amount": 2500}, "50": {"unit_amount": 5000},
    "custom": {"custom_unit_amount[enabled]": "true", "custom_unit_amount[preset]": "1000", "custom_unit_amount[minimum]": "200"},
    "monthly": {"unit_amount": 300, "recurring[interval]": "month"},
    "day": {"unit_amount": 11000},
}
DAY_FIELDS = [   # what a day's sponsor tells us (Stripe allows 3 fields)
    ("name", "Name to show on the site", False),
    ("website", "Link for the name (optional)", True),
    ("day", "Preferred day, e.g. 2026-10-20 (optional)", True),
]


def stripe(method: str, path: str, params: dict | None = None) -> dict:
    data = urllib.parse.urlencode(params or {}).encode() if method == "POST" else None
    url = f"https://api.stripe.com/v1/{path}" + ("?" + urllib.parse.urlencode(params) if method == "GET" and params else "")
    req = urllib.request.Request(url, data=data, method=method, headers={"Authorization": f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        err = json.loads(e.read()).get("error", {})
        sys.exit(f"Stripe refused {method} /{path}: {err.get('message')}\n"
                 "(a restricted key needs write access to Products, Prices and Payment Links)")


def every(path: str, params: dict | None = None):
    params = {"limit": 100, **(params or {})}
    while True:
        page = stripe("GET", path, params)
        yield from page["data"]
        if not page.get("has_more"):
            return
        params["starting_after"] = page["data"][-1]["id"]


def link_params(name: str, price: str) -> dict:
    p = {"line_items[0][price]": price, "line_items[0][quantity]": "1",
         "after_completion[type]": "redirect", "after_completion[redirect][url]": f"{SITE}/?thanks",
         "custom_text[submit][message]": "Science made readable is free for everyone and kept running by its readers. "
                                         "This is support for the project, not a donation to a charity.",
         "metadata[smr]": f"support-{name}", "metadata[smr_version]": "1"}
    if name != "monthly":
        p["submit_type"] = "pay"
    if name == "day":
        for i, (key, label, optional) in enumerate(DAY_FIELDS):
            p |= {f"custom_fields[{i}][key]": key, f"custom_fields[{i}][label][type]": "custom",
                  f"custom_fields[{i}][label][custom]": label, f"custom_fields[{i}][type]": "text",
                  f"custom_fields[{i}][optional]": str(optional).lower()}
    return p


def main():
    product = next((p for p in every("products", {"active": "true"}) if p["metadata"].get("smr") == "support"), None)
    if not product:
        product = stripe("POST", "products", PRODUCT)
        print(f"made the product {product['id']}")
    prices = {p["metadata"].get("smr"): p for p in every("prices", {"product": product["id"], "active": "true"})}
    links = {l["metadata"].get("smr"): l for l in every("payment_links", {"active": "true"})}
    out = {}
    for name, spec in LINKS.items():
        price = prices.get(f"support-{name}")
        if not price:
            price = stripe("POST", "prices", {"product": product["id"], "currency": "usd",
                                               "metadata[smr]": f"support-{name}", **spec})
            print(f"made the price for {name}")
        link = links.get(f"support-{name}")
        if link and (link["line_items"]["data"][0]["price"]["id"] if "line_items" in link else None) not in (None, price["id"]):
            link = None
        if not link or link["metadata"].get("smr_version") != "1" \
                or link.get("after_completion", {}).get("redirect", {}).get("url") != f"{SITE}/?thanks":
            if link:
                stripe("POST", f"payment_links/{link['id']}", {"active": "false"})
            link = stripe("POST", "payment_links", link_params(name, price["id"]))
            print(f"made the link for {name}")
        out[name] = link["url"]
    cfg = json.loads(SUPPORT.read_text()) if SUPPORT.exists() else {}
    cfg["stripe"] = out
    cfg["stripe_mode"] = MODE
    SUPPORT.write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"\n{MODE} mode: {len(out)} links saved in {SUPPORT.relative_to(APP.parent)}")
    for k, v in out.items():
        print(f"  {k:8} {v}")
    if MODE == "test":
        print("\nTest card: 4242 4242 4242 4242, any future date, any CVC. Then run again with the live key.")


main()
