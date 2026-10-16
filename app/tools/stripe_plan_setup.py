"""Create the Stripe subscription link for the API's paid plan and save it in app/support.json (plan_payment),
apart from the support and whole-field links (tools/stripe_setup.py, tools/stripe_field_setup.py).

    STRIPE_SETUP_KEY=rk_test_... .venv/bin/python app/tools/stripe_plan_setup.py              # test mode first
    STRIPE_SETUP_KEY=rk_live_... .venv/bin/python app/tools/stripe_plan_setup.py --tax        # then for real
    ... stripe_plan_setup.py --price 1500                                                       # another price (cents)
    ... stripe_plan_setup.py --portal https://billing.stripe.com/p/login/...                    # the customer portal

A restricted key with write access to Products, Prices and Payment Links; it is not saved: delete it afterwards.
--tax turns on Stripe Tax for the link (activate Stripe Tax and add your registrations in the dashboard first;
prices are then before tax). --portal: the no-code customer portal's login link (Dashboard → Settings → Billing →
Customer portal → Activate link), where subscribers update their card and cancel; the account page links to it.
The link sends the payer back to /?account&subscribed={CHECKOUT_SESSION_ID}; the page names the account with
?client_reference_id=<account id>; app/accounts.py checks payments with the read-only key (stripe_read_key),
which needs "Checkout Sessions: read" and "Subscriptions: read", and must be of the same mode as the link.
Run again safely: it reuses what matches and turns off links that don't.
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
PRODUCT = {"name": "Science search API: paid plan",
           "description": "300 requests a minute for your API keys and AI agents on sciencemadereadable.com "
                          "(100,000 a month), instead of 60 with a free account. Cancel at any time.",
           "metadata[smr]": "plan"}


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
    ap.add_argument("--price", type=int, default=1000, help="a month, in cents (default $10)")
    ap.add_argument("--tax", action="store_true", help="Stripe Tax on the link (set it up in the dashboard first)")
    ap.add_argument("--portal", help="the customer portal's login link, to save with the plan")
    a = ap.parse_args()
    cfg = json.loads(SUPPORT.read_text()) if SUPPORT.exists() else {}
    if a.portal and not KEY:   # only the portal link changes: no Stripe key needed
        if not cfg.get("plan_payment"):
            sys.exit("Make the plan's link first (with STRIPE_SETUP_KEY).")
        cfg["plan_payment"]["portal"] = a.portal
        SUPPORT.write_text(json.dumps(cfg, indent=1) + "\n")
        return print(f"portal saved: {a.portal}")
    if not KEY.startswith(("sk_", "rk_")):
        sys.exit("Set STRIPE_SETUP_KEY to a Stripe secret or restricted key (sk_... or rk_...).")
    mode = "test" if "_test_" in KEY else "live"
    product = next((p for p in every("products", {"active": "true"}) if p["metadata"].get("smr") == "plan"), None) \
        or stripe("POST", "products", PRODUCT)
    behaviour = "exclusive" if a.tax else "unspecified"
    price = next((p for p in every("prices", {"product": product["id"], "active": "true"})
                  if p.get("unit_amount") == a.price and p.get("currency") == "usd"
                  and (p.get("recurring") or {}).get("interval") == "month"
                  and (not a.tax or p.get("tax_behavior") == "exclusive")), None) \
        or stripe("POST", "prices", {"product": product["id"], "currency": "usd", "unit_amount": a.price,
                                     "recurring[interval]": "month", "tax_behavior": behaviour, "metadata[smr]": "plan"})
    redirect = f"{SITE}/?account&subscribed={{CHECKOUT_SESSION_ID}}"
    link = None
    for l in every("payment_links", {"active": "true"}):
        if l["metadata"].get("smr") != "plan":
            continue
        items = stripe("GET", f"payment_links/{l['id']}/line_items")["data"]
        same = (l["metadata"].get("smr_version") == VERSION and items and items[0]["price"]["id"] == price["id"]
                and (l.get("after_completion") or {}).get("redirect", {}).get("url") == redirect
                and bool((l.get("automatic_tax") or {}).get("enabled")) == a.tax)
        if same and link is None:
            link = l
        else:
            stripe("POST", f"payment_links/{l['id']}", {"active": "false"})
    if link is None:
        link = stripe("POST", "payment_links", {
            "line_items[0][price]": price["id"], "line_items[0][quantity]": "1",
            "after_completion[type]": "redirect", "after_completion[redirect][url]": redirect,
            "automatic_tax[enabled]": "true" if a.tax else "false",
            "billing_address_collection": "required" if a.tax else "auto",
            "custom_text[submit][message]": "Your API keys get the paid limits as soon as the payment goes through. "
                                            "Cancel at any time; you keep them until the end of the month paid for. "
                                            "Within 14 days of your first payment, ask and you get it back in full.",
            "subscription_data[metadata][smr]": "plan",
            "metadata[smr]": "plan", "metadata[smr_version]": VERSION})
        print(f"made the link {link['id']}")
    old = cfg.get("plan_payment") or {}
    cfg["plan_payment"] = {"url": link["url"], "id": link["id"], "price": price["id"], "amount": a.price, "mode": mode,
                           "tax": a.tax, "portal": a.portal or (old.get("portal") if old.get("mode") == mode else None)}
    SUPPORT.write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"{mode} mode: {link['url']} (${a.price / 100:.2f} a month{' + tax' if a.tax else ''}) saved in "
          f"{SUPPORT.relative_to(APP.parent)}")
    if not cfg["plan_payment"]["portal"]:
        print("No customer portal yet: activate its login link in the dashboard, then run with --portal URL.")
    if mode == "test":
        print("Test card: 4242 4242 4242 4242. Test links show only on the private port; the read key must be a test "
              "key meanwhile. Then run again with the live key.")


main()
