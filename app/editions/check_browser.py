"""Real browser checks. --upload creates a public-sample private job; --confirm resumes it.
The state file contains a capability and must stay private, never in the repository.
"""

import argparse
import json
import os
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
STATE = Path("/tmp/smr-editions-browser-state.json")
OUT = Path("/tmp/smr-editions-browser")


def wait(page, expression, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if page.evaluate(expression):
            return
        page.wait_for_timeout(200)
    raise AssertionError("Timed out: " + expression)


def check(base, upload=False, confirm=False):
    OUT.mkdir(exist_ok=True)
    errors = []
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM_BIN"), headless=True
        )
        c = b.new_context(
            viewport={"width": 1280, "height": 900},
            storage_state=str(STATE) if STATE.exists() and not upload else None,
        )
        page = c.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base.rstrip("/") + "/?reading", wait_until="networkidle")
        page.wait_for_selector("#edition-submit")
        assert "Private by default" in page.inner_text("#reading-desk")
        assert page.locator("#edition-consent").count() == 1
        page.screenshot(path=OUT / "desk.png", full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")
        page.screenshot(path=OUT / "desk-phone.png", full_page=True)
        page.set_viewport_size({"width": 1280, "height": 900})
        if upload:
            page.locator("#edition-file").set_input_files(
                ROOT / "pdf_prototype/runs/toxics/source.pdf"
            )
            page.click("#edition-submit")
            assert "Confirm permission" in page.inner_text("#edition-message")
            page.check("#edition-rights")
            page.check("#edition-consent")
            page.uncheck("#edition-audio")
            page.click("#edition-submit")
            page.wait_for_selector(".edition-job", timeout=30000)
            c.storage_state(path=str(STATE))
            STATE.chmod(0o600)
            page.wait_for_selector('button[data-action="review"]', timeout=180000)
            page.click('button[data-action="review"]')
            page.wait_for_selector(".edition-block")
            print("Real upload reached section review.", flush=True)
            for block in page.locator(".edition-block[open]").all():
                print(
                    block.locator("summary").inner_text(),
                    block.locator("pre").inner_text()[:140],
                    flush=True,
                )
            page.screenshot(path=OUT / "review.png", full_page=True)
        if confirm:
            page.wait_for_selector('button[data-action="review"]', timeout=20000)
            page.click('button[data-action="review"]')
            page.wait_for_selector(".edition-block")
            # Only approve the exact kinds of uncertain passages inspected in the prototype.
            for block in page.locator(".edition-block[open]").all():
                text = block.locator("pre").inner_text()
                known = text.startswith(
                    (
                        "## Article",
                        "ND, not detected",
                        "40. Arbuckle",
                        "**Disclaimer/Publisher",
                    )
                )
                if not known:
                    raise AssertionError(
                        "Unexpected uncertain block; inspect before confirming: "
                        + text[:160]
                    )
                block.locator(".block-reviewed").check()
            first = page.locator(".edition-block[open]").first
            first.locator(".source-page").click()
            wait(
                page,
                "document.querySelector('#edition-page-dialog img').naturalWidth>0",
            )
            page.click("#edition-page-dialog button")
            page.check("#edition-confirm-all")
            page.click("#edition-confirm")
            page.wait_for_selector('button[data-action="open"]', timeout=900000)
            print("Real uploaded PDF completed rewriting and exports.", flush=True)
        if confirm:
            with c.expect_page() as opened:
                page.click('button[data-action="open"]')
            reader = opened.value
            reader.wait_for_load_state("networkidle")
            reader.wait_for_selector("#start")
            reader.click("#start")
            wait(
                reader,
                "document.querySelector('#progress').textContent.includes(' / ')",
            )
            assert reader.locator("#flow").inner_text().strip()
            reader.screenshot(path=OUT / "private-reader.png", full_page=True)
            assert reader.url.count("token") == 0
            # No cookie or bearer: a fresh browser cannot open this private reader.
            stranger = b.new_context()
            response = stranger.request.get(reader.url)
            assert response.status == 404
            stranger.close()
        assert not errors, errors
        c.storage_state(path=str(STATE))
        STATE.chmod(0o600)
        b.close()
    print("Browser checks passed.", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("base", nargs="?", default="http://127.0.0.1:8940")
    a.add_argument("--upload", action="store_true")
    a.add_argument("--confirm", action="store_true")
    x = a.parse_args()
    check(x.base, x.upload, x.confirm)
