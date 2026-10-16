"""Exercise the report in Chromium, without contacting external services."""
import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
from threading import Thread

from playwright.sync_api import sync_playwright


def check(folder):
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(folder)))
    Thread(target=server.serve_forever, daemon=True).start()
    errors = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_BIN") or shutil.which("chromium"), headless=True)
            page = browser.new_page(viewport={"width": 1350, "height": 950})
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")
            assert page.locator("#rewritten").inner_text().strip()
            assert page.locator("#stats .stat").count() == 4
            page.select_option("#section", "results")
            assert page.locator("#source table").count() > 0
            assert page.locator("#rewritten table").count() == 0
            assert page.locator("#rewritten .notice").count() > 0
            page.click('[data-tab="mapping"]')
            assert page.locator("#ledger details").count() > 0
            page.locator("#reviews summary").first.click()
            page.locator("#reviews button").first.click()
            page.wait_for_function("document.getElementById('pdf-page').complete && document.getElementById('pdf-page').naturalWidth > 0")
            page.select_option("#page", "13")
            page.wait_for_function("document.getElementById('pdf-page').complete && document.getElementById('pdf-page').naturalWidth > 0")
            page.click('[data-tab="checks"]')
            assert "32/33" in page.locator("#check-summary").inner_text()
            page.click('[data-tab="read"]')
            page.select_option("#section", "abstract")
            page.screenshot(path=str(folder.parent / "report-desktop.png"), full_page=True)
            page.set_viewport_size({"width": 390, "height": 844})
            for tab in ("read", "mapping", "pdf", "checks"):
                page.click(f'[data-tab="{tab}"]')
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Horizontal overflow in {tab}"
            page.click('[data-tab="read"]')
            page.add_style_tag(content=":root{--color-background-primary:#161819;--color-background-secondary:#24282a;--color-text-primary:#edf1f2;--color-text-secondary:#b6bec2;--color-border-primary:#50585c;--color-text-info:#9ad3ef;--color-text-warning:#ebc270;--color-text-success:#a6dcb9}")
            page.screenshot(path=str(folder.parent / "report-mobile-dark.png"), full_page=True)
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
    print("Report checked: desktop/mobile, all tabs, page images, table safety, dark theme; no script errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, default=Path("demo"), nargs="?")
    check(parser.parse_args().folder.resolve())
