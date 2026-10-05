# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright>=1.50,<2"]
# ///
"""Check the local companion in an isolated, headless browser.

Setup: uv run --with playwright playwright install chromium --only-shell
Run:   uv run pine-nut-visuals/check.py
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1100, "height": 900})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(Path(__file__).with_name("index.html").resolve().as_uri())
    assert page.locator("#yield-difference").inner_text().startswith("4.3 grams")
    assert page.locator("#light-waffle .damaged").count() == 16
    assert page.locator("#heavy-waffle .damaged").count() == 9
    for kind in ("light", "heavy"):
        assert page.locator(f"#{kind}-waffle .seed").count() == 100
    assert page.locator("#leaves-table tbody tr").count() == 6
    assert page.evaluate("studyData.leaves.reduce((sum, group) => sum + group.n, 0)") == 328
    page.get_by_role("button", name="100 grams", exact=True).click()
    assert "0.43 grams" in page.locator("#yield-difference").inner_text()
    assert "3.62 g" in page.locator("#yield-chart").inner_text()
    paths = [
        (["many", "higherShare", "higherHigher"], "5.10 g"),
        (["many", "higherShare", "higherLower"], "4.35 g"),
        (["many", "lowerShare", "lowerLower"], "3.00 g"),
        (["many", "lowerShare", "lowerHigher"], "3.94 g"),
        (["few", "fewest"], "1.83 g"),
        (["few", "some"], "3.02 g"),
    ]
    for path, outcome in paths:
        for group in path:
            page.locator(f"#choice-{group}").click()
        assert outcome in page.locator(".tree-result").inner_text()
        assert page.evaluate("document.activeElement.id") == f"choice-{path[-1]}"
    page.get_by_role("button", name="1 kilogram", exact=True).click()
    for width in (360, 390, 768, 1100):
        page.set_viewport_size({"width": width, "height": 850})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), width
    assert not errors, errors
    browser.close()
print("PASS: charts, units, all six branches, focus preservation, mobile layout, and no script errors.")
