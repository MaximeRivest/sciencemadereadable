"""Walk through the site like a visitor, in a real (headless) browser. Exit code 0 = every check passed.

    tools/dev.sh check [URL]        (default: the dev site; publish.sh runs it on dev before and on live after)

Desktop and phone (touch): home, search by meaning and by exact words with dots on the globe, a study's page with
its citation walk, one step of the walk, the map opened by a tap, pinch zoom of the map (not the page), every
study as dots once zoomed in, a readable paper opening in the reader, light/dark. Any page error fails.
Screenshots: /tmp/smr-check/.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8940/").rstrip("/") + "/"
OUT = Path("/tmp/smr-check")
OUT.mkdir(exist_ok=True)
results: list[tuple[bool, str]] = []


def check(ok, what: str):
    results.append((bool(ok), what))
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)


def until(pg, js: str, timeout: float = 45) -> bool:
    t = time.time()
    while time.time() - t < timeout:
        if pg.evaluate(js):
            return True
        pg.wait_for_timeout(250)
    return False


def zoom(pg) -> float:
    # the live map's camera (the CSS world stops following it once the WebGL map has taken over)
    z = pg.evaluate('window.__mapStats ? window.__mapStats().z : null')
    return float(z if z is not None else (pg.evaluate('getComputedStyle(document.querySelector("#world")).getPropertyValue("--z")') or 1))


def desktop(b, errs):
    print("desktop", flush=True)
    pg = b.new_page(viewport={"width": 1280, "height": 800}, color_scheme="light")
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(BASE)
    check(until(pg, '!!document.querySelector("form.search.big input")', 15), "home page loads")
    pg.screenshot(path=OUT / "home.png")
    pg.fill("form.search.big input", "coral reefs and marine heatwaves")
    pg.press("form.search.big input", "Enter")
    check(until(pg, 'document.querySelectorAll("#results .hit").length >= 5'), "search by meaning returns results")
    check(pg.evaluate('document.querySelectorAll("#dots .dot").length') >= 3, "results glow on the globe")
    check("meaning" in (pg.evaluate('document.querySelector(".search-how")?.textContent') or ""), "says it searched by meaning")
    pg.screenshot(path=OUT / "results.png")
    pg.goto(BASE + '?q=%22gut%20microbiome%22%20autism')
    check(until(pg, 'document.querySelectorAll("#results .hit").length >= 5')
          and "contain these words" in pg.evaluate('document.querySelector(".search-how").textContent'), "exact-word search")
    pg.goto(BASE + "?work=10.1371/journal.pone.0322636")
    check(until(pg, 'document.querySelectorAll(".walk-group").length >= 2'), "a study's page shows its citation walk")
    check(pg.evaluate('document.querySelectorAll("#links line").length') > 0, "the walk is drawn on the map")
    pg.screenshot(path=OUT / "walk.png")
    href = pg.evaluate('document.querySelector(".walk-group .hit").getAttribute("href")')
    pg.click(f'.walk-group .hit[href="{href}"]')
    check(until(pg, f'location.search.includes({href[1:]!r}.split("=")[1]) && !!document.querySelector("#work h1")'),
          "one step of the walk opens the next study")
    pg.go_back()
    check(until(pg, 'document.querySelectorAll(".walk-group").length >= 2', 15), "back returns to the walk")
    pg.click("#walk-map")
    check(until(pg, 'document.body.dataset.view === "explore"', 10), "the walk opens on the map")
    for _ in range(3):
        pg.click("#ex-in")
        pg.wait_for_timeout(250)
    check(until(pg, 'document.body.classList.contains("gl") && !!window.__mapStats && window.__mapStats().stars > 1000', 25),
          "zoomed in, the map engine draws studies as stars")
    spot = None
    for my in range(260, 620, 29):          # a study under the mouse (hover ring) and no walk marker on top
        for mx in range(380, 960, 23):
            pg.mouse.move(mx, my); pg.wait_for_timeout(25)
            if pg.evaluate(f'!document.querySelector("#hover-ring").hidden && !document.elementFromPoint({mx}, {my}).closest(".dot")'):
                spot = (mx, my); break
        if spot: break
    if spot:
        pg.mouse.click(*spot)
    check(spot and until(pg, '!document.querySelector("#pick").hidden && !/Finding/.test(document.querySelector("#pick").innerText)', 15),
          f"tapping the map identifies a study (at {spot})")
    # the "selected" ring sits on the tapped study (it drifted ~50 px at deep zoom when it lived in the scaled world)
    pg.evaluate("window.__fly(0.62, 0.45, 150)"); pg.wait_for_timeout(3000)
    pg.evaluate('document.querySelector("#pick-close").click()')
    ring = None
    for mx in range(400, 900, 17):          # find a study under the mouse (the hover ring shows on it)
        pg.mouse.move(mx, 420); pg.wait_for_timeout(40)
        ring = pg.evaluate('(() => { const r = document.querySelector("#hover-ring"); if (r.hidden) return null;'
                           f' if (document.elementFromPoint({mx}, 420).closest(".dot")) return null;'
                           ' const b = r.getBoundingClientRect(); return [b.x + b.width / 2, b.y + b.height / 2] })()')
        if ring:
            pg.mouse.click(mx, 420)
            break
    until(pg, '!!document.querySelector(".picked")', 10)
    c = pg.evaluate('(() => { const b = document.querySelector(".picked")?.getBoundingClientRect(); return b ? [b.x + b.width / 2, b.y + b.height / 2] : null })()')
    check(ring and c and abs(c[0] - ring[0]) < 3 and abs(c[1] - ring[1]) < 3,
          f"deep zoom: the selected ring is on the tapped study (study {ring}, ring {c})")
    pg.screenshot(path=OUT / "map.png")
    pg.goto(BASE + "?q=coral%20bleaching&readable=1")
    until(pg, 'document.querySelectorAll("#results .hit").length > 0')
    pg.click("#results .hit")
    check(until(pg, '!!document.querySelector("#to-walk")', 40), "a readable paper opens in the reader")
    pg.click("#theme")
    pg.wait_for_timeout(600)
    check(pg.evaluate("document.documentElement.dataset.theme") == "dark", "light/dark button switches")
    pg.screenshot(path=OUT / "reader-dark.png")
    pg.close()


def phone(p, b, errs):
    print("phone", flush=True)
    c = b.new_context(**p.devices["iPhone 13"])
    pg = c.new_page()
    pg.on("pageerror", lambda e: errs.append(str(e)))
    cdp = c.new_cdp_session(pg)
    pg.goto(BASE)
    until(pg, '!!document.querySelector("form.search.big input")', 15)
    pg.wait_for_timeout(1500)
    w, h = pg.viewport_size["width"], pg.viewport_size["height"]
    pg.touchscreen.tap(w / 2, h - 50)
    check(until(pg, 'document.body.dataset.view === "explore"', 5), "phone: tapping the home map opens it")
    pg.wait_for_timeout(1000)   # the map grows to full screen (0.8 s)
    z0 = zoom(pg)
    pts = lambda d: [{"x": w / 2 - d / 2, "y": h / 2, "id": 1}, {"x": w / 2 + d / 2, "y": h / 2, "id": 2}]
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": pts(80)})
    for i in range(1, 9):
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": pts(80 + 20 * i)})
        pg.wait_for_timeout(30)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    pg.wait_for_timeout(300)
    z1, scale = zoom(pg), pg.evaluate("visualViewport.scale")
    check(z1 > z0 * 2 and scale == 1, f"phone: pinch zooms the map, not the page (map x{z1 / z0:.1f}, page x{scale})")
    pg.screenshot(path=OUT / "phone-map.png")
    pg.goto(BASE + "?q=why%20do%20we%20need%20sleep")
    check(until(pg, 'document.querySelectorAll("#results .hit").length >= 5'), "phone: search")
    check(pg.evaluate("document.querySelector('nav.top').scrollWidth <= innerWidth"), "phone: nothing wider than the screen")
    pg.screenshot(path=OUT / "phone-results.png")
    c.close()


def main():
    errs: list[str] = []
    chromium = os.environ.get("CHROMIUM_BIN")
    print(f"checking {BASE}", flush=True)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chromium, args=["--no-sandbox"]) if chromium else p.chromium.launch()
        desktop(b, errs)
        phone(p, b, errs)
        b.close()
    check(not errs, "no page errors" + (f": {errs[:3]}" if errs else ""))
    bad = [w for ok, w in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed; screenshots in {OUT}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
