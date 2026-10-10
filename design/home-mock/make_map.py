"""Render the science atlas as still images for the homepage (no WebGL needed to show them).

  cd citewalk/science && uv run python ../../sciencemadereadable/design/home-mock/make_map.py

Every publication dot of the atlas's 1 % sample (1,279,555, all years), density tone-mapped.
Colours follow the site, not the atlas: one muted family per OpenAlex domain, a shade per field.
  Life sciences     sage (the site's accent)      Health sciences   clay
  Physical sciences slate blue                    Social sciences   warm stone
Gold stays free for search results. Two themes, same geometry:
  light: darker inks for the cream page; dark: lighter tints with a soft glow for the dark page.
Writes science-map-{light,dark}.webp (+ -small, 480 px, shown first) and regions.json (labels).
Image row = atlas y (y grows downward, as in the atlas viewer).
"""
import colorsys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BUILD = Path("/mnt/fast/science-atlas/build/20261009T190750-tsne-mode")   # same geography, placement "mode"
OUT = Path(__file__).resolve().parent
SIZE = 1400

# hue (degrees), saturation per domain
DOMAINS = {"Life Sciences": (150, 0.62), "Health Sciences": (14, 0.66),
           "Physical Sciences": (214, 0.60), "Social Sciences": (38, 0.50)}
LIGHTNESS = {"light": (0.26, 0.40), "dark": (0.56, 0.70)}  # range over the fields of a domain

meta = json.loads((BUILD / "meta.json").read_text())
fields = meta["fields"]


def palette(theme: str) -> np.ndarray:
    lo, hi = LIGHTNESS[theme]
    pal = np.zeros((len(fields), 3), np.float32)
    for dom, (hue, sat) in DOMAINS.items():
        members = [i for i, f in enumerate(fields) if f["domain"] == dom]
        for rank, i in enumerate(members):  # neighbouring fields: different shade and a small hue shift
            t = rank / max(1, len(members) - 1)
            light = lo + (hi - lo) * ((rank * 0.618) % 1)
            h = (hue + (t - 0.5) * 22) % 360
            pal[i] = colorsys.hls_to_rgb(h / 360, light, sat)
    for i, f in enumerate(fields):
        if not f["domain"]:
            pal[i] = (0.5, 0.49, 0.47) if theme == "light" else (0.62, 0.6, 0.57)
    return pal


xy = np.fromfile(BUILD / "points.f32", np.float32).reshape(-1, 2)
field = np.fromfile(BUILD / "field.u8", np.uint8)
unclassified = [i for i, f in enumerate(fields) if not f["domain"]]
gx = np.clip((xy[:, 0] * SIZE).astype(int), 0, SIZE - 1)
gy = np.clip((xy[:, 1] * SIZE).astype(int), 0, SIZE - 1)
cell = gy * SIZE + gx
count = np.bincount(cell, minlength=SIZE * SIZE).astype(np.float32)
grey = np.bincount(cell, weights=np.isin(field, unclassified).astype(np.float32), minlength=SIZE * SIZE)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a float [H, W, C] image, per channel."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=(sigma, sigma, 0), mode="constant")


def saturate(rgb: np.ndarray, k: float) -> np.ndarray:
    grey = rgb.mean(-1, keepdims=True)
    return np.clip(grey + (rgb - grey) * k, 0, None)


import os
# exposure: the densest places saturate. 99.7 for the first atlas; the "mode" placement packs papers into their
# clusters (higher peaks), so the same brightness overall needs a lower percentile (EXPOSURE_PCT, matched by eye
# and by mean brightness against the first images)
ref = np.percentile(count[count > 0], float(os.environ.get("EXPOSURE_PCT", 96.5 if "mode" in BUILD.name else 99.7)))
for theme in ("light", "dark"):
    pal = palette(theme)
    # summed colour per place (not averaged): dense places collect more light / more ink
    rgb = np.stack([np.bincount(cell, weights=pal[field, c], minlength=SIZE * SIZE) for c in range(3)], 1)
    rgb = rgb.reshape(SIZE, SIZE, 3)
    mute = np.where(grey >= np.maximum(count, 1) * 0.999, 0.3, 1.0).reshape(SIZE, SIZE, 1)
    rgb = rgb * mute / ref
    if theme == "dark":
        # long exposure: stars + glow at three scales, then a film-like curve; bright cores burn toward white
        light = rgb * 1.6 + blur(rgb, 1.2) * 1.4 + blur(rgb, 6) * 2.2 + blur(rgb, 22) * 3.0
        light = saturate(light, 1.35)
        lum = light.max(-1, keepdims=True)
        exposed = 1 - np.exp(-light * 1.6)
        white = np.clip((1 - np.exp(-lum * 0.9)) - 0.55, 0, 1) / 0.45   # only the densest cores
        exposed = exposed * (1 - white * 0.6) + white * 0.6
        a = np.clip(exposed.max(-1, keepdims=True), 0, 1)
        out_rgb = np.where(a > 1e-4, exposed / np.maximum(a, 1e-4), 0)
    else:
        # backlit glass: the same summed light as the dark map, shown as vivid colour on cream;
        # the densest cores turn into pale, warm light ringed by their colour (glow without darkness)
        light = rgb * 1.6 + blur(rgb, 1.2) * 1.3 + blur(rgb, 6) * 2.0 + blur(rgb, 22) * 2.6
        exposed = 1 - np.exp(-light * 1.8)
        lum = exposed.max(-1, keepdims=True)
        hue = saturate(exposed / np.maximum(lum, 1e-6), 1.6)
        hue = hue / np.maximum(hue.max(-1, keepdims=True), 1e-6)       # pure, bright colour
        core = np.clip((lum - 0.62) / 0.38, 0, 1) ** 1.5                # only the densest places
        glowcol = np.array([1.0, 0.985, 0.94])                          # warm white, brighter than the page
        haze_rgb = hue * 0.86 * (1 - core) + glowcol * core
        haze_a = np.clip(lum ** 0.9 * 0.85 + core * 0.15, 0, 1)
        # sharp grains on top, in the deep version of their colour: the "stars" inside the glow
        grain = rgb * 1.0 + blur(rgb, 0.7) * 1.2
        g_lum = grain.max(-1, keepdims=True)
        g_hue = saturate(grain / np.maximum(g_lum, 1e-6), 1.5)
        g_hue = g_hue / np.maximum(g_hue.max(-1, keepdims=True), 1e-6)
        g_a = np.clip(1 - np.exp(-g_lum * 4.0), 0, 1) * 0.75 * (1 - core * 0.85)   # cores stay luminous
        g_rgb = g_hue * 0.50
        a = g_a + haze_a * (1 - g_a)                                     # grains over haze ("over")
        out_rgb = (g_rgb * g_a + haze_rgb * haze_a * (1 - g_a)) / np.maximum(a, 1e-6)
    img = np.concatenate([np.clip(out_rgb, 0, 1), a], -1)
    out = Image.fromarray((img * 255).astype(np.uint8), "RGBA")
    out.save(OUT / f"science-map-{theme}.webp", quality=74, method=6)
    out.resize((480, 480), Image.LANCZOS).save(OUT / f"science-map-{theme}-small.webp", quality=70, method=6)
    print(theme, f"{(OUT / f'science-map-{theme}.webp').stat().st_size / 1e6:.2f} MB")

hexs = lambda c: "#%02x%02x%02x" % tuple(int(round(max(0, min(1, v)) * 255)) for v in c)
PAL = {th: palette(th) for th in ("light", "dark")}
regions = [{"name": f["name"], "domain": f["domain"], "x": f["anchor"]["x"], "y": f["anchor"]["y"],
            "dots": int(sum(f["per_year"])), "light": hexs(PAL["light"][i]), "dark": hexs(PAL["dark"][i])}
           for i, f in enumerate(fields) if f["domain"]]
(OUT / "regions.json").write_text(json.dumps({"atlas": meta["id"], "fields": regions}, indent=1))
