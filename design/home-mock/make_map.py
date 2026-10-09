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

BUILD = Path("/mnt/fast/science-atlas/build/20261008T131030-tsne")
OUT = Path(__file__).resolve().parent
SIZE = 1400

# hue (degrees), saturation per domain
DOMAINS = {"Life Sciences": (150, 0.50), "Health Sciences": (14, 0.55),
           "Physical Sciences": (214, 0.46), "Social Sciences": (36, 0.36)}
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

a = np.log1p(count) / np.log1p(np.percentile(count[count > 0], 99.5))
a = np.clip(a, 0, 1) ** 0.8
a = a * np.where(grey >= np.maximum(count, 1) * 0.999, 0.25, 1.0)  # specks of unclassified papers: faint

for theme in ("light", "dark"):
    pal = palette(theme)
    rgb = np.stack([np.bincount(cell, weights=pal[field, c], minlength=SIZE * SIZE) for c in range(3)], 1)
    rgb = rgb / np.maximum(count, 1)[:, None]
    alpha = a ** 0.75 if theme == "light" else a  # on cream, faint ink reads as grey: lift it
    img = np.concatenate([rgb, alpha[:, None]], 1).reshape(SIZE, SIZE, 4)
    sharp = Image.fromarray((img * 255).astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(0.7))
    glow = np.asarray(sharp.filter(ImageFilter.GaussianBlur(10))).astype(np.float32)
    glow[..., 3] *= 0.30 if theme == "light" else 0.55
    out = Image.alpha_composite(Image.fromarray(glow.astype(np.uint8), "RGBA"), sharp)
    out.save(OUT / f"science-map-{theme}.webp", quality=72, method=6)
    out.resize((480, 480), Image.LANCZOS).save(OUT / f"science-map-{theme}-small.webp", quality=70, method=6)
    print(theme, f"{(OUT / f'science-map-{theme}.webp').stat().st_size / 1e6:.2f} MB")

regions = [{"name": f["name"], "domain": f["domain"], "x": f["anchor"]["x"], "y": f["anchor"]["y"],
            "dots": int(sum(f["per_year"]))}
           for f in fields if f["domain"]]
(OUT / "regions.json").write_text(json.dumps({"atlas": meta["id"], "fields": regions}, indent=1))
