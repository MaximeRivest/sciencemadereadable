"""Render the science atlas as one still image for the homepage (no WebGL needed to show it).

  cd citewalk/science && uv run python ../../sciencemadereadable/design/home-mock/make_map.py

Every publication dot of the atlas's 1 % sample (1,279,555, all years), coloured by its OpenAlex
field, density tone-mapped, with a soft glow. Transparent background: the page decides the paper
colour and can grey the map with CSS. Writes science-map.webp (and regions.json: field labels).
Image row = atlas y (y grows downward, as in the atlas viewer).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

BUILD = Path("/mnt/fast/science-atlas/build/20261008T131030-tsne")
OUT = Path(__file__).resolve().parent
SIZE = 1400

meta = json.loads((BUILD / "meta.json").read_text())
xy = np.fromfile(BUILD / "points.f32", np.float32).reshape(-1, 2)
field = np.fromfile(BUILD / "field.u8", np.uint8)
pal = np.array([[int(f["colour"][1 + 2 * k:3 + 2 * k], 16) for k in range(3)] for f in meta["fields"]], np.float32) / 255
unclassified = [i for i, f in enumerate(meta["fields"]) if f["name"] == "Unclassified"]
pal[unclassified] = [0.55, 0.55, 0.58]

gx = np.clip((xy[:, 0] * SIZE).astype(int), 0, SIZE - 1)
gy = np.clip((xy[:, 1] * SIZE).astype(int), 0, SIZE - 1)
cell = gy * SIZE + gx
count = np.bincount(cell, minlength=SIZE * SIZE).astype(np.float32)
rgb = np.stack([np.bincount(cell, weights=pal[field, c], minlength=SIZE * SIZE) for c in range(3)], 1)
rgb = rgb / np.maximum(count, 1)[:, None]

# tone map: log density, so sparse regions still show
a = np.log1p(count) / np.log1p(np.percentile(count[count > 0], 99.5))
a = np.clip(a, 0, 1) ** 0.8
# places holding only unclassified papers (isolated grey specks at the edges): faint
grey = np.bincount(cell, weights=np.isin(field, unclassified).astype(np.float32), minlength=SIZE * SIZE)
a = a * np.where(grey >= np.maximum(count, 1) * 0.999, 0.25, 1.0)
img = np.concatenate([rgb, a[:, None]], 1).reshape(SIZE, SIZE, 4)
sharp = Image.fromarray((img * 255).astype(np.uint8), "RGBA").filter(ImageFilter.GaussianBlur(0.7))

# glow: the same image blurred and dimmed underneath
glow = sharp.filter(ImageFilter.GaussianBlur(10))
g = np.asarray(glow).astype(np.float32)
g[..., 3] *= 0.55
out = Image.alpha_composite(Image.fromarray(g.astype(np.uint8), "RGBA"), sharp)
out.save(OUT / "science-map.webp", quality=72, method=6)
regions = [{"name": f["name"], "colour": f["colour"], "x": f["anchor"]["x"], "y": f["anchor"]["y"]}
           for f in meta["fields"] if f["name"] != "Unclassified"]
(OUT / "regions.json").write_text(json.dumps({"atlas": meta["id"], "fields": regions}, indent=1))
print(f"{(OUT / 'science-map.webp').stat().st_size / 1e6:.2f} MB, {len(regions)} fields")

# a small copy that shows at once; the page swaps in the full one when it has loaded
out.resize((480, 480), Image.LANCZOS).save(OUT / "science-map-small.webp", quality=70, method=6)
print(f"small: {(OUT / 'science-map-small.webp').stat().st_size / 1e3:.0f} kB")
