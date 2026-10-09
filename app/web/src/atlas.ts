/**
 * The map of science, always on screen in some form: a planet rising behind the home page, a quiet globe
 * beside the results (where they glow), a small "you are here" beside a study, a home button in the reader,
 * and the whole map to explore. One element; CSS places it by the view (body[data-view]).
 *
 * Layers inside #globe, bottom to top: the rendered images (design/home-mock/make_map.py; the haze of all of
 * science), #tiles (every study as a dot once zoomed in: the release's quadtree tiles, drawn into canvases),
 * #links (citations of a walk), #labels (field names), #dots (search results, a walk, "you are here").
 * All positions are atlas coordinates in [0, 1]; the data API gives them.
 */

export type Kind = "result" | "ref" | "similar" | "next";
export interface Spot { id: string; x: number; y: number; readable?: boolean; title?: string; meta?: string; kind?: Kind; strong?: boolean }
export interface Line { x1: number; y1: number; x2: number; y2: number; kind: string }

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector(sel) as T;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
const MAX_ZOOM = 160;          // deep enough to reach the finest tiles (every study)
const TILES_FROM = 1.5;        // below this the rendered image is sharper than the dots

let api = "";
let results: Spot[] = [];
let walkSpots: Spot[] = [];
let walkLines: Line[] = [];
let context: "results" | "walk" = "results";
let question: { x: number; y: number } | null = null;
let here: { x: number; y: number } | null = null;
let openSpot: (s: Spot) => void = () => {};
let picked: { x: number; y: number } | null = null;
const view = { z: 1, x: 0, y: 0 };

export function initMap(opts: { api: string; explore: () => void; open: (s: Spot) => void }) {
  api = opts.api;
  openSpot = opts.open;
  const map = $("#map");
  // sharp images after the page is up, the current theme first
  const order = document.documentElement.dataset.theme === "dark" ? ["dark", "light"] : ["light", "dark"];
  order.forEach((th, n) => setTimeout(() => {
    const im = new Image();
    im.onload = () => { ($(`.map-img.${th}`) as HTMLImageElement).src = im.src; };
    im.src = `map/science-map-${th}.webp`;
  }, 400 + n * 1500));
  fetch("map/regions.json").then((r) => r.json()).then((d) => {
    const big = new Set([...d.fields].sort((a: any, b: any) => b.dots - a.dots).slice(0, 10).map((f: any) => f.name));
    const box = $("#labels");
    for (const f of d.fields) {
      fieldColour[f.name] = { light: hexRGB(f.light), dark: hexRGB(f.dark) };
      const s = at(document.createElement("span"), f.x, f.y);
      if (!big.has(f.name)) s.className = "small";
      s.textContent = f.name;
      box.appendChild(s);
    }
  }).catch(() => {});

  map.addEventListener("click", () => { if (document.body.dataset.view !== "explore") opts.explore(); });
  map.addEventListener("keydown", (e) => { if (e.key === "Enter" && document.body.dataset.view !== "explore") opts.explore(); });
  map.addEventListener("wheel", (e) => {
    if (document.body.dataset.view !== "explore") return;
    e.preventDefault();
    const r = map.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    zoomAt(mx, my, Math.exp(-e.deltaY * 0.0015));
  }, { passive: false });
  // one finger (or the mouse) moves, two fingers pinch and move together; a tap identifies a study;
  // a double tap zooms in
  const pts = new Map<number, { x: number; y: number }>();
  let g: { cx: number; cy: number; d: number } | null = null;
  let down: { x: number; y: number; t: number; many: boolean } | null = null;
  let tapTimer = 0;
  const gesture = () => {
    const a = [...pts.values()], r = map.getBoundingClientRect();
    const cx = a.reduce((s, p) => s + p.x, 0) / a.length - r.left, cy = a.reduce((s, p) => s + p.y, 0) / a.length - r.top;
    return { cx, cy, d: a.length > 1 ? Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y) : 0 };
  };
  map.addEventListener("pointerdown", (e) => {
    if (document.body.dataset.view !== "explore" || (e.target as HTMLElement).classList.contains("dot")) return;
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    down = pts.size === 1 ? { x: e.clientX, y: e.clientY, t: performance.now(), many: false } : down && { ...down, many: true };
    try { map.setPointerCapture(e.pointerId); } catch { /* */ }
    map.classList.add("dragging");
    g = gesture();
  });
  map.addEventListener("pointermove", (e) => {
    if (!pts.has(e.pointerId)) return;
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const n = gesture();
    if (g) {
      view.x += n.cx - g.cx; view.y += n.cy - g.cy;
      if (n.d && g.d) zoomAt(n.cx, n.cy, n.d / g.d); else apply();
    }
    g = n;
  });
  const up = (e: PointerEvent) => {
    pts.delete(e.pointerId);
    g = pts.size ? gesture() : null;
    if (pts.size) return;
    map.classList.remove("dragging");
    if (e.type === "pointerup" && down && !down.many && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 8
        && performance.now() - down.t < 500) {
      const x = e.clientX, y = e.clientY;
      clearTimeout(tapTimer);
      tapTimer = window.setTimeout(() => pick(x, y), 260);   // a second tap (zoom) cancels it
    }
    down = null;
  };
  map.addEventListener("pointerup", up);
  map.addEventListener("pointercancel", up);
  map.addEventListener("dblclick", (e) => {
    if (document.body.dataset.view !== "explore") return;
    clearTimeout(tapTimer);
    const r = map.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, 2);
  });
  // Safari: no page zoom while the map is being pinched
  document.addEventListener("gesturestart", (e) => { if (document.body.dataset.view === "explore") e.preventDefault(); });
  const centre = (f: number) => { const r = map.getBoundingClientRect(); zoomAt(innerWidth / 2 - r.left, innerHeight / 2 - r.top, f); };
  $("#ex-in").onclick = () => centre(1.8);
  $("#ex-out").onclick = () => centre(1 / 1.8);
  $("#ex-reset").onclick = () => resetView();
  $("#ex-back").onclick = () => history.back();
  $("#pick-close").onclick = () => closePick();

  // light / dark: follows the device until the reader chooses; the choice is remembered
  const root = document.documentElement;
  $("#theme").onclick = () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch { /* private mode */ }
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    let chosen = null;
    try { chosen = localStorage.getItem("theme"); } catch { /* */ }
    if (!chosen) root.dataset.theme = e.matches ? "dark" : "light";
  });
  // the dots are drawn in the theme's colours: redraw them when it changes
  new MutationObserver(() => { for (const t of built.values()) t.cv.remove(); built.clear(); scheduleTiles(); })
    .observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  addEventListener("resize", () => scheduleTiles());
}

function zoomAt(mx: number, my: number, f: number) {
  const z = Math.min(MAX_ZOOM, Math.max(1, view.z * f));
  if (z === view.z) return apply();
  view.x = mx - (mx - view.x) * (z / view.z);
  view.y = my - (my - view.y) * (z / view.z);
  view.z = z;
  apply();
}
function apply() {
  const w = $("#world");
  w.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.z})`;
  w.style.setProperty("--z", String(view.z));
  w.classList.toggle("zoomed", view.z >= 1.6);
  w.classList.toggle("deep", view.z >= 7);       // field names are too broad there; the image is only haze
  w.classList.toggle("deeper", view.z >= 24);
  document.body.classList.toggle("map-zoomed", view.z >= 1.6);
  lineWidths();
  scheduleTiles();
}
export function resetView() { view.z = 1; view.x = 0; view.y = 0; apply(); }

/** Called by the page on every view change. */
export function mapView(v: string) {
  $("#map-tip").hidden = true;
  if (v === "results") context = "results";
  if (v === "work" || v === "reader") context = "walk";
  if (v !== "explore") { resetView(); closePick(); }
  $("#ex-legend").hidden = !(context === "walk" && walkSpots.length);
  draw();
}

/** What the explorer shows: the last search's results, or the walk around a study. */
export function useContext(c: "results" | "walk") {
  context = c;
  $("#ex-legend").hidden = !(c === "walk" && walkSpots.length);
  draw();
}

/** Where the explorer opens: the study being looked at, else the question. */
export const focusPoint = () => context === "walk" ? (here ?? question) : (question ?? here);

/** Open the explorer centred on a place (the question or a study), zoomed in a little. */
export function focus(p: { x: number; y: number } | null) {
  resetView();
  if (!p) return;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  setTimeout(() => {
    // centre the place on the screen (the map can be larger than the screen on phones)
    const r = $("#map").getBoundingClientRect(), size = r.width, z = 2.5;
    view.z = z;
    view.x = innerWidth / 2 - r.left - p.x * size * z;
    view.y = innerHeight / 2 - r.top - p.y * size * z;
    apply();
  }, still ? 0 : 820);
}

export function setResults(list: Spot[], q: { x: number; y: number } | null) { results = list; question = q; context = "results"; draw(); }
export function setWalk(list: Spot[], lines: Line[]) {
  walkSpots = list; walkLines = lines;
  $("#ex-legend").hidden = !(context === "walk" && list.length);
  draw();
}
export function setHere(p: { x: number; y: number } | null) { here = p; draw(); }
export function hot(id: string, on: boolean) {
  for (const d of document.querySelectorAll(`.dot[data-id="${CSS.escape(id)}"]`)) d.classList.toggle("hot", on);
}

/** Positions go through the CSSOM: the page's policy (style-src 'self') ignores inline style attributes. */
function at<T extends HTMLElement | SVGElement>(el: T, x: number, y: number): T {
  el.style.left = `${x * 100}%`;
  el.style.top = `${y * 100}%`;
  return el;
}

function draw() {
  const box = $("#dots");
  if (!box) return;
  const v = document.body.dataset.view;
  box.replaceChildren();
  drawLines(v);
  if (v === "home") return;
  const walk = v === "work" || (v === "explore" && context === "walk");
  if (question && (v === "results" || (v === "explore" && context === "results"))) {
    const q = at(document.createElement("span"), question.x, question.y);
    q.className = "qmark";
    q.title = "Your question lands here";
    box.appendChild(q);
  }
  const list = v === "results" || (v === "explore" && context === "results") ? results : walk ? walkSpots : [];
  for (const s of list) {
    const el = at(document.createElement("span"), s.x, s.y);
    el.className = `dot k-${s.kind ?? "result"}` + (s.readable ? " readable" : "") + (s.strong ? " strong" : "");
    el.dataset.id = s.id;
    // a mouse only: on touch screens a tap counts as hovering and the popup would stay over the map
    el.onpointerenter = (e) => { if (e.pointerType === "mouse") tip(e, s); };
    el.onpointerleave = () => { $("#map-tip").hidden = true; };
    el.onclick = (e) => { if (document.body.dataset.view === "explore") { e.stopPropagation(); $("#map-tip").hidden = true; openSpot(s); } };
    box.appendChild(el);
  }
  if (here && (v === "work" || v === "reader" || (v === "explore" && context === "walk"))) {
    const h = at(document.createElement("span"), here.x, here.y);
    h.className = "dot here";
    box.appendChild(h);
  }
  if (picked && v === "explore") {
    const p = at(document.createElement("span"), picked.x, picked.y);
    p.className = "picked";
    box.appendChild(p);
  }
}

/** Line widths in map units, so lines stay ~1 px on screen at any zoom (the zoom is a CSS transform, which
 *  SVG's non-scaling-stroke does not see). */
function lineWidths() {
  const svg = document.querySelector("#links") as SVGSVGElement | null;
  if (!svg) return;
  const px = 1 / Math.max(1, ($("#map").offsetWidth || 1) * view.z);
  svg.style.setProperty("--sw", String(1.3 * px));
  svg.style.setProperty("--dash", `${3 * px} ${4 * px}`);
}

function drawLines(v?: string) {
  lineWidths();
  setTimeout(lineWidths, 850);   // after the map has moved to its place for the view
  const svg = document.querySelector("#links") as SVGSVGElement | null;
  if (!svg) return;
  svg.replaceChildren();
  if (!(v === "work" || (v === "explore" && context === "walk"))) return;
  for (const l of walkLines) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", "line");
    e.setAttribute("x1", String(l.x1)); e.setAttribute("y1", String(l.y1));
    e.setAttribute("x2", String(l.x2)); e.setAttribute("y2", String(l.y2));
    e.setAttribute("class", `ln-${l.kind}`);
    svg.appendChild(e);
  }
}

function tip(e: MouseEvent, s: Spot) {
  const v = document.body.dataset.view;
  if (v !== "explore" && v !== "results") return;
  const t = $("#map-tip");
  t.innerHTML = `<b>${esc(s.title ?? "")}</b>${s.meta ? `<br><span>${esc(s.meta)}</span>` : ""}${s.readable ? `<br><span class="ok">Readable in plain words</span>` : ""}`;
  t.hidden = false;
  t.style.left = Math.min(e.clientX + 14, innerWidth - 360) + "px";
  t.style.top = e.clientY + 14 + "px";
}

// ======================================================================= every study, as dots (tiles)
// A quadtree of the release (pipeline/release/tiles.py): level z is a 2^z x 2^z grid; a tile keeps up to
// `capacity` studies not taken by a coarser level. To show level L, a display tile draws its own points and
// those of its ancestors that fall inside it. Zooming in only adds dots; a dot never moves.

let TI: any = null;                     // index.json
let release = "";
let tiLoading: Promise<void> | null = null;
let subField: string[] = [];            // subfield code -> field name
const fieldColour: Record<string, { light: number[]; dark: number[] }> = {};
const ptsCache = new Map<string, Promise<Uint16Array | null>>();
const idsCache = new Map<string, Promise<BigUint64Array | null>>();
interface Built { cv: HTMLCanvasElement; x: Float32Array; y: Float32Array; src: string[]; si: Uint8Array; idx: Uint32Array }
const built = new Map<string, Built>();
const building = new Set<string>();
let tileTimer = 0;

function hexRGB(h?: string): number[] {
  if (!h) return [0.6, 0.6, 0.6];
  return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
}

function loadIndex(): Promise<void> {
  tiLoading ??= (async () => {
    const r = await fetch(`${api}/api/data/map/regions`);
    release = (await r.json()).release;
    if (!release) throw new Error("no release");
    TI = await (await fetch(`${api}/api/data/map/tiles/${release}/index.json`)).json();
    subField = TI.subfields.map((s: any) => s.field);
  })().catch((e) => { tiLoading = null; throw e; });
  return tiLoading;
}

function pts(key: string): Promise<Uint16Array | null> {
  let p = ptsCache.get(key);
  if (!p) {
    p = fetch(`${api}/api/data/map/tiles/${release}/${key}.pts`)
      .then((r) => r.ok ? r.arrayBuffer() : null).then((b) => b ? new Uint16Array(b) : null).catch(() => null);
    ptsCache.set(key, p);
    if (ptsCache.size > 400) ptsCache.delete(ptsCache.keys().next().value!);
  }
  return p;
}

function scheduleTiles() {
  clearTimeout(tileTimer);
  tileTimer = window.setTimeout(updateTiles, 140);
}

async function updateTiles() {
  const layer = $("#tiles");
  if (!layer) return;
  const on = document.body.dataset.view === "explore" && view.z >= TILES_FROM;
  layer.hidden = !on;
  if (!on) return;
  try { await loadIndex(); } catch { return; }
  const map = $("#map"), S = map.offsetWidth, r = map.getBoundingClientRect();
  const L = Math.max(0, Math.min(TI.levels - 1, Math.floor(Math.log2(S * view.z / 420))));
  const n = 1 << L, span = S * view.z;
  const x0 = (-r.left - view.x) / span, x1 = (innerWidth - r.left - view.x) / span;
  const y0 = (-r.top - view.y) / span, y1 = (innerHeight - r.top - view.y) / span;
  const want = new Set<string>();
  for (let ty = Math.max(0, Math.floor(y0 * n)); ty <= Math.min(n - 1, Math.floor(y1 * n)); ty++)
    for (let tx = Math.max(0, Math.floor(x0 * n)); tx <= Math.min(n - 1, Math.floor(x1 * n)); tx++) {
      // something to draw if this square or one of its ancestors holds studies
      for (let l = 0; l <= L; l++) if (`${l}/${tx >> (L - l)}/${ty >> (L - l)}` in TI.tiles) { want.add(`${L}/${tx}/${ty}`); break; }
    }
  for (const [k, t] of built) t.cv.hidden = !want.has(k);
  // forget far-away canvases
  if (built.size > 64) for (const [k, t] of built) if (!want.has(k)) { t.cv.remove(); built.delete(k); if (built.size <= 48) break; }
  for (const k of want) if (!built.has(k) && !building.has(k)) buildTile(k);
}

async function buildTile(key: string) {
  building.add(key);
  try {
    const [L, tx, ty] = key.split("/").map(Number);
    const n = 1 << L, ox = tx / n, oy = ty / n;
    const srcs: { key: string; z: number; sx: number; sy: number; a: Uint16Array }[] = [];
    for (let l = 0; l <= L; l++) {
      const sx = tx >> (L - l), sy = ty >> (L - l), k = `${l}/${sx}/${sy}`;
      if (!(k in TI.tiles)) continue;
      const a = await pts(k);
      if (a) srcs.push({ key: k, z: l, sx, sy, a });
    }
    const dark = document.documentElement.dataset.theme === "dark";
    const C = Math.min(1280, Math.round(640 * Math.min(devicePixelRatio || 1, 2)));
    const acc = new Float32Array(C * C * 4);   // r, g, b, count
    const xs: number[] = [], ys: number[] = [], si: number[] = [], idx: number[] = [];
    const cols = subField.map((f) => (fieldColour[f] ?? { light: [0.45, 0.45, 0.45], dark: [0.7, 0.7, 0.7] })[dark ? "dark" : "light"]);
    const splat = (cx: number, cy: number, c: number[], w: number) => {
      if (cx < 0 || cy < 0 || cx >= C || cy >= C) return;
      const o = (cy * C + cx) * 4;
      acc[o] += c[0] * w; acc[o + 1] += c[1] * w; acc[o + 2] += c[2] * w; acc[o + 3] += w;
    };
    srcs.forEach((s, sIdx) => {
      const m = 1 << s.z, a = s.a;
      for (let i = 0, j = 0; j < a.length; i++, j += 4) {
        const px = (s.sx + a[j] / 65535) / m, py = (s.sy + a[j + 1] / 65535) / m;
        const u = (px - ox) * n, v = (py - oy) * n;
        if (u < 0 || v < 0 || u >= 1 || v >= 1) continue;
        const cx = (u * C) | 0, cy = (v * C) | 0, c = cols[a[j + 3]] ?? [0.6, 0.6, 0.6];
        splat(cx, cy, c, 1);
        splat(cx + 1, cy, c, 0.3); splat(cx - 1, cy, c, 0.3); splat(cx, cy + 1, c, 0.3); splat(cx, cy - 1, c, 0.3);
        xs.push(px); ys.push(py); si.push(sIdx); idx.push(i);
      }
    });
    const cv = document.createElement("canvas");
    cv.width = cv.height = C;
    const ctx = cv.getContext("2d")!;
    const img = ctx.createImageData(C, C), d = img.data;
    for (let p = 0, o = 0; p < C * C; p++, o += 4) {
      const w = acc[o + 3];
      if (!w) continue;
      let r = acc[o] / w, g = acc[o + 1] / w, b = acc[o + 2] / w;
      const a = 1 - Math.exp(-w * (dark ? 1.25 : 1.1));
      if (dark) { const white = Math.min(0.55, Math.max(0, (w - 2.5) / 12)); r += (1 - r) * white; g += (1 - g) * white; b += (1 - b) * white; }
      else { const deep = Math.min(0.35, w / 20); r *= 1 - deep; g *= 1 - deep; b *= 1 - deep; }
      d[o] = r * 255; d[o + 1] = g * 255; d[o + 2] = b * 255; d[o + 3] = a * 255;
    }
    ctx.putImageData(img, 0, 0);
    cv.className = "tile";
    cv.style.left = `${ox * 100}%`; cv.style.top = `${oy * 100}%`;
    cv.style.width = cv.style.height = `${100 / n}%`;
    $("#tiles").appendChild(cv);
    built.set(key, { cv, x: Float32Array.from(xs), y: Float32Array.from(ys), src: srcs.map((s) => s.key),
                     si: Uint8Array.from(si), idx: Uint32Array.from(idx) });
    scheduleTiles();
  } finally {
    building.delete(key);
  }
}

// ----------------------------------------------------------------------- tap a dot: which study is it?
async function pick(cx: number, cy: number) {
  if (document.body.dataset.view !== "explore" || view.z < TILES_FROM || !TI) return;
  const map = $("#map"), S = map.offsetWidth, r = map.getBoundingClientRect(), span = S * view.z;
  const mx = (cx - r.left - view.x) / span, my = (cy - r.top - view.y) / span;
  const reach = 14 / span;
  let best: { t: Built; i: number; d: number } | null = null;
  for (const t of built.values()) {
    if (t.cv.hidden) continue;
    for (let i = 0; i < t.x.length; i++) {
      const dx = t.x[i] - mx, dy = t.y[i] - my;
      if (Math.abs(dx) > reach || Math.abs(dy) > reach) continue;
      const d = dx * dx + dy * dy;
      if (!best || d < best.d) best = { t, i, d };
    }
  }
  if (!best) return closePick();
  const { t, i } = best, src = t.src[t.si[i]];
  picked = { x: t.x[i], y: t.y[i] };
  draw();
  const card = $("#pick");
  card.hidden = false;
  card.querySelector(".pick-body")!.innerHTML = `<span class="quiet">Finding this study…</span>`;
  let ids = idsCache.get(src);
  if (!ids) {
    ids = fetch(`${api}/api/data/map/tiles/${release}/${src}.ids`).then((r) => r.ok ? r.arrayBuffer() : null)
      .then((b) => b ? new BigUint64Array(b) : null).catch(() => null);
    idsCache.set(src, ids);
  }
  const arr = await ids;
  if (!arr) return closePick();
  const w = `W${arr[t.idx[i]]}`;
  try {
    const d = await (await fetch(`${api}/api/data/works/${w}`)).json();
    if (!picked || picked.x !== t.x[i]) return;   // another tap since
    const title = (d.title ?? "").replace(/<[^>]+>/g, "") || "(no title)";
    const meta = [d.venue, d.year].filter(Boolean).join(" · ");
    card.querySelector(".pick-body")!.innerHTML =
      `<b>${esc(title)}</b><span class="quiet">${esc(meta)}${d.smr ? ' · <span class="ok">Readable</span>' : ""}</span>`;
    ($("#pick-open") as HTMLButtonElement).onclick = () => { closePick(); openSpot({ id: w, x: picked?.x ?? 0, y: picked?.y ?? 0, readable: Boolean(d.smr) }); };
  } catch { closePick(); }
}

function closePick() {
  const card = document.querySelector("#pick") as HTMLElement | null;
  if (card) card.hidden = true;
  if (picked) { picked = null; draw(); }
}
