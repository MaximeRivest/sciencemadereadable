/**
 * The map of science, always on screen in some form: a planet rising behind the home page, a quiet globe
 * beside the results (where they glow), a small "you are here" beside a study, a home button in the reader,
 * and the whole map to explore. One element; CSS places it by the view (body[data-view]).
 *
 * Small views draw the rendered images (design/home-mock/make_map.py). The explorer draws the release's map
 * data with WebGL (mapgl.ts): every study's density, the studies themselves as stars (most cited first), and
 * names that never collide (fields, subfields, topics, famous studies). Layers, bottom to top: canvas#gl,
 * #glabels (names, screen space), #world (the images, citation lines, result / walk dots, CSS-transformed).
 * All positions are atlas coordinates in [0, 1]; the data API gives them.
 *
 * In the explorer, markers (results, the walk, "you are here", the picked study) and citation lines are drawn
 * in screen space (#sdots, #slinks), placed every frame with the same camera as the WebGL map. Inside the
 * CSS-scaled #world, Chrome rounds positions at large zooms (up to 400x) and a marker drifted ~50 px away from
 * its study. The small views keep #world (zoom 1, no rounding problem).
 *
 * The camera: view.z (zoom), view.x / view.y (css px offset of the world inside the #map box). One
 * animation loop moves it (glides, flings, flights) and redraws everything in the same frame.
 */
import { MapGL, type Cam } from "./mapgl.ts";

export type Kind = "result" | "ref" | "similar" | "next";
export interface Spot { id: string; x: number; y: number; readable?: boolean; title?: string; meta?: string; kind?: Kind; strong?: boolean }
export interface Line { x1: number; y1: number; x2: number; y2: number; kind: string }

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector(sel) as T;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
const MAX_ZOOM = 400;          // deep enough to separate single studies in the densest clusters
const still = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

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
const fieldColour: Record<string, { light: number[]; dark: number[] }> = {};

let engine: MapGL | null = null;
let base = "";

export function initMap(opts: { api: string; explore: () => void; open: (s: Spot) => void }) {
  api = opts.api;
  openSpot = opts.open;
  const map = $("#map");
  // sharp images after the page is up, the current theme first
  const order = document.documentElement.dataset.theme === "dark" ? ["dark", "light"] : ["light", "dark"];
  const first = new URLSearchParams(location.search).has("map") ? 0 : 400;
  order.forEach((th, n) => setTimeout(() => {
    const im = new Image();
    im.onload = () => { ($(`.map-img.${th}`) as HTMLImageElement).src = im.src; };
    im.src = `map/science-map-${th}.webp`;
  }, first + n * 1500));
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
    palette();
  }).catch(() => {});

  map.addEventListener("click", () => { if (document.body.dataset.view !== "explore") opts.explore(); });
  map.addEventListener("keydown", (e) => {
    if (document.body.dataset.view !== "explore") { if (e.key === "Enter") opts.explore(); return; }
    const r = map.getBoundingClientRect(), step = 80;
    const mid = () => [innerWidth / 2 - r.left, innerHeight / 2 - r.top] as const;
    if (e.key === "ArrowLeft") glide(step, 0); else if (e.key === "ArrowRight") glide(-step, 0);
    else if (e.key === "ArrowUp") glide(0, step); else if (e.key === "ArrowDown") glide(0, -step);
    else if (e.key === "+" || e.key === "=") smoothZoom(...mid(), 2); else if (e.key === "-") smoothZoom(...mid(), 0.5);
    else return;
    e.preventDefault();
  });
  const onWheel = (e: WheelEvent) => {
    if (document.body.dataset.view !== "explore") return;
    e.preventDefault();
    const r = map.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    const dy = e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY;
    // trackpads (small steps, or a pinch: ctrlKey) follow the fingers; a mouse wheel glides
    if (e.ctrlKey || Math.abs(dy) < 40) { stopMotion(); zoomAt(mx, my, Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.004))); }
    else smoothZoom(mx, my, Math.exp(-dy * 0.0028));
  };
  map.addEventListener("wheel", onWheel, { passive: false });
  $("#sdots").addEventListener("wheel", onWheel, { passive: false });

  // one finger (or the mouse) moves, two fingers pinch and move together; a tap identifies a study or
  // follows a name; a double tap zooms in; a flick keeps the map moving
  const pts = new Map<number, { x: number; y: number }>();
  let g: { cx: number; cy: number; d: number } | null = null;
  let down: { x: number; y: number; t: number; many: boolean; label: HTMLElement | null } | null = null;
  let tapTimer = 0;
  const track: { x: number; y: number; t: number }[] = [];
  const gesture = () => {
    const a = [...pts.values()], r = map.getBoundingClientRect();
    const cx = a.reduce((s, p) => s + p.x, 0) / a.length - r.left, cy = a.reduce((s, p) => s + p.y, 0) / a.length - r.top;
    return { cx, cy, d: a.length > 1 ? Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y) : 0 };
  };
  map.addEventListener("pointerdown", (e) => {
    if (document.body.dataset.view !== "explore" || (e.target as HTMLElement).classList.contains("dot")) return;
    stopMotion();
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const label = (e.target as HTMLElement).closest<HTMLElement>(".gl-label");
    down = pts.size === 1 ? { x: e.clientX, y: e.clientY, t: performance.now(), many: false, label } : down && { ...down, many: true };
    track.length = 0;
    try { map.setPointerCapture(e.pointerId); } catch { /* */ }
    map.classList.add("dragging");
    g = gesture();
  });
  map.addEventListener("pointermove", (e) => {
    if (!pts.has(e.pointerId)) { if (e.pointerType === "mouse" && !e.buttons) hover(e.clientX, e.clientY); return; }
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const n = gesture();
    if (g) {
      view.x += n.cx - g.cx; view.y += n.cy - g.cy;
      if (n.d && g.d) zoomAt(n.cx, n.cy, n.d / g.d); else apply();
    }
    if (pts.size === 1) { track.push({ x: e.clientX, y: e.clientY, t: performance.now() }); if (track.length > 8) track.shift(); }
    g = n;
  });
  const up = (e: PointerEvent) => {
    pts.delete(e.pointerId);
    g = pts.size ? gesture() : null;
    if (pts.size) return;
    map.classList.remove("dragging");
    const tap = e.type === "pointerup" && down && !down.many && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 8
      && performance.now() - down.t < 500;
    if (tap && down!.label) { const l = down!.label; down = null; return followLabel(l); }
    if (tap) {
      const x = e.clientX, y = e.clientY;
      clearTimeout(tapTimer);
      tapTimer = window.setTimeout(() => pick(x, y), 260);   // a second tap (zoom) cancels it
    } else if (down && !down.many && track.length > 2 && !still()) {
      // a flick: keep going at the release speed, slowing down
      const a = track[0], b = track[track.length - 1], dt = b.t - a.t;
      if (dt > 0 && performance.now() - b.t < 60) {
        const vx = (b.x - a.x) / dt, vy = (b.y - a.y) / dt;
        if (Math.hypot(vx, vy) > 0.25) motion = { kind: "fling", vx, vy, t: performance.now() };
        frame();
      }
    }
    down = null;
  };
  map.addEventListener("pointerup", up);
  map.addEventListener("pointercancel", up);
  map.addEventListener("pointerleave", () => hoverRing(null));
  map.addEventListener("dblclick", (e) => {
    if (document.body.dataset.view !== "explore") return;
    clearTimeout(tapTimer);
    const r = map.getBoundingClientRect();
    smoothZoom(e.clientX - r.left, e.clientY - r.top, 2.5);
  });
  // Safari: no page zoom while the map is being pinched
  document.addEventListener("gesturestart", (e) => { if (document.body.dataset.view === "explore") e.preventDefault(); });
  const centre = (f: number) => { const r = map.getBoundingClientRect(); smoothZoom(innerWidth / 2 - r.left, innerHeight / 2 - r.top, f); };
  $("#ex-in").onclick = () => centre(2);
  $("#ex-out").onclick = () => centre(0.5);
  $("#ex-reset").onclick = () => flyTo(0.5, 0.5, 1);
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
  new MutationObserver(() => { palette(); frame(); }).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  addEventListener("resize", () => frame());
  startEngine();
  // for the browser check (tools/check_site.py) and screenshots
  (window as any).__fly = (x: number, y: number, z: number) => { motion = null; Object.assign(view, viewFor(x, y, z)); apply(); };
  (window as any).__expose = (k: number, st?: number, gr?: number) => { if (engine) { engine.EXPOSURE = k > 0 ? k : null; if (st != null) engine.STARS = st; if (gr != null) engine.GRAIN = gr; engine.lastStats = 0; engine.k = 0; frame(); } };
  (window as any).__mapStats = () => ({ z: view.z, stars: engine ? [...engine.ptiles.values()].reduce((s, t) => s + t.shown, 0) : 0,
                                        density: engine?.dtiles.size ?? 0, names: labelEls.size, complete: engine?.complete ?? false });
}

// ======================================================================= the engine
function startEngine() {
  const cv = $("#gl") as HTMLCanvasElement | null;
  // older browsers (no WebGL2, or no gzip streams: iOS < 16.4) keep the rendered images
  if (!cv || typeof DecompressionStream === "undefined") return;
  try { engine = new MapGL(cv); } catch { engine = null; return; }   // no WebGL2: the images stay
  engine.onChange = () => frame();
  cv.addEventListener("webglcontextlost", (e) => { e.preventDefault(); engine = null; document.body.classList.remove("gl"); });
  (async () => {
    const r = await fetch(`${api}/api/data/map/regions`);
    const rel = (await r.json()).release;
    if (!rel) throw new Error("no release");
    base = `${api}/api/data/map/tiles/${rel}/v2`;
    // names in parallel with the index; the rendered image's own names stay until these arrive
    labelsReady = fetch(`${base}/labels.json`).then((r) => r.json())
      .then((l: Label[]) => { labels = l; document.body.classList.add("gl-names"); frame(); }).catch(() => {});
    await engine!.load(base);
    palette();
    glReady = true;
    mapView(document.body.dataset.view ?? "home");
  })().catch(() => { engine = null; });
}
let glReady = false;

function palette() {
  if (!engine) return;
  const dark = document.documentElement.dataset.theme === "dark";
  const bg = hexRGB(getComputedStyle(document.documentElement).getPropertyValue("--explore").trim() || (dark ? "#10100f" : "#f2eee5"));
  const fields: Record<string, number[]> = {};
  for (const [n, c] of Object.entries(fieldColour)) fields[n] = dark ? c.dark : c.light;
  engine.setPalette({ dark, bg, fields });
}

const explore = () => document.body.dataset.view === "explore";
const glOn = () => !!engine && glReady && explore();

// ----------------------------------------------------------------------- camera
type Motion = { kind: "zoom"; mx: number; my: number; target: number }
  | { kind: "fling"; vx: number; vy: number; t: number }
  | { kind: "fly"; from: { x: number; y: number; z: number }; to: { x: number; y: number; z: number }; t0: number; ms: number }
  | { kind: "glide"; dx: number; dy: number; left: number };
let motion: Motion | null = null;
let raf = 0, settleUntil = 0, lastT = 0;

function stopMotion() { motion = null; }

function zoomAt(mx: number, my: number, f: number) {
  const z = Math.min(MAX_ZOOM, Math.max(1, view.z * f));
  if (z !== view.z) {
    view.x = mx - (mx - view.x) * (z / view.z);
    view.y = my - (my - view.y) * (z / view.z);
    view.z = z;
  }
  apply();
}

function smoothZoom(mx: number, my: number, f: number) {
  if (still()) return zoomAt(mx, my, f);
  const cur = motion?.kind === "zoom" ? motion.target : view.z;
  motion = { kind: "zoom", mx, my, target: Math.min(MAX_ZOOM, Math.max(1, cur * f)) };
  frame();
}
function glide(dx: number, dy: number) { motion = { kind: "glide", dx, dy, left: 1 }; frame(); }

/** World point at the centre of the screen, at zoom z: the view that puts it there. */
function viewFor(x: number, y: number, z: number) {
  const r = $("#map").getBoundingClientRect(), S = r.width;
  return { z, x: innerWidth / 2 - r.left - x * S * z, y: innerHeight / 2 - r.top - y * S * z };
}
function centreNow() {
  const r = $("#map").getBoundingClientRect(), S = r.width * view.z;
  return { x: (innerWidth / 2 - r.left - view.x) / S, y: (innerHeight / 2 - r.top - view.y) / S };
}

/** Fly to a world point: out a little when far, then in (zoom and pan feel like one movement). */
export function flyTo(x: number, y: number, z: number) {
  z = Math.min(MAX_ZOOM, Math.max(1, z));
  if (still()) { Object.assign(view, viewFor(x, y, z)); return apply(); }
  const c = centreNow();
  motion = { kind: "fly", from: { ...c, z: view.z }, to: { x, y, z }, t0: performance.now(), ms: 0 };
  const S = $("#map").offsetWidth, d = Math.hypot(x - c.x, y - c.y) * S * Math.min(view.z, z);
  motion.ms = Math.min(1600, 520 + 140 * Math.log2(1 + d / 300) + 60 * Math.abs(Math.log2(z / view.z)));
  frame();
}

function apply() { frame(); }

/** One frame: move the camera one step, place the world, draw the map and the names. */
function frame() {
  if (raf) return;
  raf = requestAnimationFrame((t) => { raf = 0; tick(t); });
}
function tick(t: number) {
  const dt = Math.min(64, lastT ? t - lastT : 16); lastT = t;
  let moving = false;
  const map = $("#map");
  if (motion?.kind === "zoom") {
    const k = 1 - Math.exp(-dt / 70), f = Math.exp(Math.log(motion.target / view.z) * k);
    zoomRaw(motion.mx, motion.my, f);
    if (Math.abs(Math.log(motion.target / view.z)) < 0.002) motion = null; else moving = true;
  } else if (motion?.kind === "fling") {
    const decay = Math.exp(-dt / 280);
    view.x += motion.vx * dt; view.y += motion.vy * dt;
    motion.vx *= decay; motion.vy *= decay;
    if (Math.hypot(motion.vx, motion.vy) < 0.02) motion = null; else moving = true;
  } else if (motion?.kind === "glide") {
    const k = 1 - Math.exp(-dt / 60), s = motion.left * k;
    view.x += motion.dx * s; view.y += motion.dy * s; motion.left -= s;
    if (motion.left < 0.01) motion = null; else moving = true;
  } else if (motion?.kind === "fly") {
    const m = motion, u = Math.min(1, (t - m.t0) / m.ms), e = u < .5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2;
    const S = map.offsetWidth, d = Math.hypot(m.to.x - m.from.x, m.to.y - m.from.y) * S;
    // out by up to a few steps when the target is far off screen at the zooms involved
    const bump = Math.max(0, Math.log2(d * Math.min(m.from.z, m.to.z) / Math.max(innerWidth, 1)) ) * 0.9;
    const lz = Math.log2(m.from.z) + (Math.log2(m.to.z) - Math.log2(m.from.z)) * e - bump * Math.sin(Math.PI * u) ;
    const z = Math.min(MAX_ZOOM, Math.max(1, 2 ** lz));
    Object.assign(view, viewFor(m.from.x + (m.to.x - m.from.x) * e, m.from.y + (m.to.y - m.from.y) * e, z));
    if (u >= 1) motion = null; else moving = true;
  }
  clampView();
  place();
  if (moving || t < settleUntil) frame();
}

function zoomRaw(mx: number, my: number, f: number) {
  const z = Math.min(MAX_ZOOM, Math.max(1, view.z * f));
  view.x = mx - (mx - view.x) * (z / view.z);
  view.y = my - (my - view.y) * (z / view.z);
  view.z = z;
}

/** Keep part of the map on screen. */
function clampView() {
  if (!explore()) return;
  const r = $("#map").getBoundingClientRect(), S = r.width * view.z, m = 0.35;
  const minX = innerWidth * m - r.left - S, maxX = innerWidth * (1 - m) - r.left;
  const minY = innerHeight * m - r.top - S, maxY = innerHeight * (1 - m) - r.top;
  view.x = Math.min(maxX, Math.max(minX, view.x));
  view.y = Math.min(maxY, Math.max(minY, view.y));
}

function place() {
  const w = $("#world");
  w.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.z})`;
  w.style.setProperty("--z", String(view.z));
  w.classList.toggle("zoomed", view.z >= 1.6);
  w.classList.toggle("deep", view.z >= 7);
  w.classList.toggle("deeper", view.z >= 24);
  document.body.classList.toggle("map-zoomed", view.z >= 1.6);
  document.body.classList.toggle("gl", glOn());
  lineWidths();
  placeMarks();
  if (glOn()) {
    const r = $("#map").getBoundingClientRect();
    const cam: Cam = { S: r.width, ox: r.left + view.x, oy: r.top + view.y, z: view.z };
    engine!.draw(cam, innerWidth, innerHeight);
    drawLabels(cam);
    // the rendered image stays on screen until the live map is final, then the live map fades in over it
    // (once per visit of the explorer; after 8 s it shows anyway, e.g. on a slow connection,
    // (or as soon as you zoom past what the image's 1,400 pixels can show sharply)
    if (!revealed && (engine!.complete || view.z > 3.5 || performance.now() - exploreSince > 8000)) {
      revealed = true;
      document.body.classList.add("gl-ready");
    }
  }
}

let revealed = false, exploreSince = 0;

export function resetView() { motion = null; view.z = 1; view.x = 0; view.y = 0; apply(); }

/** Called by the page on every view change. */
export function mapView(v: string) {
  $("#map-tip").hidden = true;
  if (v === "results") context = "results";
  if (v === "work" || v === "reader") context = "walk";
  if (v !== "explore") { resetView(); closePick(); hoverRing(null); revealed = false; document.body.classList.remove("gl-ready"); }
  else if (!revealed) exploreSince = performance.now();
  $("#ex-legend").hidden = !(context === "walk" && walkSpots.length);
  settleUntil = performance.now() + 950;   // the box moves to its place for the view (CSS, .8 s): follow it
  draw();
  frame();
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
  setTimeout(() => flyTo(p.x, p.y, 3), still() ? 0 : 820);
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

/** Where markers go: screen space in the explorer, the CSS world elsewhere. */
const marks: { el: HTMLElement; x: number; y: number }[] = [];
function put<T extends HTMLElement>(el: T, x: number, y: number): T {
  if (explore()) marks.push({ el, x, y }); else at(el, x, y);
  return el;
}

function draw() {
  const world = $("#dots"), screen = $("#sdots");
  if (!world || !screen) return;
  const v = document.body.dataset.view;
  world.replaceChildren(); screen.replaceChildren(); marks.length = 0;
  const box = explore() ? screen : world;
  drawLines(v);
  if (v === "home") return;
  const walk = v === "work" || (v === "explore" && context === "walk");
  if (question && (v === "results" || (v === "explore" && context === "results"))) {
    const q = put(document.createElement("span"), question.x, question.y);
    q.className = "qmark";
    q.title = "Your question lands here";
    box.appendChild(q);
  }
  const list = v === "results" || (v === "explore" && context === "results") ? results : walk ? walkSpots : [];
  for (const s of list) {
    const el = put(document.createElement("span"), s.x, s.y);
    el.className = `dot k-${s.kind ?? "result"}` + (s.readable ? " readable" : "") + (s.strong ? " strong" : "");
    el.dataset.id = s.id;
    // a mouse only: on touch screens a tap counts as hovering and the popup would stay over the map
    el.onpointerenter = (e) => { if (e.pointerType === "mouse") tip(e, s); };
    el.onpointerleave = () => { $("#map-tip").hidden = true; };
    el.onclick = (e) => { if (document.body.dataset.view === "explore") { e.stopPropagation(); $("#map-tip").hidden = true; openSpot(s); } };
    box.appendChild(el);
  }
  if (here && (v === "work" || v === "reader" || (v === "explore" && context === "walk"))) {
    const h = put(document.createElement("span"), here.x, here.y);
    h.className = "dot here";
    box.appendChild(h);
  }
  if (picked && v === "explore") {
    const p = put(document.createElement("span"), picked.x, picked.y);
    p.className = "picked";
    box.appendChild(p);
  }
  placeMarks();
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

const lineEls: { el: SVGLineElement; l: Line }[] = [];
function drawLines(v?: string) {
  lineWidths();
  const svg = document.querySelector("#links") as SVGSVGElement | null, ssvg = document.querySelector("#slinks") as SVGSVGElement | null;
  if (!svg || !ssvg) return;
  svg.replaceChildren(); ssvg.replaceChildren(); lineEls.length = 0;
  if (!(v === "work" || (v === "explore" && context === "walk"))) return;
  const into = v === "explore" ? ssvg : svg;
  for (const l of walkLines) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", "line");
    if (v === "explore") lineEls.push({ el: e, l });
    else { e.setAttribute("x1", String(l.x1)); e.setAttribute("y1", String(l.y1)); e.setAttribute("x2", String(l.x2)); e.setAttribute("y2", String(l.y2)); }
    e.setAttribute("class", `ln-${l.kind}`);
    into.appendChild(e);
  }
}

/** Screen-space markers and lines follow the camera (same formula as the WebGL map). */
function placeMarks() {
  if (!explore()) return;
  const r = $("#map").getBoundingClientRect(), span = r.width * view.z, ox = r.left + view.x, oy = r.top + view.y;
  for (const m of marks) m.el.style.translate = `${ox + m.x * span}px ${oy + m.y * span}px`;
  for (const { el, l } of lineEls) {
    el.setAttribute("x1", (ox + l.x1 * span).toFixed(1)); el.setAttribute("y1", (oy + l.y1 * span).toFixed(1));
    el.setAttribute("x2", (ox + l.x2 * span).toFixed(1)); el.setAttribute("y2", (oy + l.y2 * span).toFixed(1));
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

function hexRGB(h?: string): number[] {
  if (!h || h[0] !== "#") return [0.6, 0.6, 0.6];
  if (h.length === 4) h = "#" + [...h.slice(1)].map((c) => c + c).join("");
  return [1, 3, 5].map((i) => parseInt(h!.slice(i, i + 2), 16) / 255);
}

// ======================================================================= names on the map
// labels.json (mapdata.py): each name shows while s0 <= s < s1, s = screen px per unit square / 1024; they
// were placed so that names showing at the same zoom never overlap.
interface Label { t: string; k: "field" | "subfield" | "topic" | "paper"; x: number; y: number; s0: number; s1: number; m: number; f: number; id?: string; full?: string }
let labels: Label[] = [];
let labelsReady: Promise<void> | null = null;
const labelEls = new Map<number, HTMLElement>();

function drawLabels(c: Cam) {
  const box = $("#glabels");
  if (!box) return;
  const s = c.S * c.z / 1024, W = innerWidth, H = innerHeight, span = c.S * c.z;
  const dark = document.documentElement.dataset.theme === "dark";
  const seen = new Set<number>();
  for (let i = 0; i < labels.length; i++) {
    const l = labels[i];
    if (s < l.s0 || s >= l.s1) continue;
    const x = c.ox + l.x * span, y = c.oy + l.y * span;
    if (x < -300 || y < -40 || x > W + 300 || y > H + 40) continue;
    const a = Math.min(1, (s / l.s0 - 1) / 0.18 + 0.15, (l.s1 / s - 1) / 0.18);
    if (a <= 0.02) continue;
    seen.add(i);
    let el = labelEls.get(i);
    if (!el) {
      el = document.createElement("span");
      el.className = `gl-label k-${l.k}`;
      el.textContent = l.t;
      el.dataset.i = String(i);
      if (l.full) el.title = l.full;
      box.appendChild(el);
      labelEls.set(i, el);
    }
    el.style.opacity = String(a);
    el.style.transform = l.k === "paper" ? `translate(${x}px, ${y + 7}px) translate(-50%, 0)` : `translate(${x}px, ${y}px) translate(-50%, -50%)`;
  }
  for (const [i, el] of labelEls) if (!seen.has(i)) { el.remove(); labelEls.delete(i); }
}

const colCache = new Map<string, string>();
function colourOf(f: number, dark: boolean) {
  const k = `${f}:${dark}`;
  let c = colCache.get(k);
  if (c === undefined) {
    const name = engine?.meta?.fields?.[f]?.name, rgb = name && fieldColour[name]?.[dark ? "dark" : "light"];
    c = rgb ? `rgb(${rgb.map((v: number) => Math.round(v * 255)).join(",")})` : "";
    colCache.set(k, c);
  }
  return c;
}

function followLabel(el: HTMLElement) {
  const l = labels[Number(el.dataset.i)];
  if (!l) return;
  const S = $("#map").offsetWidth;
  if (l.k === "paper" && l.id) {
    picked = { x: l.x, y: l.y };
    draw();
    showCard(l.id, l.x, l.y);
    return;
  }
  // into the group: far enough that its own topics (or studies) start to show
  const s = Math.min(l.s1 * 0.6, Math.max(l.s0 * 3, l.k === "topic" ? 14 : 2));
  flyTo(l.x, l.y, s * 1024 / S);
}

// ======================================================================= which study is it?
function hover(cx: number, cy: number) {
  if (!glOn() || view.z < 1.3) return hoverRing(null);
  const r = $("#map").getBoundingClientRect();
  const s = engine!.nearest({ S: r.width, ox: r.left + view.x, oy: r.top + view.y, z: view.z }, cx, cy, 9);
  hoverRing(s ? { x: r.left + view.x + s.x * r.width * view.z, y: r.top + view.y + s.y * r.width * view.z } : null);
}
function hoverRing(p: { x: number; y: number } | null) {
  const el = document.querySelector("#hover-ring") as HTMLElement | null;
  if (!el) return;
  el.hidden = !p;
  $("#map").classList.toggle("on-star", !!p);
  if (p) el.style.transform = `translate(${p.x}px, ${p.y}px)`;
}

async function pick(cx: number, cy: number) {
  if (!glOn()) return;
  const r = $("#map").getBoundingClientRect();
  const s = engine!.nearest({ S: r.width, ox: r.left + view.x, oy: r.top + view.y, z: view.z }, cx, cy, 16);
  if (!s) return closePick();
  picked = { x: s.x, y: s.y };
  draw();
  const card = $("#pick");
  card.hidden = false;
  card.querySelector(".pick-body")!.innerHTML = `<span class="quiet">Finding this study…</span>`;
  const ids = await idsOf(s.key);
  if (!ids || !picked || picked.x !== s.x) return ids ? undefined : closePick();
  showCard(`W${ids[s.i]}`, s.x, s.y);
}

const idsCache = new Map<string, Promise<BigUint64Array | null>>();
function idsOf(key: string) {
  let p = idsCache.get(key);
  if (!p) {
    p = fetch(`${base}/i/${key}.bin`).then((r) => r.ok ? r.arrayBuffer() : null).then((b) => b ? new BigUint64Array(b) : null).catch(() => null);
    idsCache.set(key, p);
    if (idsCache.size > 60) idsCache.delete(idsCache.keys().next().value!);
  }
  return p;
}

async function showCard(w: string, x: number, y: number) {
  const card = $("#pick");
  card.hidden = false;
  card.querySelector(".pick-body")!.innerHTML = `<span class="quiet">Finding this study…</span>`;
  try {
    const d = await (await fetch(`${api}/api/data/works/${w}`)).json();
    if (!picked || picked.x !== x) return;   // another tap since
    const title = (d.title ?? "").replace(/<[^>]+>/g, "") || "(no title)";
    const meta = [d.venue, d.year, d.citations ? `cited ${Number(d.citations).toLocaleString()} times` : ""].filter(Boolean).join(" · ");
    card.querySelector(".pick-body")!.innerHTML =
      `<b>${esc(title)}</b><span class="quiet">${esc(meta)}${d.smr ? ' · <span class="ok">Readable</span>' : ""}</span>`;
    ($("#pick-open") as HTMLButtonElement).onclick = () => { closePick(); openSpot({ id: w, x, y, readable: Boolean(d.smr) }); };
  } catch { closePick(); }
}

function closePick() {
  const card = document.querySelector("#pick") as HTMLElement | null;
  if (card) card.hidden = true;
  if (picked) { picked = null; draw(); }
}
