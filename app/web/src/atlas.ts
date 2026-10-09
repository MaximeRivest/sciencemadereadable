/**
 * The map of science, always on screen in some form: a planet rising behind the home page, a quiet globe
 * beside the results (where they glow), a small "you are here" beside a study, a home button in the reader,
 * and the whole map to explore. One element; CSS places it by the view (body[data-view]).
 * The images are rendered from the atlas (design/home-mock/make_map.py); positions come from the data API.
 */

export interface Spot { id: string; x: number; y: number; readable?: boolean; title?: string; meta?: string }

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector(sel) as T;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

let spots: Spot[] = [];
let question: { x: number; y: number } | null = null;
let here: { x: number; y: number } | null = null;
let openSpot: (s: Spot) => void = () => {};
const view = { z: 1, x: 0, y: 0 };

export function initMap(opts: { explore: () => void; open: (s: Spot) => void }) {
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
  // one finger (or the mouse) moves, two fingers pinch and move together; double tap zooms in
  const pts = new Map<number, { x: number; y: number }>();
  let g: { cx: number; cy: number; d: number } | null = null;
  const gesture = () => {
    const a = [...pts.values()], r = map.getBoundingClientRect();
    const cx = a.reduce((s, p) => s + p.x, 0) / a.length - r.left, cy = a.reduce((s, p) => s + p.y, 0) / a.length - r.top;
    return { cx, cy, d: a.length > 1 ? Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y) : 0 };
  };
  map.addEventListener("pointerdown", (e) => {
    if (document.body.dataset.view !== "explore" || (e.target as HTMLElement).classList.contains("dot")) return;
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
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
    if (!pts.size) map.classList.remove("dragging");
  };
  map.addEventListener("pointerup", up);
  map.addEventListener("pointercancel", up);
  map.addEventListener("dblclick", (e) => {
    if (document.body.dataset.view !== "explore") return;
    const r = map.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, 2);
  });
  // Safari: no page zoom while the map is being pinched
  document.addEventListener("gesturestart", (e) => { if (document.body.dataset.view === "explore") e.preventDefault(); });
  const centre = (f: number) => { const r = map.getBoundingClientRect(); zoomAt(innerWidth / 2 - r.left, innerHeight / 2 - r.top, f); };
  $("#ex-in").onclick = () => centre(1.6);
  $("#ex-out").onclick = () => centre(1 / 1.6);
  $("#ex-reset").onclick = () => resetView();
  $("#ex-back").onclick = () => history.back();

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
}

function zoomAt(mx: number, my: number, f: number) {
  const z = Math.min(12, Math.max(1, view.z * f));
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
}
export function resetView() { view.z = 1; view.x = 0; view.y = 0; apply(); }

/** Called by the page on every view change. */
export function mapView(v: string) {
  $("#map-tip").hidden = true;
  if (v !== "explore") resetView();
  draw();
}

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

export function setResults(list: Spot[], q: { x: number; y: number } | null) { spots = list; question = q; draw(); }
export function setHere(p: { x: number; y: number } | null) { here = p; draw(); }
export const currentQuestion = () => question;
export const currentHere = () => here;
export function hot(id: string, on: boolean) { document.querySelector(`.dot[data-id="${CSS.escape(id)}"]`)?.classList.toggle("hot", on); }

/** Positions go through the CSSOM: the page's policy (style-src 'self') ignores inline style attributes. */
function at<T extends HTMLElement>(el: T, x: number, y: number): T {
  el.style.left = `${x * 100}%`;
  el.style.top = `${y * 100}%`;
  return el;
}

function draw() {
  const box = $("#dots");
  if (!box) return;
  const v = document.body.dataset.view;
  box.replaceChildren();
  if (v === "home") return;
  if (question && (v === "results" || v === "explore")) {
    const q = at(document.createElement("span"), question.x, question.y);
    q.className = "qmark";
    q.title = "Your question lands here";
    box.appendChild(q);
  }
  if (v === "results" || v === "explore" || v === "work")
    for (const s of spots) {
      const el = at(document.createElement("span"), s.x, s.y);
      el.className = "dot" + (s.readable ? " readable" : "");
      el.dataset.id = s.id;
      // a mouse only: on touch screens a tap counts as hovering and the popup would stay over the map
      el.onpointerenter = (e) => { if (e.pointerType === "mouse") tip(e, s); };
      el.onpointerleave = () => { $("#map-tip").hidden = true; };
      el.onclick = (e) => { if (document.body.dataset.view === "explore") { e.stopPropagation(); $("#map-tip").hidden = true; openSpot(s); } };
      box.appendChild(el);
    }
  if (here && (v === "work" || v === "reader" || v === "explore")) {
    const h = at(document.createElement("span"), here.x, here.y);
    h.className = "dot here";
    box.appendChild(h);
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
