/**
 * A topic map on one screen. Top left, the field as boxes: each research group a box sized by its studies, its
 * subgroups the boxes inside it (the data API makes them with the map). Top right, the map of science (the page's
 * own globe, placed over an empty slot here). Under the globe, what the box pointed at (or picked) is about: its
 * word cloud, then its description, name and size at the bottom, in rows of fixed size; under the boxes, the
 * picked one's reading lists.
 *
 * Pointing at a box previews it (the globe lights its studies, the panel under the globe shows its words); clicking
 * keeps it, and a group with subgroups zooms in: its box grows to fill the space while its subgroups spread out to
 * be read, and "↑" goes back up. Layout: squarified treemap (Bruls, Huizing & van Wijk 2000), boxes as close to
 * squares as the sizes allow, biggest first, so sizes compare and names fit; laid out again for the zoomed group.
 */
import type { CloudWord, Deps, Group, Lenses, Sub, TopicMap } from "./topics.ts";

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

export interface Parts {
  cloud: (w: CloudWord[], k: number, vs: string) => HTMLElement;
  lists: (host: HTMLElement, key: string, l: Lenses, unit: string, names: Map<number, string>, deps: Deps,
          want: string, onPick: (lens: string) => void) => void;
  span: (y?: number[] | null) => string;
  info: HTMLElement;                 // what "?" opens: how the map was made, its limits, other ways to see it
  act: HTMLElement | null;           // one link in the bar (map the whole field, back to the whole map...)
  splitFrom: number;                 // a group this big can be mapped again on its own
}

interface Rect { x: number; y: number; w: number; h: number }

/** Squarified treemap: values (largest first) into rectangles filling r, in the same order. */
export function squarify(values: number[], r: Rect): Rect[] {
  const total = values.reduce((a, b) => a + b, 0);
  const out: Rect[] = new Array(values.length);
  if (!total) return values.map(() => ({ x: r.x, y: r.y, w: 0, h: 0 }));
  const area = values.map((v) => v / total * r.w * r.h);
  let { x, y, w, h } = r, i = 0;
  while (i < area.length) {
    const side = Math.min(w, h);
    // the worst aspect ratio of a row with this sum, largest and smallest box
    const worst = (s: number, big: number, small: number) => Math.max(side * side * big / (s * s), s * s / (side * side * small));
    let n = 1, sum = area[i], best = worst(sum, area[i], area[i]);
    while (i + n < area.length) {
      const next = worst(sum + area[i + n], area[i], area[i + n]);
      if (next > best) break;
      sum += area[i + n]; best = next; n++;
    }
    const thick = sum / side;
    let off = 0;
    for (let j = i; j < i + n; j++) {
      const len = area[j] / thick;
      out[j] = w >= h ? { x, y: y + off, w: thick, h: len } : { x: x + off, y, w: len, h: thick };
      off += len;
    }
    if (w >= h) { x += thick; w -= thick; } else { y += thick; h -= thick; }
    i += n;
  }
  return out;
}

const HEAD = 24;    // px: a group box's name strip, above its subgroups (style.css .tov-head)
const HEAD_Z = 32;  // px: the same, for the group zoomed into (.tov-g.z .tov-head)
const LINE = 14.4;  // px: a line of a subgroup's name (.tov-s: 11.5px × 1.25)
const LINE_Z = 16.3; // px: the same, zoomed in (.tov-g.z .tov-s: 13px × 1.25)
const ANIM = 480;   // ms: the zoom (.tov-map.anim)

/** A name gets the whole lines its box has room for (then "…"), never a line cut in half. */
function clamp(el: HTMLElement, px: number, line = LINE) {
  const name = el.querySelector<HTMLElement>(".tov-name");
  if (name) name.style.setProperty("-webkit-line-clamp", String(Math.max(1, Math.floor(px / line))));
}
const place = (el: HTMLElement, r: Rect, ow: number, oh: number) => {   // in % of the parent: follows its size
  el.style.left = `${100 * r.x / ow}%`; el.style.top = `${100 * r.y / oh}%`;
  el.style.width = `${100 * r.w / ow}%`; el.style.height = `${100 * r.h / oh}%`;
};
const colour = (id: number) => `g${((id - 1) % 9) + 1}`;

/** Where things stand: the whole (g null), a group, or one of its subgroups. */
type At = { g: number | null; s: number | null };
const parse = (at: string | undefined, d: TopicMap): At => {
  const [a, b] = (at ?? "").split(".").map(Number);
  const g = d.groups.find((x) => x.id === a);
  if (!g) return { g: null, s: null };
  return { g: g.id, s: g.subgroups?.some((x) => x.id === b) ? b : null };
};
const atStr = (a: At) => a.g == null ? "" : a.s == null ? `${a.g}` : `${a.g}.${a.s}`;
const same = (a: At, b: At) => a.g === b.g && a.s === b.s;

interface Tile { g: Group | null; size: number; box: HTMLElement; head?: HTMLButtonElement; inner?: HTMLElement;
                 kids: { s: Sub | null; size: number; el: HTMLElement }[] }

let stop: (() => void) | null = null;   // the shown dashboard's listeners, removed when another replaces it

export function renderOverview(box: HTMLElement, d: TopicMap, deps: Deps, parts: Parts) {
  stop?.();
  const whole = d.parent ? `All of group ${d.parent.group}` : d.mode === "field" ? "Whole field" : "Whole topic";
  const tv = box.appendChild(document.createElement("div"));
  tv.className = "tv";
  tv.innerHTML = `<h2 class="sr-only">Research groups: ${esc(d.question)}</h2>` +
    `<div class="tv-left"><div class="tv-bar"><nav class="tov-path" aria-label="Where you are in this map"></nav>` +
    `<span class="tv-act"></span><button type="button" class="tv-q" aria-expanded="false" aria-controls="tv-info" ` +
    `title="About this map" aria-label="About this map">?</button></div>` +
    `<div class="tv-info" id="tv-info" role="region" aria-label="About this map" hidden></div>` +
    `<div class="tov-map" role="group"></div></div>` +
    `<div class="tv-globe" aria-hidden="true"></div>` +
    `<section class="tv-cloud" aria-label="Its words"></section>` +
    `<section class="tv-text" aria-label="What it is about" aria-live="polite"></section>` +
    `<section class="tv-read" aria-label="Reading lists"></section>`;
  const $ = <T extends HTMLElement>(s: string) => tv.querySelector<T>(s)!;
  const path = $(".tov-path"), tm = $(".tov-map"), cloudRow = $(".tv-cloud"), textRow = $(".tv-text"), read = $(".tv-read");
  const slot = $(".tv-globe"), q = $<HTMLButtonElement>(".tv-q"), info = $("#tv-info");
  info.append(parts.info);
  const actBox = $(".tv-act");
  const anySubs = d.groups.some((g) => g.subgroups?.length);
  tm.setAttribute("aria-label", `The groups of ${whole.toLowerCase()}${anySubs ? " and their subgroups" : ""}, sized by their studies`);

  let sel = parse(deps.at, d);
  if (sel.g == null && !d.overall) sel = { g: d.groups[0]?.id ?? null, s: null };   // maps made before 2026-10-15
  const hasSubs = (id: number | null) => !!d.groups.find((g) => g.id === id)?.subgroups?.length;
  let zoom: number | null = deps.at && hasSubs(sel.g) ? sel.g : null;
  let lens = "core";

  // ------------------------------------------------------------------ the boxes
  const tiles: Tile[] = [];
  const pressable = new Map<string, HTMLButtonElement>();
  const items: { g: Group | null; size: number }[] = d.groups.map((g) => ({ g, size: g.size }));
  if (d.coverage.other) items.push({ g: null, size: d.coverage.other });
  for (const { g, size } of items) {
    const b = tm.appendChild(document.createElement("div"));
    if (!g) {                                          // studies in no group: a grey box, nothing to read
      b.className = "tov-g tov-none g0";
      b.title = `${size.toLocaleString()} studies in no clear group`;
      b.innerHTML = `<span class="tov-name">In no clear group</span> <span class="tov-n">${size.toLocaleString()}</span>`;
      tiles.push({ g, size, box: b, kids: [] });
      continue;
    }
    b.className = `tov-g ${colour(g.id)}`;
    b.onpointerenter = () => preview({ g: g.id, s: null });
    const subs = g.subgroups ?? [];
    const head = b.appendChild(document.createElement("button"));
    head.type = "button";
    head.className = "tov-head" + (subs.length ? "" : " tov-fill");
    head.innerHTML = `<span class="tov-name"><b>${g.id}</b> ${esc(g.name)}</span> <span class="tov-n">${g.size.toLocaleString()}</span>`;
    head.title = `Group ${g.id}: ${g.name} · ${g.size.toLocaleString()} studies`;
    head.setAttribute("aria-label", `Group ${g.id}: ${g.name}, ${g.size.toLocaleString()} studies` +
      (subs.length ? `, in ${subs.length} subgroups` : ""));
    head.onclick = () => pick({ g: g.id, s: null });
    head.onfocus = () => preview({ g: g.id, s: null });
    pressable.set(`${g.id}`, head);
    const t: Tile = { g, size, box: b, head, kids: [] };
    tiles.push(t);
    if (!subs.length) continue;
    const inner = t.inner = b.appendChild(document.createElement("div"));
    inner.className = "tov-inner";
    for (const s of subs) {
      const k = inner.appendChild(document.createElement("button"));
      k.type = "button";
      k.className = "tov-s";
      k.innerHTML = `<span class="tov-name">${esc(s.name)}</span> <span class="tov-n">${s.size.toLocaleString()}</span>`;
      k.title = `${g.id}.${s.id}: ${s.name} · ${s.size.toLocaleString()} studies`;
      k.setAttribute("aria-label", `Subgroup ${g.id}.${s.id} of group ${g.id}: ${s.name}, ${s.size.toLocaleString()} studies`);
      k.onclick = () => pick({ g: g.id, s: s.id });
      k.onpointerenter = () => preview({ g: g.id, s: s.id });
      k.onpointerleave = () => preview({ g: g.id, s: null });
      k.onfocus = () => preview({ g: g.id, s: s.id });
      pressable.set(`${g.id}.${s.id}`, k);
      t.kids.push({ s, size: s.size, el: k });
    }
    if (g.subgroups_other) {
      const k = inner.appendChild(document.createElement("div"));
      k.className = "tov-s tov-none";
      k.title = `${g.subgroups_other.toLocaleString()} studies of group ${g.id} in no clear subgroup`;
      k.innerHTML = `<span class="tov-name">no clear subgroup</span>`;
      t.kids.push({ s: null, size: g.subgroups_other, el: k });
    }
  }

  /** Every box where it belongs: the field's boxes, or the zoomed group filling the space with its subgroups. */
  let settle = 0;
  const layout = (animate: boolean) => {
    const W = tm.clientWidth, H = tm.clientHeight;
    if (!W || !H) return;
    if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      tm.classList.add("anim");
      clearTimeout(settle);
      settle = window.setTimeout(() => tm.classList.remove("anim"), ANIM + 60);
    }
    tm.classList.toggle("zoomed", zoom != null);
    const outer = squarify(tiles.map((t) => t.size), { x: 0, y: 0, w: W, h: H });
    tiles.forEach((t, i) => {
      const z = t.g != null && t.g.id === zoom, gone = zoom != null && !z;
      const r = z ? { x: 0, y: 0, w: W, h: H } : outer[i];
      place(t.box, r, W, H);
      t.box.classList.toggle("z", z);
      t.box.classList.toggle("gone", gone);
      t.box.inert = gone;
      t.box.classList.toggle("tiny", !z && (r.w < 90 || r.h < 34));
      if (!t.inner) { if (t.head) clamp(t.head, r.h - 10 - LINE); return; }
      const head = z ? HEAD_Z : HEAD;
      const iw = Math.max(1, r.w - 4), ih = Math.max(1, r.h - head - 3);
      t.box.classList.toggle("cramped", !z && (r.h < HEAD + 24 || r.w < 40));
      const rects = squarify(t.kids.map((k) => k.size), { x: 0, y: 0, w: iw, h: ih });
      t.kids.forEach((k, j) => {
        const rr = rects[j];
        place(k.el, rr, iw, ih);
        k.el.classList.toggle("tiny", rr.w < (z ? 56 : 64) || rr.h < (z ? 26 : 30));
        clamp(k.el, rr.h - (z ? 12 + LINE_Z : 8 + LINE), z ? LINE_Z : LINE);
      });
    });
  };

  // ------------------------------------------------------------------ pointing, picking, zooming
  /** The unit shown under the globe and lit on it: the one pointed at, else the one picked. Leaving a box waits a
   *  moment before going back, so sliding across the gap between two boxes doesn't flash the picked one. */
  let shown: At | null = null, back = 0;
  const preview = (a: At | null) => {
    clearTimeout(back);
    if (a) show(a); else back = window.setTimeout(() => show(sel), 140);
  };
  const show = (a: At) => {
    if (shown && same(a, shown)) return;
    shown = a;
    deps.focusGroup(a.g, a.s);
    cloudRow.replaceChildren(cloudOf(a));
    textRow.replaceChildren(textOf(a));
    fitCloud();
    fitText();
  };
  tm.onpointerleave = () => preview(null);
  tm.addEventListener("focusout", (e) => { if (!tm.contains(e.relatedTarget as Node)) preview(null); });

  const pick = (a: At) => {
    sel = a;
    const z = a.g != null && hasSubs(a.g) ? a.g : null;
    if (z !== zoom) { zoom = z; layout(true); }
    tv.dataset.touched = "1";              // a late refresh of this map (topics.ts) leaves it alone now
    deps.select?.(atStr(a));
    draw();
    show(a);
  };
  const up = () => pick(d.overall ? { g: null, s: null } : { g: zoom ?? sel.g, s: null });

  /** The pressed box, the path back up, the reading lists. */
  const draw = () => {
    for (const [k, b] of pressable) b.setAttribute("aria-pressed", String(k === atStr(sel)));
    path.replaceChildren();
    const g = sel.g == null ? null : d.groups.find((x) => x.id === sel.g)!;
    const s = g && sel.s != null ? g.subgroups!.find((x) => x.id === sel.s)! : null;
    const step = (label: string, to: At | null, cls = "") => {
      if (path.childNodes.length) path.append(Object.assign(document.createElement("span"), { className: "tov-sep", textContent: "›" }));
      const el = document.createElement(to ? "button" : "b");
      el.textContent = label;
      el.className = cls;
      if (to) { (el as HTMLButtonElement).type = "button"; el.onclick = () => pick(to); }
      if (!to) el.setAttribute("aria-current", "location");
      path.append(el);
    };
    if (g || zoom != null) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "tov-up"; b.innerHTML = `<span aria-hidden="true">↑</span> ${esc(whole)}`;
      b.title = "Back to the whole map (Esc)";
      b.onclick = up;
      if (!d.overall && zoom == null) { step(whole, null); } else path.append(b);
    } else step(whole, null);
    if (g) step(`${g.id} · ${g.name}`, s ? { g: g.id, s: null } : null, "tov-crumb");
    if (s) step(`${g!.id}.${s.id} · ${s.name}`, null, "tov-crumb");
    actBox.replaceChildren();
    if (g && !s && !d.parent && g.size >= parts.splitFrom) {   // the picked group, on a map of its own
      const l = actBox.appendChild(document.createElement("a"));
      l.href = deps.url({ group: g.id });
      l.textContent = g.subgroups?.length ? `Map group ${g.id} on its own →` : `Split group ${g.id} →`;
      l.title = g.subgroups?.length ? `Group ${g.id} as a map of its own, grouped again` : `Group ${g.id}, split into smaller groups`;
      l.onclick = (e) => { e.preventDefault(); deps.go(l.getAttribute("href")!); };
    } else if (parts.act) actBox.append(parts.act);
    readingLists();
  };

  const readingLists = () => {
    read.replaceChildren();
    const g = sel.g == null ? null : d.groups.find((x) => x.id === sel.g)!;
    const s = g && sel.s != null ? g.subgroups!.find((x) => x.id === sel.s)! : null;
    const it = s ?? g ?? d.overall;
    if (!it) return;
    const unit = s ? "subgroup" : g ? "group" : d.mode === "field" ? "field" : "topic";
    const name = s ? s.name : g ? g.name : d.question;
    read.insertAdjacentHTML("beforeend", `<h3 class="tv-read-h">${g ? `<span class="tg-chip ${colour(g.id)}" aria-hidden="true"></span>` : ""}` +
      `<span>What to read in <b>${esc(name)}</b></span></h3>`);
    const names = new Map((s ? g!.subgroups! : d.groups).map((x) => [x.id, x.name]));
    if (Object.values(it.lenses ?? {}).some((l) => l.length))
      parts.lists(read, `o${atStr(sel) || 0}`, it.lenses, unit, names, deps, lens, (k) => { lens = k; });
    else read.insertAdjacentHTML("beforeend", `<p class="tg-help">No reading lists for this one.</p>`);
  };

  /** Under the globe, two rows with sizes of their own, so pointing from box to box changes what they hold and
   *  never where anything sits: the unit's word cloud in the middle, centred between the globe and the text; its
   *  description, then its name and size at the very bottom. */
  const unitOf = (a: At) => {
    const g = a.g == null ? null : d.groups.find((x) => x.id === a.g)!;
    const s = g && a.s != null ? g.subgroups!.find((x) => x.id === a.s)! : null;
    return { g, s, it: s ?? g ?? d.overall ?? { size: d.coverage.papers, years: null, cloud: undefined } };
  };
  const buildText = (a: At): HTMLElement => {
    const { g, s, it } = unitOf(a);
    const div = document.createElement("div");
    div.className = "tv-text-in";
    const title = s ? s.name : g ? g.name : d.question[0].toUpperCase() + d.question.slice(1);
    const where = s ? `Subgroup ${g!.id}.${s.id}` : g ? `Group ${g.id}` : d.parent ? `Group ${d.parent.group}, grouped again` :
      d.mode === "field" ? "The whole field" : "The closest studies";
    const meta = [where, `${it.size.toLocaleString()} studies`, parts.span(it.years)].filter(Boolean).join(" · ");
    // the whole field: the definition it was made from (in the address), if any
    const about = g ? (s ?? g).label?.about : d.mode === "field" && !d.parent ? new URLSearchParams(location.search).get("definition") : null;
    div.innerHTML = (about ? `<p class="tg-about" title="${esc(about)}">${esc(about)}</p>` : "") +
      `<h3 class="tv-h">${g ? `<span class="tg-chip ${colour(g.id)}" aria-hidden="true"></span>` : ""}<span>${esc(title)}</span></h3>` +
      `<p class="tg-meta">${esc(meta)}</p>`;
    return div;
  };
  const buildCloud = (a: At): HTMLElement => {
    const { g, s, it } = unitOf(a);
    const div = document.createElement("div");
    div.className = "tv-cloud-in";
    if (!it.cloud?.length) return div;
    const vs = s ? `the rest of group ${g!.id}` : g ? "the other groups" : "all of science";
    div.append(parts.cloud(it.cloud, 30, vs));
    div.insertAdjacentHTML("beforeend", `<p class="tv-legend"><span class="cl-up">blue</span>: used more than in ${vs} · ` +
      `<span class="cl-down">red</span>: less · bigger: in more of its studies</p>`);
    return div;
  };
  const kept = new Map<string, [HTMLElement, HTMLElement]>();     // made once each
  const keptOf = (a: At) => {
    let k = kept.get(atStr(a));
    if (!k) kept.set(atStr(a), k = [buildCloud(a), buildText(a)]);
    return k;
  };
  const cloudOf = (a: At) => keptOf(a)[0], textOf = (a: At) => keptOf(a)[1];
  const units: At[] = [{ g: null, s: null }, ...d.groups.flatMap((g) =>
    [{ g: g.id, s: null }, ...(g.subgroups ?? []).map((x) => ({ g: g.id, s: x.id }))])];

  /** The rows' sizes: the text row as tall as most texts of this map need at this width (a longer description is
   *  cut to the lines that fit, whole in its tooltip); the cloud row the rest of the column (one column: as tall as
   *  the tallest cloud). Measured off screen, at each change of width. */
  let measuredAt = 0;
  const measure = () => {
    const w = textRow.clientWidth;
    if (!w || w === measuredAt) return;
    measuredAt = w;
    const probe = tv.appendChild(document.createElement("div"));
    probe.className = "tv-probe";
    probe.style.width = `${w}px`;
    const texts: number[] = [];
    let cloudH = 0;
    const wide = getComputedStyle(tv).display === "grid";
    for (const a of units) {
      const t = probe.appendChild(buildText(a));
      texts.push(t.offsetHeight);
      t.remove();
      if (!wide) { const c = probe.appendChild(buildCloud(a)); cloudH = Math.max(cloudH, c.offsetHeight); c.remove(); }
    }
    probe.remove();
    // wide: room for 85% of the texts whole (a few long ones would leave a gap under every other), as long as the
    // cloud keeps ~120 px (row gaps and margin: ~50 px); one column: all of them
    texts.sort((x, y) => x - y);
    const need = texts[wide ? Math.floor(0.85 * (texts.length - 1)) : texts.length - 1] ?? 0;
    const cap = wide ? Math.max(150, Math.min(300, tv.clientHeight - slot.offsetHeight - 50 - 120)) : 1e9;
    tv.style.setProperty("--tv-text", `${Math.ceil(Math.min(need, cap))}px`);
    tv.style.setProperty("--tv-cloud", wide ? "auto" : `${Math.ceil(cloudH)}px`);
  };
  /** A cloud too big for its row shrinks as a whole (fewer lines once smaller, so a few tries). */
  const fitCloud = () => {
    const box = cloudRow.firstElementChild as HTMLElement | null;
    if (!box) return;
    box.style.zoom = "1";
    const room = cloudRow.clientHeight, tall = () => box.getBoundingClientRect().height;   // zoomed size (offsetHeight isn't)
    for (let z = 1, i = 0; room && tall() > room && i < 8; i++) {
      z *= Math.max(0.6, Math.min(0.97, Math.sqrt(room / tall())));
      box.style.zoom = String(z);
    }
  };
  /** A description too long for the text row keeps the whole lines that fit. */
  const fitText = () => {
    const box = textRow.firstElementChild as HTMLElement | null;
    const p = box?.querySelector<HTMLElement>(".tg-about");
    if (!box || !p) return;
    p.style.removeProperty("-webkit-line-clamp");
    p.classList.remove("cut");
    const over = box.offsetHeight - textRow.clientHeight;
    if (over <= 1) return;
    const line = parseFloat(getComputedStyle(p).lineHeight) || 22;
    p.classList.add("cut");
    p.style.setProperty("-webkit-line-clamp", String(Math.max(1, Math.floor((p.offsetHeight - over) / line))));
  };

  // ------------------------------------------------------------------ "?"
  const open = (on: boolean) => {
    info.hidden = !on;
    q.setAttribute("aria-expanded", String(on));
  };
  q.onclick = () => open(info.hasAttribute("hidden"));
  const outside = (e: PointerEvent) => { if (!info.hidden && !info.contains(e.target as Node) && e.target !== q) open(false); };
  document.addEventListener("pointerdown", outside);
  tv.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!info.hidden) { open(false); q.focus(); } else if (zoom != null || sel.g != null) up();
  });

  // ------------------------------------------------------------------ the globe over its slot, the screen's height
  document.body.classList.add("topics-dash");
  const fit = () => {
    if (!tv.isConnected) return stop?.();
    tv.style.setProperty("--tv-top", `${Math.round(tv.getBoundingClientRect().top + scrollY)}px`);
    const r = slot.getBoundingClientRect();
    if (r.width) for (const [k, v] of [["--tv-x", r.left], ["--tv-y", r.top], ["--tv-s", r.width]] as const)
      document.body.style.setProperty(k, `${Math.round(v)}px`);
  };
  const ro = new ResizeObserver(() => {
    if (!tv.isConnected) return stop?.();
    fit(); layout(false); measure(); fitCloud(); fitText();
  });
  ro.observe(tm); ro.observe(slot); ro.observe(textRow); ro.observe(cloudRow);
  addEventListener("resize", fit);
  stop = () => {
    ro.disconnect(); removeEventListener("resize", fit); document.removeEventListener("pointerdown", outside);
    clearTimeout(back); clearTimeout(settle); stop = null;
  };

  fit();
  layout(false);
  draw();
  measure();
  show(sel);
}
