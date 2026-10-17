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
import type { CloudWord, Deps, Group, Lenses, More, Sub, TopicMap } from "./topics.ts";

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

export interface Parts {
  cloud: (w: CloudWord[], k: number, vs: string) => HTMLElement;
  lists: (host: HTMLElement, key: string, l: Lenses, unit: string, names: Map<number, string>, deps: Deps,
          want: string, onPick: (lens: string) => void,
          further?: (lens: string, offset: number, limit: number) => ReturnType<More>) => void;
  span: (y?: number[] | null) => string;
  info: HTMLElement;                 // what "?" opens: how the map was made, its limits, other ways to see it
  act: HTMLElement | null;           // one link in the bar (map the whole field, back to the whole map...)
  splitFrom: number;                 // a group this big can be mapped again on its own
  legend?: string;                   // under the cloud (default: the word cloud's colours)
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

const HEAD_Z = 32;  // px: at least, the name strip of the group zoomed into (.tov-g.z .tov-head)
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

interface Tile { g: Group | null; size: number; box: HTMLElement; head?: HTMLButtonElement; label: HTMLElement; inner?: HTMLElement;
                 kids: { s: Sub | null; size: number; el: HTMLElement }[] }

let stop: (() => void) | null = null;   // the shown dashboard's listeners, removed when another replaces it

export function renderOverview(box: HTMLElement, d: TopicMap, deps: Deps, parts: Parts) {
  stop?.();
  const person = d.mode === "author" || d.mode === "institution" || d.mode === "source";
  const their = d.mode === "author" ? "their" : "its";
  const whole = d.parent ? `All of group ${d.parent.group}` : d.mode === "field" ? "Whole field" : person ? `All ${their} work` : "Whole topic";
  const unitName = (sub: boolean) => person ? (d.level === "subfields" ? (sub ? "topic" : "subfield") : (sub ? "subfield" : "field")) : sub ? "subgroup" : "group";
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
  tm.setAttribute("aria-label", person ? `${their[0].toUpperCase()}${their.slice(1)} ${unitName(false)}s${anySubs ? ` and the ${unitName(true)}s inside` : ""}, sized by their studies`
    : `The groups of ${whole.toLowerCase()}${anySubs ? " and their subgroups" : ""}, sized by their studies`);

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
      b.title = person ? `${size.toLocaleString()} studies with no field label, or in fields too small to show` : `${size.toLocaleString()} studies in no clear group`;
      b.innerHTML = `<div class="tov-head"><span class="tov-name">${person ? "Other" : "In no clear group"}</span> <span class="tov-n">${size.toLocaleString()}</span></div>`;
      tiles.push({ g, size, box: b, label: b.firstElementChild as HTMLElement, kids: [] });
      continue;
    }
    b.className = `tov-g ${colour(g.id)}`;
    b.onpointerenter = () => preview({ g: g.id, s: null });
    const subs = g.subgroups ?? [];
    const head = b.appendChild(document.createElement("button"));
    head.type = "button";
    head.className = "tov-head";
    head.innerHTML = `<span class="tov-name">${person ? "" : `<b>${g.id}</b> `}${esc(g.name)}</span> <span class="tov-n">${g.size.toLocaleString()}` +
      (subs.length ? `<span class="tov-k"> · ${subs.length} ${unitName(true)}s</span>` : "") + `</span>`;
    head.title = `${person ? "" : `Group ${g.id}: `}${g.name} · ${g.size.toLocaleString()} studies`;
    head.setAttribute("aria-label", `${person ? `${unitName(false)[0].toUpperCase()}${unitName(false).slice(1)}` : `Group ${g.id}`}: ${g.name}, ${g.size.toLocaleString()} studies` +
      (subs.length ? `, in ${subs.length} ${unitName(true)}s` : ""));
    head.onclick = () => pick({ g: g.id, s: null });
    head.onfocus = () => preview({ g: g.id, s: null });
    pressable.set(`${g.id}`, head);
    const t: Tile = { g, size, box: b, head, label: head, kids: [] };
    tiles.push(t);
    if (!subs.length) continue;
    const inner = t.inner = b.appendChild(document.createElement("div"));
    inner.className = "tov-inner";
    for (const s of subs) {
      const k = inner.appendChild(document.createElement("button"));
      k.type = "button";
      k.className = "tov-s";
      k.innerHTML = `<span class="tov-name">${esc(s.name)}</span> <span class="tov-n">${s.size.toLocaleString()}</span>`;
      k.title = `${person ? "" : `${g.id}.${s.id}: `}${s.name} · ${s.size.toLocaleString()} studies`;
      k.setAttribute("aria-label", person ? `${s.name}, a ${unitName(true)} of ${g.name}: ${s.size.toLocaleString()} studies`
        : `Subgroup ${g.id}.${s.id} of group ${g.id}: ${s.name}, ${s.size.toLocaleString()} studies`);
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
      k.title = person ? `${g.subgroups_other.toLocaleString()} studies of ${g.name} with no ${unitName(true)} label`
        : `${g.subgroups_other.toLocaleString()} studies of group ${g.id} in no clear subgroup`;
      k.innerHTML = `<span class="tov-name">${person ? `no ${unitName(true)} label` : "no clear subgroup"}</span>`;
      t.kids.push({ s: null, size: g.subgroups_other, el: k });
    }
  }

  /** Names are fitted on a hidden copy of a box at its final size (the boxes themselves may be moving): the
   *  largest type, 15 px down to 10.5, that shows the whole name and its count (then the name and count only);
   *  a box too small even then keeps the lines that fit and ends in "…" (its name is in its tooltip, and shown
   *  in full under the globe when pointed at). Zoomed in, the group's name strip is as tall as its name needs. */
  const probe = tm.appendChild(document.createElement("div"));
  probe.className = "tov-g tov-probe";
  probe.setAttribute("aria-hidden", "true");
  const SIZES = [15, 14, 13, 12.5, 12, 11.5, 11, 10.5];
  const fitName = (t: Tile, w: number, h: number) => {
    const el = t.label;
    probe.className = `tov-g tov-probe${t.g ? "" : " tov-none"}`;
    probe.style.width = `${w}px`; probe.style.height = `${h}px`;
    const c = probe.appendChild(el.cloneNode(true) as HTMLElement);
    c.classList.remove("cut", "bare");
    const name = c.querySelector<HTMLElement>(".tov-name")!, k = c.querySelector<HTMLElement>(".tov-k");
    const n = c.querySelector<HTMLElement>(".tov-n")!;
    name.style.setProperty("-webkit-line-clamp", "none");
    // whole words only (no hyphenation here): a word wider than the box makes it scroll sideways, so it doesn't fit
    const fits = () => c.scrollHeight <= c.clientHeight + 1 && c.scrollWidth <= c.clientWidth + 1 &&
      name.scrollWidth <= name.clientWidth + 1;
    // the name first: the subgroups' count goes first, then the size (both are under the globe when pointed at)
    let size = 0, withK = false, bare = false;
    outer: for (const f of SIZES) {
      c.style.fontSize = `${f}px`;
      for (const [kOn, nOn] of [[true, true], [false, true]] as const) {
        if (k) k.hidden = !kOn;
        n.hidden = !nOn;
        if ((kOn && !k) || !fits()) continue;
        size = f; withK = kOn && !!k; break outer;
      }
    }
    if (!size) {                                       // smallest type, the name alone
      size = SIZES[SIZES.length - 1];
      c.style.fontSize = `${size}px`;
      n.hidden = true;
      if (k) k.hidden = true;
      bare = true;
    }
    let lines = 0;
    if (!fits()) lines = Math.max(1, Math.floor((c.clientHeight - 10) / (size * 1.25)));   // whole lines, then "…"
    c.remove();
    el.style.fontSize = `${size}px`;
    el.classList.toggle("cut", lines > 0);
    el.classList.toggle("bare", bare);
    el.querySelector<HTMLElement>(".tov-name")!.style.setProperty("-webkit-line-clamp", lines ? String(lines) : "none");
    const kk = el.querySelector<HTMLElement>(".tov-k");
    if (kk) kk.hidden = !withK;
  };
  const stripHeight = (t: Tile, w: number) => {        // zoomed in: the name strip, wrapped to the box's width
    probe.className = "tov-g tov-probe z";
    probe.style.width = `${w}px`; probe.style.height = "auto";
    const c = probe.appendChild(t.head!.cloneNode(true) as HTMLElement);
    c.style.removeProperty("font-size");
    const hz = Math.max(HEAD_Z, Math.ceil(c.offsetHeight));
    c.remove();
    return hz;
  };

  /** Every box where it belongs. The field: the groups alone, their subgroups hidden (pointing at a group is about
   *  the group); zoomed into one: it fills the space and its subgroups show. */
  let settle = 0;
  const layout = (animate: boolean) => {
    const W = tm.clientWidth, H = tm.clientHeight;
    if (!W || !H) return;
    if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      tm.classList.add("anim");
      clearTimeout(settle);
      settle = window.setTimeout(() => tm.classList.remove("anim"), ANIM + 360);
    }
    tm.classList.toggle("zoomed", zoom != null);
    const outer = squarify(tiles.map((t) => t.size), { x: 0, y: 0, w: W, h: H });
    tiles.forEach((t, i) => {
      const z = t.g != null && t.g.id === zoom, gone = zoom != null && !z;
      const r = outer[i];
      place(t.box, z ? { x: 0, y: 0, w: W, h: H } : r, W, H);
      t.box.classList.toggle("z", z);
      t.box.classList.toggle("gone", gone);
      t.box.inert = gone;
      t.box.classList.toggle("tiny", r.w < 34 || r.h < 26);
      if (z) {                                         // zoomed: the strip's own type (style.css), the whole name
        t.label.style.removeProperty("font-size");
        t.label.classList.remove("cut");
        t.label.querySelector<HTMLElement>(".tov-name")!.style.setProperty("-webkit-line-clamp", "none");
      } else if (!gone) fitName(t, r.w, r.h);          // its name sized to its box
      if (!t.inner) return;
      t.inner.inert = !z;                              // subgroups: only once inside the group
      // subgroups laid out for the zoomed box, so they grow into place with it
      const hz = stripHeight(t, W);
      t.box.style.setProperty("--hz", `${hz}px`);
      const iw = Math.max(1, W - 4), ih = Math.max(1, H - hz - 3);
      const rects = squarify(t.kids.map((k) => k.size), { x: 0, y: 0, w: iw, h: ih });
      t.kids.forEach((k, j) => {
        const rr = rects[j];
        place(k.el, rr, iw, ih);
        k.el.classList.toggle("tiny", rr.w < 56 || rr.h < 26);
        clamp(k.el, rr.h - 12 - LINE_Z, LINE_Z);
      });
    });
  };
  document.fonts?.ready.then(() => { if (tm.isConnected) layout(false); });   // names measured in the real font

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
    if (g) step(person ? g.name : `${g.id} · ${g.name}`, s ? { g: g.id, s: null } : null, "tov-crumb");
    if (s) step(person ? s.name : `${g!.id}.${s.id} · ${s.name}`, null, "tov-crumb");
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
    const unit = s ? unitName(true) : g ? unitName(false) : person ? "whole" : d.mode === "field" ? "field" : "topic";
    const name = s ? s.name : g ? g.name : d.question;
    read.insertAdjacentHTML("beforeend", `<h3 class="tv-read-h">${g ? `<span class="tg-chip ${colour(g.id)}" aria-hidden="true"></span>` : ""}` +
      `<span>${person ? (g ? `${their[0].toUpperCase()}${their.slice(1)} work in <b>${esc(name)}</b>` : `All the work of <b>${esc(name)}</b>`) : `What to read in <b>${esc(name)}</b>`}</span></h3>`);
    const names = new Map((s ? g!.subgroups! : d.groups).map((x) => [x.id, x.name]));
    if (Object.values(it.lenses ?? {}).some((l) => l.length))
      parts.lists(read, `o${atStr(sel) || 0}`, it.lenses, unit, names, deps, lens, (k) => { lens = k; },
                  deps.more ? (k, offset, limit) => deps.more!({ ...sel }, k, offset, limit) : undefined);
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
    const cap = (u: string) => u[0].toUpperCase() + u.slice(1);
    const where = person ? (s ? `${cap(unitName(true))} of ${g!.name}` : g ? cap(unitName(false)) : `All ${their} work`)
      : s ? `Subgroup ${g!.id}.${s.id}` : g ? `Group ${g.id}` : d.parent ? `Group ${d.parent.group}, grouped again` :
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
    div.dataset.at = atStr(a);          // which unit a clicked word belongs to (topics.ts)
    if (!it.cloud?.length) return div;
    const vs = s ? `the rest of group ${g!.id}` : g ? "the other groups" : "all of science";
    div.append(parts.cloud(it.cloud, 30, vs));
    div.insertAdjacentHTML("beforeend", parts.legend ?? `<p class="tv-legend"><span class="cl-up">blue</span>: used more than in ${vs} · ` +
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
