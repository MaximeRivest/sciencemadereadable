/**
 * The overview of a topic map: the whole field at a glance, as boxes. Each research group is a box sized by its
 * studies, and the boxes inside it are its subgroups (the data API makes them with the map), so how the field
 * divides, and how each part divides again, shows before anyone reads a list. Pick a box (the whole, a group or a
 * subgroup) and its words and reading lists open right under the boxes; pointing at one lights its studies on
 * the map of science. Layout: squarified treemap (Bruls, Huizing & van Wijk 2000), boxes as close to squares as
 * the sizes allow, biggest first, so sizes compare and names fit.
 */
import type { CloudWord, Deps, Group, Lenses, Sub, TopicMap } from "./topics.ts";
import { about, legend } from "./topics.ts";

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

export interface Parts {
  cloud: (w: CloudWord[], k: number, vs: string, note?: string) => HTMLElement;
  lists: (sec: HTMLElement, key: string, l: Lenses, unit: string, names: Map<number, string>, deps: Deps) => void;
  span: (y?: number[] | null) => string;
  hover: (deps: Deps, g: number | null, sub?: number | null) => void;
  keep: (deps: Deps, f: [number, number | null] | null) => void;
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

const HEAD = 24;   // px: a group box's name strip, above its subgroups
const LINE = 14.4; // px: a line of a box's name (style.css .tov-s)

/** A name gets the whole lines its box has room for (then "…"), never a line cut in half. */
function clamp(el: HTMLElement, px: number) {
  const name = el.querySelector<HTMLElement>(".tov-name");
  if (name) name.style.setProperty("-webkit-line-clamp", String(Math.max(1, Math.floor(px / LINE))));
}

/** Where things stand: "" the whole, "3" group 3, "3.2" subgroup 2 of group 3. */
type At = { g: number | null; s: number | null };
const parse = (at: string | undefined, d: TopicMap): At => {
  const [a, b] = (at ?? "").split(".").map(Number);
  const g = d.groups.find((x) => x.id === a);
  if (!g) return { g: null, s: null };
  return { g: g.id, s: g.subgroups?.some((x) => x.id === b) ? b : null };
};
const atStr = (a: At) => a.g == null ? "" : a.s == null ? `${a.g}` : `${a.g}.${a.s}`;

export function renderOverview(d: TopicMap, deps: Deps, parts: Parts): HTMLElement {
  const sec = document.createElement("section");
  sec.className = "tov";
  sec.setAttribute("aria-labelledby", "tov-h");
  const whole = d.parent ? `group ${d.parent.group}` : d.mode === "field" ? "the field" : "the topic";
  const anySubs = d.groups.some((g) => g.subgroups?.length);
  sec.innerHTML = `<h2 class="tov-title" id="tov-h">${esc(whole[0].toUpperCase() + whole.slice(1))} at a glance</h2>` +
    `<p class="tg-help">Each box is a research group, sized by its studies${anySubs ? "; the boxes inside it are its subgroups" : ""}. ` +
    `Pick one to see its words and reading lists below; point at one to find its studies on the map.</p>`;
  const tm = sec.appendChild(document.createElement("div"));
  tm.className = "tov-map";
  tm.setAttribute("role", "group");
  tm.setAttribute("aria-label", `The groups of ${whole}${anySubs ? " and their subgroups" : ""}, sized by their studies`);
  const panel = sec.appendChild(document.createElement("div"));
  panel.className = "tov-panel";
  const tiles = new Map<string, HTMLButtonElement>();
  let at = parse(deps.at, d);
  if (at.g == null && !d.overall) at = { g: d.groups[0]?.id ?? null, s: null };   // maps made before 2026-10-15

  const pick = (a: At, scroll = false) => {
    at = a;
    for (const [k, b] of tiles) b.setAttribute("aria-pressed", String(k === atStr(a)));
    parts.keep(deps, a.g == null ? null : [a.g, a.s]);
    deps.select?.(atStr(a));
    detail(panel, d, a, deps, parts, pick);
    if (scroll && panel.getBoundingClientRect().top > innerHeight - 120) panel.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  // the layout, in px of the box as it is now (positions are kept in % so it follows the page's width)
  const W = Math.max(280, Math.min(760, (document.querySelector("#results") as HTMLElement | null)?.clientWidth || 680));
  const H = Math.round(W * (W < 520 ? 1.15 : 0.62));
  const pctOf = (r: Rect, ow: number, oh: number, el: HTMLElement) => {
    el.style.left = `${100 * r.x / ow}%`; el.style.top = `${100 * r.y / oh}%`;
    el.style.width = `${100 * r.w / ow}%`; el.style.height = `${100 * r.h / oh}%`;
  };
  tm.style.aspectRatio = `${W} / ${H}`;
  const items: { g: Group | null; size: number }[] = d.groups.map((g) => ({ g, size: g.size }));
  if (d.coverage.other) items.push({ g: null, size: d.coverage.other });
  const outer = squarify(items.map((x) => x.size), { x: 0, y: 0, w: W, h: H });
  items.forEach(({ g, size }, i) => {
    const r = outer[i];
    const colour = g ? `g${((g.id - 1) % 9) + 1}` : "g0";
    if (!g) {                                          // studies in no group: a grey box, nothing to read
      const div = tm.appendChild(document.createElement("div"));
      div.className = `tov-g tov-none ${colour}` + (r.w < 90 || r.h < 34 ? " tiny" : "");
      div.title = `${size.toLocaleString()} studies in no clear group`;
      div.innerHTML = `<span class="tov-name">In no clear group</span> <span class="tov-n">${size.toLocaleString()}</span>`;
      pctOf(r, W, H, div);
      return;
    }
    const box = tm.appendChild(document.createElement("div"));
    box.className = `tov-g ${colour}`;
    pctOf(r, W, H, box);
    box.onpointerenter = () => parts.hover(deps, g.id);
    box.onpointerleave = () => parts.hover(deps, null);
    const subs = g.subgroups ?? [];
    const head = box.appendChild(document.createElement("button"));
    head.type = "button";
    head.className = "tov-head" + (subs.length ? "" : " tov-fill") + (r.w < 90 || r.h < 34 ? " tiny" : "");
    head.innerHTML = `<span class="tov-name"><b>${g.id}</b> ${esc(g.name)}</span> <span class="tov-n">${g.size.toLocaleString()}</span>`;
    head.title = `Group ${g.id}: ${g.name} · ${g.size.toLocaleString()} studies`;
    head.setAttribute("aria-label", `Group ${g.id}: ${g.name}, ${g.size.toLocaleString()} studies` +
      (subs.length ? `, in ${subs.length} subgroups` : ""));
    if (!subs.length) clamp(head, r.h - 10 - LINE);
    head.onclick = () => pick({ g: g.id, s: null }, true);
    head.onfocus = () => parts.hover(deps, g.id);
    tiles.set(`${g.id}`, head);
    if (!subs.length || r.h < HEAD + 24 || r.w < 40) return;   // too small to show inside: the group's panel lists them
    const inner = box.appendChild(document.createElement("div"));
    inner.className = "tov-inner";
    const iw = r.w - 4, ih = r.h - HEAD - 3;
    const kids: { s: Sub | null; size: number }[] = subs.map((s) => ({ s, size: s.size }));
    if (g.subgroups_other) kids.push({ s: null, size: g.subgroups_other });
    const rects = squarify(kids.map((k) => k.size), { x: 0, y: 0, w: iw, h: ih });
    kids.forEach(({ s, size }, j) => {
      const rr = rects[j];
      const tiny = rr.w < 64 || rr.h < 30;
      if (!s) {
        const div = inner.appendChild(document.createElement("div"));
        div.className = "tov-s tov-none" + (tiny ? " tiny" : "");
        div.title = `${size.toLocaleString()} studies of group ${g.id} in no clear subgroup`;
        div.innerHTML = `<span class="tov-name">no clear subgroup</span>`;
        pctOf(rr, iw, ih, div);
        return;
      }
      const b = inner.appendChild(document.createElement("button"));
      b.type = "button";
      b.className = "tov-s" + (tiny ? " tiny" : "");
      b.innerHTML = `<span class="tov-name">${esc(s.name)}</span> <span class="tov-n">${s.size.toLocaleString()}</span>`;
      b.title = `${g.id}.${s.id}: ${s.name} · ${s.size.toLocaleString()} studies`;
      b.setAttribute("aria-label", `Subgroup ${g.id}.${s.id} of group ${g.id}: ${s.name}, ${s.size.toLocaleString()} studies`);
      clamp(b, rr.h - 8 - LINE);
      b.onclick = () => pick({ g: g.id, s: s.id }, true);
      b.onpointerenter = () => parts.hover(deps, g.id, s.id);
      b.onpointerleave = () => parts.hover(deps, g.id);
      b.onfocus = () => parts.hover(deps, g.id, s.id);
      pctOf(rr, iw, ih, b);
      tiles.set(`${g.id}.${s.id}`, b);
    });
  });
  tm.onpointerleave = () => parts.hover(deps, null);
  tm.addEventListener("focusout", (e) => { if (!tm.contains(e.relatedTarget as Node)) parts.hover(deps, null); });
  pick(at);
  return sec;
}

/** The picked box: where it sits (a path back up), its size, words and reading lists. */
function detail(panel: HTMLElement, d: TopicMap, a: At, deps: Deps, parts: Parts, pick: (a: At, scroll?: boolean) => void) {
  panel.replaceChildren();
  const g = a.g == null ? null : d.groups.find((x) => x.id === a.g)!;
  const s = g && a.s != null ? g.subgroups!.find((x) => x.id === a.s)! : null;
  const whole = d.parent ? `All of group ${d.parent.group}` : d.mode === "field" ? "The whole field" : "The whole topic";
  // the path: The whole field › Group 3 › 3.2
  const path = panel.appendChild(document.createElement("p"));
  path.className = "tov-path";
  const step = (label: string, to: At | null) => {
    if (path.childNodes.length) path.append(" › ");
    if (!to) { const b = document.createElement("b"); b.textContent = label; path.append(b); return; }
    const b = document.createElement("button");
    b.type = "button"; b.className = "linklike"; b.textContent = label;
    b.onclick = () => pick(to);
    path.append(b);
  };
  if (!g) path.hidden = true;                          // at the top: nothing to go back to
  if (d.overall) step(whole, g ? { g: null, s: null } : null);
  if (g) step(`Group ${g.id}`, s ? { g: g.id, s: null } : null);
  if (s) step(`Subgroup ${g!.id}.${s.id}`, null);

  const it = s ?? g ?? d.overall!;
  const title = s ? s.name : g ? g.name : whole;
  const unit = s ? "subgroup" : g ? "group" : d.mode === "field" ? "field" : "topic";
  const h = panel.appendChild(document.createElement("h3"));
  h.className = "tov-h" + (g ? ` g${((g.id - 1) % 9) + 1}` : "");
  h.innerHTML = (g ? `<span class="tg-chip" aria-hidden="true"></span>` : "") + esc(title);
  const subs = !s && g ? g.subgroups ?? [] : [];
  const meta = [`${it.size.toLocaleString()} studies`, parts.span(it.years),
    subs.length ? `${subs.length} subgroups` : "", !g && d.groups.length ? `${d.groups.length} groups` : ""].filter(Boolean).join(" · ");
  panel.insertAdjacentHTML("beforeend", (s || g ? about(it as Sub) : "") + `<p class="tg-meta">${meta}</p>`);
  const vs = s ? `the rest of group ${g!.id}` : g ? "the other groups" : "all of science";
  if (it.cloud?.length) panel.append(parts.cloud(it.cloud, 30, vs, legend(vs)));
  const names = new Map((s ? g!.subgroups! : d.groups).map((x) => [x.id, x.name]));
  if (Object.values(it.lenses ?? {}).some((l) => l.length))
    parts.lists(panel, `o${atStr(a) || 0}`, it.lenses, unit, names, deps);
  if (subs.length) {                                  // a group's subgroups, also as a list (boxes can be too small to read)
    const ul = document.createElement("ul");
    ul.className = "tov-subs";
    for (const x of subs) {
      const li = ul.appendChild(document.createElement("li"));
      const b = li.appendChild(document.createElement("button"));
      b.type = "button"; b.className = "linklike";
      b.textContent = `${g!.id}.${x.id} ${x.name}`;
      b.onclick = () => pick({ g: g!.id, s: x.id });
      li.append(` · ${x.size.toLocaleString()} studies`);
    }
    panel.insertAdjacentHTML("beforeend", `<p class="tg-help">It divides into:</p>`);
    panel.append(ul);
  }
  if (g && !d.parent) {
    const a2 = panel.appendChild(document.createElement("a"));
    a2.className = "tg-split";
    a2.href = deps.url({ group: g.id });
    a2.textContent = subs.length ? `Open group ${g.id} as its own map →` : `Split group ${g.id} into smaller ones →`;
    a2.onclick = (e) => { e.preventDefault(); deps.go(a2.getAttribute("href")!); };
  }
}
