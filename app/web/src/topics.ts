/**
 * Topic maps: the research groups inside a search, and what to read in each (the data API's /topics; method in
 * search/srl_search/topics.py). Shown in the results view, instead of the list: one card per group, coloured like
 * its studies on the map, with reading lists as tabs. The page (app.ts) owns navigation and the map.
 */
import { API } from "./jobs.ts";
import { fieldsMade, fieldUrl, inTime, offerHtml, type FieldOffer } from "./fields.ts";
import { fetchWord, openWord, type WordAnswer } from "./words.ts";
import { renderOverview } from "./overview.ts";
import type { Hit } from "./paper.ts";

export interface TopicPaper {
  id: string; title: string; year?: number; type?: string; venue?: string; authors?: string[]; authors_count?: number;
  citations?: number; doi?: string; pmcid?: string; readable?: boolean; in_map?: boolean; why?: string;
  editions?: number;                         // author pages: the same work printed this many times (listed once)
}
export interface Expert { author: string; name: string; h: number; papers_in_group: number; top: TopicPaper; latest?: TopicPaper; why: string }
/** Someone an author publishes with (author pages, people.ts): n = papers together. */
export interface Coauthor { author?: string; institution?: string; country?: string | null; name: string; n: number; years: [number, number]; works?: number; top: TopicPaper; latest?: TopicPaper }
/** A word of a cloud: share = of the studies that use it (the size); lift = how many times as often as elsewhere
 *  (the other groups, or all of science for the whole topic: the colour); score = how telling it is. */
export interface CloudWord { t: string; share: number; lift: number; score: number; author?: string; institution?: string }
export type Lenses = Record<string, TopicPaper[]> & { experts: Expert[]; coauthors?: Coauthor[]; people?: Coauthor[]; partners?: Coauthor[];
  authors?: Coauthor[]; institutions?: Coauthor[]; _total?: Record<string, number> };
/** A group of a group, carried by the map (the data API makes the splits with the map). */
/** An AI-written name and description of a group (whole-field maps; search/srl_search/suggest.py). */
export interface Label { name: string; about?: string | null; by?: string }
export interface Sub {
  id: number; name: string; size: number; years?: number[] | null; typical?: TopicPaper; cites_groups?: Record<string, number>;
  cloud?: CloudWord[]; lenses: Lenses; label?: Label | null; words?: string;
}
export interface Group {
  id: number; name: string; size: number; years?: number[]; typical: TopicPaper; centre?: { x: number; y: number } | null;
  cites_groups: Record<string, number>; between: number; with_references: number; cloud?: CloudWord[];
  lenses: Lenses; subgroups?: Sub[]; subgroups_other?: number; label?: Label | null; words?: string;
}
/** The whole map read as one group (maps made before 2026-10-15 have none). */
export interface Overall { size: number; years?: number[] | null; cloud?: CloudWord[]; lenses: Lenses }
export interface TopicMap {
  question: string; mode: string; level?: string; groups: Group[]; overall?: Overall; cached?: boolean; papers_shown?: { shown: number; of: number }; parent?: { group: number; name: string; size: number };
  papers: { id: string[]; group: number[]; sub?: number[]; between: boolean[]; x: (number | null)[]; y: (number | null)[] };
  coverage: { papers: number; with_references: number; in_groups: number; other: number; references_thin: boolean;
              without_text_vector?: number };
  field?: { papers: number; made: string; credit?: string };
}
export interface TopicOptions { by?: "" | "meaning" | "words"; since?: string; group?: number | null; field?: boolean; definition?: string }
/** A whole-field map waiting or being made: the job's progress (data API /topics?scope=field answers 202). */
export interface FieldPending {
  status: string; key?: string; stage?: string; stage_index?: number; eta_s?: number; start_s?: number; position?: number;
  detail?: string; lane?: "paid" | "free" | "rebuild"; paid_eta_s?: number; gpu_reserved?: boolean; credit?: string;
}

export const GROUP_COLOURS = 9;   // style.css .g1 ... .g9 (a colour-blind-safe set: Okabe-Ito, Tol)

/** The data API's address of a map: the same for the map and for reading it further (its cache key). */
function mapParams(q: string, o: TopicOptions): URLSearchParams {
  const p = new URLSearchParams({ q });
  if (o.field) p.set("scope", "field");
  if (o.field && o.definition) p.set("definition", o.definition);
  if (o.by === "words") p.set("mode", "keyword"); else if (o.by === "meaning") p.set("mode", "semantic");
  if (o.since) p.set("year_from", o.since);
  if (o.group) p.set("group", String(o.group));
  return p;
}

/** A list read further: items from `offset` of a reading list of the map's whole (at.g null), a group, or a
 *  subgroup (the split map of its group). null: not available here (a subgroup inside a group's own map). */
export type More = (at: { g: number | null; s: number | null }, lens: string, offset: number, limit: number) =>
  Promise<{ items: TopicPaper[] & Expert[]; total: number } | null>;
export function moreOf(q: string, o: TopicOptions): More {
  return async (at, lens, offset, limit) => {
    const p = mapParams(q, o);
    let group: number | null = o.group ?? null, read: number;
    if (at.g == null) read = 0;
    else if (at.s == null) read = at.g;
    else if (group == null) { group = at.g; read = at.s; }
    else return null;
    if (group != null) p.set("group", String(group));
    p.set("read", String(read)); p.set("part", "more"); p.set("list", lens);
    p.set("offset", String(offset)); p.set("limit", String(limit));
    const r = await fetch(`${API}/api/data/topics?${p}`, { signal: AbortSignal.timeout(30000) });
    if (!r.ok) throw new TopicError(r.status, "");
    return r.json();
  };
}

/** A word of a unit's cloud: the unit's studies that use it (data API part=word), addressed like `moreOf`. */
export type WordFn = (at: { g: number | null; s: number | null }, word: string) => Promise<WordAnswer | null>;
export function wordOf(q: string, o: TopicOptions): WordFn {
  return async (at, word) => {
    const p = mapParams(q, o);
    let group: number | null = o.group ?? null, read: number;
    if (at.g == null) read = 0;
    else if (at.s == null) read = at.g;
    else if (group == null) { group = at.g; read = at.s; }
    else return null;
    if (group != null) p.set("group", String(group));
    p.set("read", String(read)); p.set("part", "word"); p.set("word", word); p.set("limit", "8");
    return fetchWord(p);
  };
}

export async function fetchTopics(q: string, o: TopicOptions, signal?: AbortSignal): Promise<TopicMap | { pending: FieldPending }> {
  const p = mapParams(q, o);
  const r = await fetch(`${API}/api/data/topics?${p}`, { signal: signal ?? AbortSignal.timeout(95000) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new TopicError(r.status, d.detail ?? d.error ?? "", Number(r.headers.get("Retry-After")) || d.retry_after);
  if (r.status === 202) return { pending: d.progress as FieldPending };
  return named(d as TopicMap);
}

/** Where an AI name exists, it becomes the group's name everywhere; its telltale phrases move to `words`. */
function named(d: TopicMap): TopicMap {
  const swap = (g: { name: string; label?: Label | null; words?: string }) => {
    if (g.label?.name && g.words === undefined) { g.words = g.name; g.name = g.label.name; }
  };
  for (const g of d.groups ?? []) { swap(g); for (const s of g.subgroups ?? []) swap(s); }
  const p = d.parent as (TopicMap["parent"] & { label?: Label | null }) | undefined;
  if (p?.label?.name) p.name = p.label.name;
  return d;
}

export class TopicError extends Error {
  constructor(public status: number, public detail: string, public retry?: number) { super(detail || String(status)); }
}

export const asHit = (p: TopicPaper): Hit => ({
  id: p.id, pmcid: p.pmcid ?? "", doi: p.doi, title: p.title, year: p.year ? String(p.year) : undefined,
  journal: p.venue, authors: (p.authors ?? []).join(", ") + ((p.authors_count ?? 0) > 3 ? ", …" : ""),
  citations: p.citations, type: p.type, readable: p.readable,
});

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
export const span = (y?: number[] | null) => y ? (y[0] === y[2] ? `${y[1]}` : `mostly ${y[0]}–${y[2]}`) : "";

// "this {u}": the group, or the whole topic (the same lists, from every study's citations). Core first: what the
// group stands on. "Overviews" (was "Start here"): recent studies citing most of the key works, often reviews; one
// way in among others, not an instruction.
const LENSES: [key: string, label: string, help: (u: string) => string][] = [
  ["core", "Core", (u) => `The works this ${u}'s studies cite most: its foundations.`],
  ["start_here", "Overviews", (u) => `Recent studies that cite most of this ${u}'s key works: often reviews that tie it together.`],
  ["rising", "Rising", (u) => `Works from the last five years already cited by many of this ${u}'s studies.`],
  ["distinctive", "Only here", () => "Works cited mostly by this group, not by the others: what sets it apart."],
  ["foundations", "From outside", (u) => `What this ${u}'s studies cite most that is not itself part of the field: methods, tools and classics from neighbouring fields.`],
  ["experts", "Experts", (u) => `The authors whose works this ${u} cites most (an h-index counted inside the ${u}: first authors get full credit, last authors half, others less).`],
  ["bridges", "Bridges", () => "Studies of this group with many links to another group."],
  // author pages (people.ts): "this whole" reads as all their work
  ["cited", "Most cited", (u) => u === "whole" ? "The most cited studies first." : `The most cited studies in this ${u}.`],
  ["first", "As first author", (u) => u === "whole" ? "Studies they are the first author of, the most cited first." : `Studies in this ${u} they are the first author of, the most cited first.`],
  ["newest", "Newest", (u) => u === "whole" ? "The latest studies." : `The latest studies in this ${u}.`],
  ["coauthors", "Co-authors", (u) => u === "whole" ? "The people they publish with most, and their most cited study together." : `The people they publish with most in this ${u}.`],
  // institution pages
  ["people", "People", (u) => u === "whole" ? "Its authors on the most of its studies, with their most cited one here." : `Its authors on the most of its studies in this ${u}.`],
  ["partners", "Partners", (u) => u === "whole" ? "The other organisations on its studies: who it works with most." : `The other organisations on its studies in this ${u}.`],
  // source pages (journals, conferences, repositories)
  ["authors", "Authors", (u) => u === "whole" ? "Who publishes here most, with their most cited study here." : `Who publishes most in this ${u} here.`],
  ["institutions", "Institutions", (u) => u === "whole" ? "Where its authors work: the organisations on the most of its studies." : `Where the authors of this ${u}'s studies work.`],
];

export interface Deps {
  row: (h: Hit, why?: string) => HTMLElement;
  go: (url: string) => void;
  focusGroup: (g: number | null, sub?: number | null) => void;
  at?: string;                               // the box picked in the overview ("", "3", "3.2"), from the address
  select?: (at: string) => void;             // remember it in the address
  url: (o: TopicOptions) => string;          // the address of this map with other options (a group, the whole map, the field)
  listUrl: string;                           // the plain list of studies for the same search
  offer?: FieldOffer | null;                 // paying to make a whole-field map now (fields.ts)
  paidNote?: string;                         // said once, after coming back from paying
  more?: More;                               // reading lists read further than the map gives
  word?: WordFn;                             // a word of a cloud, clicked: the studies that use it
}

/** The whole map on one screen (overview.ts); how it was made waits behind its "?". */
export function renderTopics(box: HTMLElement, d: TopicMap, deps: Deps) {
  box.replaceChildren();
  const wire = (el: HTMLElement) => {
    for (const a of el.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
      a.onclick = (e) => { e.preventDefault(); deps.go(a.getAttribute("href")!); };
  };
  if (!d.groups.length) {
    box.innerHTML = `<p>These studies don't split into clear groups. <a href="${esc(deps.listUrl)}" data-go>See them as a list</a>.</p>`;
    return wire(box);
  }
  const back = `<a href="${esc(deps.listUrl)}" data-go>The studies as a list</a>`;
  const info = document.createElement("div");
  let act: HTMLElement | null = null;
  const named = d.groups.some((g) => g.label);
  let made: string;
  if (d.parent) {
    made = `<p><b>Group ${d.parent.group}, ${esc(d.parent.name)}, grouped again:</b> its ${d.parent.size.toLocaleString()} ` +
      `studies in ${d.groups.length} smaller groups.</p>`;
    act = link(deps.url({ group: null }), "← The whole map", deps);
  } else if (d.mode === "field") {
    made = `<p><b>${d.coverage.papers.toLocaleString()} studies in ${d.groups.length} research groups.</b> Every study we ` +
      `could find on this topic: close in meaning and judged one by one by an AI, or cited by many of the field's own ` +
      `studies (which brings in classics that have no abstract). Big groups are split into smaller ones.</p>` +
      `<p class="quiet">How complete? We tested this method once, on habitat fragmentation: it found about 9 in 10 of the ` +
      `studies cited by 19 published reviews of that field.</p>` +
      (d.papers_shown ? `<p class="quiet">To keep the page fast, the dots on the map are a random sample of its studies; ` +
        `the counts, groups and reading lists cover all ${d.papers_shown.of.toLocaleString()}.</p>` : "") +
      (d.field?.credit ? `<p class="quiet">Made sooner thanks to ${esc(d.field.credit)}.</p>` : "");
  } else {
    const how = d.mode === "keyword" ? "studies containing your words" : "studies closest in meaning to your search";
    made = `<p><b>The ${d.coverage.papers.toLocaleString()} ${how}, in ${d.groups.length} research groups.</b></p>` +
      `<div class="tv-opts"></div>` +
      `<p class="quiet"><a href="?fields&ask=${encodeURIComponent(d.question)}" data-go>Map the whole field instead</a>: ` +
      `every study on this topic, from thousands to a million, not just the closest 1,000. The first time takes from half ` +
      `an hour to two hours; then it opens at once for everyone.</p>`;
    act = link(`?fields&ask=${encodeURIComponent(d.question)}`, "Map the whole field", deps);
    fieldsMade(d.question, 1).then(([m]) => {      // a whole-field map that already holds most of these: go there instead
      if (!m || (m.score ?? 0) < NEAR_FIELD || !act?.isConnected) return;
      const a = act as HTMLAnchorElement;
      a.href = fieldUrl(m.topic, m.definition);
      a.textContent = `Whole-field map: ${m.topic}`;
      a.title = `Most of these studies are in a whole-field map already made: ${m.topic} (${(m.papers ?? 0).toLocaleString()} studies)`;
    });
  }
  info.innerHTML = made +
    `<p>Studies land in the same group when they cite the same works and use the same words. Each box is a group, ` +
    `sized by its studies; the boxes inside it are its subgroups. Point at one to see what it is about and light its ` +
    `studies on the map; click it to keep it, open its subgroups and read its lists. Esc or ↑ goes back up.</p>` +
    `<p class="quiet">${named ? "Group names and descriptions are written by an AI from each group's studies. " :
      "Each group is named by the words that set it apart, chosen automatically. "}The reading lists come from what the ` +
    `group's own studies cite: what matters to <em>that</em> group.</p>` +
    (d.coverage.other ? `<p class="quiet">Grey: ${d.coverage.other.toLocaleString()} studies that fit no group well.</p>` : "") +
    (d.coverage.references_thin ? `<p class="quiet">Only ${d.coverage.with_references.toLocaleString()} of these studies ` +
      `list their references in our data, so here the groups rely more on wording.</p>` : "") +
    `<p class="tv-links">${d.mode === "field" && !d.parent ? `<a href="${esc(deps.url({ field: false }))}" data-go>The 1,000 closest instead</a> · ` : ""}` +
    `${back} · <a href="?fields" data-go>All whole-field maps</a></p>`;
  const opts = info.querySelector<HTMLElement>(".tv-opts");
  if (opts) searchOpts(opts, d, deps);
  wire(info);
  renderOverview(box, d, deps, { cloud, lists, span, info, act, splitFrom: d.mode === "field" ? 300 : 60 });
  if (deps.word) {
    const word = deps.word;
    box.addEventListener("click", (e) => {
      const b = (e.target as HTMLElement).closest<HTMLButtonElement>(".tg-word");
      const host = b?.closest<HTMLElement>("[data-at]");
      if (!b || !host) return;
      const [g, s] = (host.dataset.at ?? "").split(".").map((x) => x === "" || x == null ? null : Number(x));
      const at = { g: g ?? null, s: s ?? null };
      const g0 = at.g == null ? null : d.groups.find((x) => x.id === at.g);
      const s0 = g0 && at.s != null ? g0.subgroups?.find((x) => x.id === at.s) : null;
      const unit = s0 ? `“${s0.name}”` : g0 ? `“${g0.name}”` : d.mode === "field" ? "the whole field" : "these studies";
      openWord({ word: b.textContent ?? "", share: Number(b.dataset.share), lift: Number(b.dataset.lift), vs: b.dataset.vs ?? "",
                 unit, topic: d.parent?.name ?? d.question, fetch: () => word(at, b.textContent ?? "") }, deps.go);
    });
  }
}

function link(href: string, text: string, deps: Deps): HTMLAnchorElement {
  const a = document.createElement("a");
  a.href = href;
  a.textContent = text;
  a.onclick = (e) => { e.preventDefault(); deps.go(a.getAttribute("href")!); };
  return a;
}

/** A map of the closest studies is made from a search: by meaning or exact words, from a year on. */
function searchOpts(host: HTMLElement, d: TopicMap, deps: Deps) {
  const p = new URLSearchParams(location.search);
  const by = d.mode === "keyword" ? "words" : "meaning", since = p.get("since") ?? "";
  host.innerHTML = `<span class="seg" role="radiogroup" aria-label="Made from studies found">` +
    `<button type="button" role="radio" data-by="meaning" aria-checked="${by === "meaning"}">By meaning</button>` +
    `<button type="button" role="radio" data-by="words" aria-checked="${by === "words"}">Exact words</button></span> ` +
    `<select aria-label="Published"><option value="">Any year</option><option value="2024">Since 2024</option>` +
    `<option value="2020">Since 2020</option><option value="2015">Since 2015</option></select>`;
  const sel = host.querySelector("select")!;
  sel.value = since;
  const to = (o: TopicOptions) => deps.go(deps.url(o));
  for (const b of host.querySelectorAll<HTMLButtonElement>("[data-by]"))
    b.onclick = () => { if (b.dataset.by !== by) to({ by: b.dataset.by as "meaning" | "words" }); };
  sel.onchange = () => to({ since: sel.value });
}

const pct = (x: number) => x >= 0.095 ? `${Math.round(100 * x)}%` : x >= 0.0095 ? `${(100 * x).toFixed(1)}%`
  : `${Number((100 * x).toPrecision(1))}%`;

/** Scimeetr's word cloud (2017): a set of studies' most frequent words, sized by how many of them use the word,
 *  coloured by how much more (blue) or less (red) than elsewhere they use it; grey: about as much. The colour is
 *  on a log scale up to `cap` times (as often or as rarely): x8 against the other groups, x1000 against all of
 *  science (a topic's own words are thousands of times commoner in it). A wrapped line of words, the biggest in
 *  the middle, so it reads as text (and to screen readers in order of size, from the centre out). */
export function cloud(words: CloudWord[], k: number, vs: string, cap = vs === "all of science" ? 1000 : 8): HTMLElement {
  const w = words.slice(0, k);
  const lo = Math.min(...w.map((x) => x.share)), hi = Math.max(...w.map((x) => x.share));
  const big = k >= 30 ? 27 : 22;
  const order: CloudWord[] = [];                      // ranks 1, 2, 3, 4, 5 ... as 5 3 1 2 4: big in the middle
  [...w].sort((a, b) => b.share - a.share).forEach((x, i) => (i % 2 ? order.push(x) : order.unshift(x)));
  const div = document.createElement("div");
  div.className = "tg-cloud-box";
  const ul = div.appendChild(document.createElement("ul"));
  ul.className = "tg-cloud";
  ul.setAttribute("aria-label", `The words its studies use most, compared with ${vs}`);
  for (const x of order) {
    const t = hi > lo ? Math.sqrt((x.share - lo) / (hi - lo)) : 1;
    const li = ul.appendChild(document.createElement("li"));
    const b = li.appendChild(document.createElement("button"));   // a word opens its card (words.ts)
    b.type = "button"; b.className = "tg-word"; b.textContent = x.t;
    b.dataset.share = String(x.share); b.dataset.lift = String(x.lift); b.dataset.vs = vs;
    li.style.fontSize = `${(12.5 + (big - 12.5) * t).toFixed(1)}px`;
    const l = Math.log(Math.max(x.lift, 1e-6));
    if (Math.abs(l) >= Math.log(1.25)) {             // grey (the text's own muted colour) within 1.25x either way
      const mix = Math.round(100 * Math.min(1, 0.25 + 0.75 * Math.abs(l) / Math.log(cap)));
      li.style.color = `color-mix(in oklab, var(${l > 0 ? "--cloud-up" : "--cloud-down"}) ${mix}%, var(--muted))`;
    }
    const elsewhere = x.share / x.lift;
    li.title = `In ${pct(x.share)} of these studies; ${pct(elsewhere)} in ${vs}`;
  }
  return div;
}

/** Reading lists as tabs: of a group (key "3"), a subgroup ("3.2") or the whole topic ("0"); `want` opens that
 *  list if the unit has it (the reader's last choice carries over as they move between groups). */
export function lists(host: HTMLElement, key: string, lenses: Lenses, unit: string, names: Map<number, string>, deps: Deps,
                      want = "core", onPick: (lens: string) => void = () => {},
                      further?: (lens: string, offset: number, limit: number) => ReturnType<More>) {
  const present = LENSES.filter(([k]) => (lenses[k] ?? []).length);
  const tabs = host.appendChild(document.createElement("div"));
  tabs.className = "tg-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", unit === "whole" ? "Lists of all their work" : `Reading lists of this ${unit}`);
  const help = host.appendChild(document.createElement("p"));
  help.className = "tg-help";
  const panel = host.appendChild(document.createElement("div"));
  panel.className = "tg-panel";
  panel.setAttribute("role", "tabpanel");
  panel.id = `tgp-${key}`;
  const buttons: HTMLButtonElement[] = [];
  const pick = (i: number, focus = false) => {
    buttons.forEach((b, j) => { b.setAttribute("aria-selected", String(i === j)); b.tabIndex = i === j ? 0 : -1; });
    if (focus) buttons[i].focus();
    const [k, , say] = present[i];
    panel.setAttribute("aria-labelledby", buttons[i].id);
    help.textContent = say(unit);
    help.title = say(unit);
    const rows = (list: (TopicPaper | Expert | Coauthor)[]) => k === "experts"
      ? (list as Expert[]).map((e) => expert(e, deps, unit)) : k === "coauthors" || k === "people" || k === "partners" || k === "authors" || k === "institutions" ? (list as Coauthor[]).map((e) => coauthor(e, deps, k))
      : (list as TopicPaper[]).map((p) => deps.row(asHit(p), why(k, p, names)));
    const first = (lenses[k] ?? []) as (TopicPaper | Expert | Coauthor)[];
    const total = lenses._total?.[k] ?? 0;      // author and institution pages say how long each list is
    pages.set(rows(first), further && (first.length >= HEAD || total > first.length) ? async (offset, limit) => {
      const r = await further(k, offset, limit);
      return r ? { rows: rows(r.items), total: r.total } : null;
    } : undefined);
  };
  const pages = pager(host, panel, unit === "whole" ? "list of all their work" : `Reading list of this ${unit}`);
  present.forEach(([k, label, say], i) => {
    const b = tabs.appendChild(document.createElement("button"));
    b.type = "button";
    b.id = `tgt-${key}-${i}`;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-controls", panel.id);
    b.title = say(unit);
    b.textContent = label;
    b.onclick = () => { onPick(k); pick(i); };
    b.onkeydown = (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const j = (i + (e.key === "ArrowRight" ? 1 : present.length - 1)) % present.length;
        onPick(present[j][0]);
        pick(j, true);
      }
    };
    buttons.push(b);
  });
  if (present.length) pick(Math.max(0, present.findIndex(([k]) => k === want)));
}

/** Where the page has a fixed height (style.css: wide and tall enough), a list shows in pages that fit its space,
 *  with ‹ › underneath, rather than scrolling inside the page; elsewhere it shows whole and the page scrolls.
 *  A list longer than the map gives is read further as it is read: up to BATCH items once it has been open LINGER ms, then the
 *  next BATCH once the page shown comes within AHEAD items of the end (the next page is there before it is asked
 *  for); one column: "Show more" under it. */
const PAGED = "(min-width: 1101px) and (min-height: 680px)";
const HEAD = 8;      // items of a list in the map itself (the data API's Options.lens_size)
const BATCH = 25;
const AHEAD = 9;
const LINGER = 1500;  // ms a list stays open before it is read further (skimming groups and tabs asks for nothing)
type Load = (offset: number, limit: number) => Promise<{ rows: HTMLElement[]; total: number } | null>;
function pager(host: HTMLElement, panel: HTMLElement, what: string) {
  const bar = host.appendChild(document.createElement("div"));
  bar.className = "tg-pager";
  bar.setAttribute("role", "group");
  bar.setAttribute("aria-label", `Pages of the ${what.toLowerCase()}`);
  bar.innerHTML = `<button type="button" class="tg-prev" aria-label="Previous page">‹</button>` +
    `<span class="tg-page" aria-live="polite"></span><button type="button" class="tg-next" aria-label="Next page">›</button>`;
  const prev = bar.querySelector<HTMLButtonElement>(".tg-prev")!, next = bar.querySelector<HTMLButtonElement>(".tg-next")!;
  const label = bar.querySelector<HTMLElement>(".tg-page")!;
  const showMore = host.appendChild(document.createElement("button"));
  showMore.type = "button";
  showMore.className = "tg-showmore";
  showMore.textContent = "Show more";
  showMore.hidden = true;
  let items: HTMLElement[] = [], starts: number[] = [0], page = 0, room = 0;
  let load: Load | undefined, done = true, busy = false, run = 0, opened = 0, wait = 0;
  const paged = () => matchMedia(PAGED).matches && panel.clientHeight > 0;
  /** The next batch, to a multiple of BATCH (8 -> 25 -> 50 ...). Kept only if the same list is still shown. */
  const fetchMore = async () => {
    if (!load || done || busy) return;
    const early = LINGER - (performance.now() - opened);
    if (early > 0) {                                   // just opened: later, if it is still this list
      clearTimeout(wait);
      const mine = run;
      wait = window.setTimeout(() => { if (mine === run && panel.isConnected) void fetchMore(); }, early);
      return;
    }
    busy = true;
    const mine = run, limit = (Math.floor(items.length / BATCH) + 1) * BATCH - items.length;
    try {
      const r = await load(items.length, limit);
      if (mine !== run) return;
      if (!r || !r.rows.length) { done = true; return; }
      items.push(...r.rows);
      done = items.length >= r.total;
      cut();
    } catch {
      /* the network or the service: tried again at the next page */
    } finally {
      if (mine === run) { busy = false; showMore.hidden = done || paged(); }
    }
  };
  const show = (p: number) => {
    page = Math.max(0, Math.min(starts.length - 1, p));
    const a = starts[page], b = starts[page + 1] ?? items.length;
    panel.replaceChildren(...items.slice(a, b));
    label.textContent = `${b - a > 1 ? `${a + 1}–${b}` : a + 1} of ${items.length}`;
    prev.disabled = page === 0;
    next.disabled = page === starts.length - 1;
    if (b >= items.length - AHEAD) void fetchMore();
  };
  /** Cut the list into pages: as many rows as fit, one after another (rows differ in height). */
  const cut = () => {
    const first = starts[page] ?? 0;
    room = panel.clientHeight;
    if (!paged()) {
      starts = [0]; page = 0;
      panel.replaceChildren(...items);
      bar.hidden = true;
      showMore.hidden = done;
      return;
    }
    showMore.hidden = true;
    starts = [0];
    panel.replaceChildren();
    items.forEach((el, i) => {
      panel.append(el);
      if (panel.scrollHeight > room + 1 && panel.childElementCount > 1) { starts.push(i); panel.replaceChildren(el); }
    });
    bar.hidden = starts.length < 2 && done;
    let p = 0;
    while (p + 1 < starts.length && starts[p + 1] <= first) p++;
    show(p);
  };
  prev.onclick = () => show(page - 1);
  next.onclick = () => show(page + 1);
  showMore.onclick = () => { opened = 0; showMore.disabled = true; void fetchMore().finally(() => { showMore.disabled = false; }); };
  const ro = new ResizeObserver(() => {
    if (!panel.isConnected) return ro.disconnect();
    if (panel.clientHeight !== room) cut();
  });
  ro.observe(panel);
  document.fonts?.ready.then(() => { if (panel.isConnected) cut(); });   // rows change height once the fonts are in
  return {
    set(list: HTMLElement[], more?: Load) {
      run++; busy = false; clearTimeout(wait); opened = performance.now();
      items = list; starts = [0]; page = 0; load = more; done = !more;
      cut();
      if (more) void fetchMore();                     // the first BATCH as soon as the list opens
    },
  };
}

function why(key: string, p: TopicPaper, names: Map<number, string>): string {
  if (p.editions) return `${p.editions} editions with this title (reprints, translations, collected works): listed once, the most cited`;
  if (key === "newest") return "";
  if (key === "bridges") {
    const m = /group (\d+)/.exec(p.why ?? "");
    return m ? `${p.why!.replace(/ go to group \d+/, "")} go to group ${m[1]}: ${names.get(Number(m[1])) ?? ""}` : p.why ?? "";
  }
  const out = (p.why ?? "").replace(/^cited by/, "Cited by").replace(/^cites/, "Cites").replace(/^from/, "From");
  return p.in_map === false && key !== "start_here" && key !== "foundations" ? `${out} · not among the mapped studies (no English abstract, or older)` : out;
}

function expert(e: Expert, deps: Deps, unit = "group"): HTMLElement {
  const div = document.createElement("div");
  div.className = "tg-expert";
  const paper = (p: TopicPaper | undefined) => p
    ? `<a href="?work=${encodeURIComponent(p.id)}" data-go>${esc(p.title)}</a> <span class="quiet">(${p.year ?? "n.d."})</span>` : "";
  div.innerHTML = `<span class="hit-title"><a href="?author=${esc(e.author)}" data-go>${esc(e.name || e.author)}</a></span>` +
    `<span class="hit-meta">h = ${e.h} inside this ${unit}` + (e.papers_in_group ? ` · author of ${e.papers_in_group} of its studies` : "") + `</span>` +
    `<span class="hit-why">Most cited here: ${paper(e.top)}` +
    (e.latest && e.latest.id !== e.top.id ? `<br>Latest here: ${paper(e.latest)}` : "") + `</span>`;
  for (const a of div.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
    a.onclick = (ev) => { ev.preventDefault(); deps.go(a.getAttribute("href")!); };
  return div;
}

/** A co-author on an author's page, or a person or partner organisation on an institution's: the name (a link to
 *  its own page), papers together, the best one. */
function coauthor(e: Coauthor, deps: Deps, k = "coauthors"): HTMLElement {
  const div = document.createElement("div");
  div.className = "tg-expert";
  const paper = (p: TopicPaper | undefined) => p
    ? `<a href="?work=${encodeURIComponent(p.id)}" data-go>${esc(p.title)}</a> <span class="quiet">(${p.year ?? "n.d."})</span>` : "";
  const span = e.years[0] === e.years[1] ? `${e.years[0]}` : `${e.years[0]}–${e.years[1]}`;
  const href = e.institution ? `?institution=${esc(e.institution)}` : `?author=${esc(e.author ?? "")}`;
  const here = k === "coauthors" ? "together" : "here";
  div.innerHTML = `<span class="hit-title"><a href="${href}" data-go>${esc(e.name || e.author || e.institution || "")}</a>` +
    (e.country ? ` <span class="quiet">(${esc(e.country)})</span>` : "") + `</span>` +
    `<span class="hit-meta">${e.n === 1 ? `1 study ${here}` : `${e.n.toLocaleString()} studies ${here}`} · ${span}` +
    (e.works != null ? ` · ${e.works.toLocaleString()} studies in all` : "") + `</span>` +
    `<span class="hit-why">Most cited ${here}: ${paper(e.top)}` +
    (e.latest && e.latest.id !== e.top.id ? `<br>Latest ${here}: ${paper(e.latest)}` : "") + `</span>`;
  for (const a of div.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
    a.onclick = (ev) => { ev.preventDefault(); deps.go(a.getAttribute("href")!); };
  return div;
}

const STAGE_NAMES: Record<string, string> = {
  archetypes: "judging the studies closest to the topic", pool: "gathering hundreds of thousands of nearby studies",
  labels: "judging a sample, near and far", classifier: "training a classifier on those judgments",
  phrases: "adding studies with the topic's exact words", citations: "following what the field cites",
  map: "grouping the whole field", subgroups: "splitting the big groups",
};

/** A whole-field map holding at least this share of a question's 50 closest studies (the data API's /fields score)
 *  is offered next to its 1,000-paper map (related questions score 0.8-1, unrelated ones under 0.1). */
const NEAR_FIELD = 0.5;
const nth = (n: number) => n === 1 ? "first" : n === 2 ? "second" : n === 3 ? "third" : `${n}th`;

/** While a whole-field map waits in line or is being made. */
export function renderPending(box: HTMLElement, st: FieldPending, deps: Deps) {
  const eta = st.eta_s ?? 1800;
  const line = st.lane === "paid" ? "paid line" : "free line";
  const now = st.status === "queued"
    ? `${st.position ? `${nth(st.position)[0].toUpperCase()}${nth(st.position).slice(1)} in the ${line}` : `Waiting in the ${line}`}. ` +
      `Ready in ${inTime(eta)}.`
    : `Step ${(st.stage_index ?? 0) + 1} of 8: ${STAGE_NAMES[st.stage ?? ""] ?? "working"}. Ready in ${inTime(eta)}.`;
  const offer = st.status === "queued" && st.lane !== "paid" && st.key
    ? offerHtml(st.key, eta, st.paid_eta_s, !!st.gpu_reserved, deps.offer ?? null) : "";
  box.innerHTML = `<h2 class="topics-title">Mapping the whole field</h2>` +
    (deps.paidNote ? `<p class="field-paid-note">${esc(deps.paidNote)}</p>` : "") +
    `<p class="checking">${esc(now)}</p>` + offer +
    `<p class="topics-intro">This finds every study on the topic, not just the closest 1,000: studies close in meaning, ` +
    `each judged by an AI, plus the studies the field keeps citing, then groups them all. You can leave: it keeps going, ` +
    `and the map will open at once next time, for anyone. This page checks every 20 seconds.</p>` +
    (st.gpu_reserved ? `<p class="quiet small">Our graphics card is busy with other work right now, so maps wait ` +
      `longer than shown, until it frees up.</p>` : "") +
    `<p class="quiet"><a href="${esc(deps.url({ field: false }))}" data-go>See the map of the 1,000 closest meanwhile</a> · ` +
    `<a href="?fields" data-go>The whole line</a></p>`;
  for (const a of box.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
    a.onclick = (e) => { e.preventDefault(); deps.go(a.getAttribute("href")!); };
}
