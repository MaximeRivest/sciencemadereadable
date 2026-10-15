/**
 * Topic maps: the research groups inside a search, and what to read in each (the data API's /topics; method in
 * search/srl_search/topics.py). Shown in the results view, instead of the list: one card per group, coloured like
 * its studies on the map, with reading lists as tabs. The page (app.ts) owns navigation and the map.
 */
import { API } from "./jobs.ts";
import { fieldsMade, fieldUrl, inTime, offerHtml, type FieldOffer } from "./fields.ts";
import type { Hit } from "./paper.ts";

export interface TopicPaper {
  id: string; title: string; year?: number; type?: string; venue?: string; authors?: string[]; authors_count?: number;
  citations?: number; doi?: string; pmcid?: string; readable?: boolean; in_map?: boolean; why?: string;
}
export interface Expert { author: string; name: string; h: number; papers_in_group: number; top: TopicPaper; latest?: TopicPaper; why: string }
/** A word of a cloud: share = of the studies that use it; lift = how many times more often than elsewhere (the
 *  other groups, or all of science for the whole topic); score = its relevance, what the size follows. */
export interface CloudWord { t: string; share: number; lift: number; score: number }
type Lenses = Record<string, TopicPaper[]> & { experts: Expert[] };
export interface Group {
  id: number; name: string; size: number; years?: number[]; typical: TopicPaper; centre?: { x: number; y: number } | null;
  cites_groups: Record<string, number>; between: number; with_references: number; cloud?: CloudWord[];
  lenses: Lenses;
}
/** The whole map read as one group (maps made before 2026-10-15 have none). */
export interface Overall { size: number; years?: number[] | null; cloud?: CloudWord[]; lenses: Lenses }
export interface TopicMap {
  question: string; mode: string; groups: Group[]; overall?: Overall; cached?: boolean; parent?: { group: number; name: string; size: number };
  papers: { id: string[]; group: number[]; between: boolean[]; x: (number | null)[]; y: (number | null)[] };
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

export async function fetchTopics(q: string, o: TopicOptions, signal?: AbortSignal): Promise<TopicMap | { pending: FieldPending }> {
  const p = new URLSearchParams({ q });
  if (o.field) p.set("scope", "field");
  if (o.field && o.definition) p.set("definition", o.definition);
  if (o.by === "words") p.set("mode", "keyword"); else if (o.by === "meaning") p.set("mode", "semantic");
  if (o.since) p.set("year_from", o.since);
  if (o.group) p.set("group", String(o.group));
  const r = await fetch(`${API}/api/data/topics?${p}`, { signal: signal ?? AbortSignal.timeout(95000) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new TopicError(r.status, d.detail ?? d.error ?? "", Number(r.headers.get("Retry-After")) || d.retry_after);
  if (r.status === 202) return { pending: d.progress as FieldPending };
  return d as TopicMap;
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
const span = (y?: number[]) => y ? (y[0] === y[2] ? `${y[1]}` : `mostly ${y[0]}–${y[2]}`) : "";

// "this {u}": the group, or the whole topic (the same lists, from every study's citations)
const LENSES: [key: string, label: string, help: (u: string) => string][] = [
  ["start_here", "Start here", (u) => `Recent studies of this ${u} that cite most of its key works: often reviews, a good first read.`],
  ["core", "Core", (u) => `The works this ${u}'s studies cite most: its foundations.`],
  ["rising", "Rising", (u) => `Works from the last five years already cited by many of this ${u}'s studies.`],
  ["distinctive", "Only here", () => "Works cited mostly by this group, not by the others: what sets it apart."],
  ["experts", "Experts", (u) => `The authors whose works this ${u} cites most (an h-index counted inside the ${u}: first authors get full credit, last authors half, others less).`],
  ["bridges", "Bridges", () => "Studies of this group with many links to another group."],
  ["newest", "Newest", (u) => `The latest studies in this ${u}.`],
];

export interface Deps {
  row: (h: Hit, why?: string) => HTMLElement;
  go: (url: string) => void;
  focusGroup: (g: number | null) => void;
  url: (o: TopicOptions) => string;          // the address of this map with other options (a group, the whole map, the field)
  listUrl: string;                           // the plain list of studies for the same search
  offer?: FieldOffer | null;                 // paying to make a whole-field map now (fields.ts)
  paidNote?: string;                         // said once, after coming back from paying
}

/** The whole map: a sentence on what was done, then one card per group. */
export function renderTopics(box: HTMLElement, d: TopicMap, deps: Deps) {
  box.replaceChildren();
  const head = box.appendChild(document.createElement("div"));
  head.className = "topics-head";
  const how = d.mode === "keyword" ? "studies containing these words" : "studies closest in meaning";
  const back = `<a href="${esc(deps.listUrl)}" data-go>Back to the list</a>`;
  if (d.parent) {
    head.innerHTML = `<p class="quiet search-how"><a href="${esc(deps.url({ group: null }))}" data-go>← The whole map</a> · ${back}</p>` +
      `<h2 class="topics-title"><span class="tg-chip g${d.parent.group}"></span>Inside group ${d.parent.group}: ${esc(d.parent.name)}</h2>` +
      `<p class="quiet">Its ${d.parent.size.toLocaleString()} studies, grouped again: ${d.groups.length} smaller groups.</p>`;
  } else if (d.mode === "field") {
    head.innerHTML = `<p class="quiet search-how">The whole field: ${d.coverage.papers.toLocaleString()} studies in ${d.groups.length} research groups · ` +
      `<a href="${esc(deps.url({ field: false }))}" data-go>The 1,000 closest instead</a> · ${back}</p>` +
      `<p class="topics-intro">Every study we could find on this topic: close in meaning and judged one by one by an AI, ` +
      `or cited by many of the field's own studies (which brings in classics without an abstract). Checked once against ` +
      `19 published reviews of habitat fragmentation, this way found about 9 in 10 of the studies they cite. Big groups ` +
      `are split into smaller ones.</p>` +
      (d.field?.credit ? `<p class="quiet small">Made sooner thanks to ${esc(d.field.credit)}.</p>` : "") +
      `<p class="quiet small"><a href="?fields" data-go>All whole-field maps</a></p>`;
  } else {
    head.innerHTML = `<p class="quiet search-how">${d.coverage.papers.toLocaleString()} ${how}, in ${d.groups.length} research groups · ${back}</p>` +
      `<p class="topics-intro">Studies land in the same group when they cite the same works and talk about the same things. ` +
      `Each group's name is made of the words that set it apart (chosen automatically), and its reading lists come from ` +
      `what its own studies cite: what matters to <em>that</em> group.</p>` +
      `<p class="quiet small"><a href="${esc(deps.url({ field: true, group: null }))}" data-go>Map the whole field instead</a>: every ` +
      `study on this topic, often tens of thousands, not just the closest 1,000. The first time takes about half an hour; then ` +
      `it opens at once for everyone.</p><p class="quiet small" id="field-near"></p>`;
    fieldsMade(d.question, 1).then(([m]) => {
      const el = head.querySelector("#field-near");
      if (!el || !m || (m.score ?? 0) < NEAR_FIELD) return;
      el.innerHTML = `Most of these studies are in a whole-field map already made: <a href="${esc(fieldUrl(m.topic, m.definition))}" ` +
        `data-go>${esc(m.topic)}</a> (${(m.papers ?? 0).toLocaleString()} studies).`;
      const a = el.querySelector("a")!;
      a.onclick = (ev) => { ev.preventDefault(); deps.go(a.getAttribute("href")!); };
    });
  }
  if (d.coverage.references_thin)
    head.insertAdjacentHTML("beforeend", `<p class="quiet small">Only ${d.coverage.with_references.toLocaleString()} of these studies list their references in our data, so here the groups rely more on wording.</p>`);
  if (!d.groups.length) {
    head.insertAdjacentHTML("beforeend", `<p>These studies don't split into clear groups. <a href="${esc(deps.listUrl)}" data-go>See them as a list</a>.</p>`);
  }
  const names = new Map(d.groups.map((g) => [g.id, g.name]));
  if (d.overall && Object.values(d.overall.lenses).some((l) => l.length)) box.appendChild(whole(d, names, deps));
  for (const g of d.groups) box.appendChild(card(g, names, deps, !d.parent, d.mode === "field" ? 300 : 60));
  if (d.coverage.other)
    box.insertAdjacentHTML("beforeend", `<p class="quiet small topics-foot">${d.coverage.other.toLocaleString()} studies fit no group well (grey on the map).</p>`);
  for (const a of box.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
    a.onclick = (e) => { e.preventDefault(); deps.go(a.getAttribute("href")!); };
}

/** The whole topic as one card, above its groups: the words that set it apart from the rest of science, and
 *  its own reading lists (from every study's citations). */
function whole(d: TopicMap, names: Map<number, string>, deps: Deps): HTMLElement {
  const o = d.overall!;
  const sec = document.createElement("section");
  sec.className = "tgroup tg-whole";
  sec.setAttribute("aria-labelledby", "tg-0");
  const what = d.parent ? `All of group ${d.parent.group}` : d.mode === "field" ? "The whole field" : "The whole topic";
  const meta = [`${o.size.toLocaleString()} studies`, span(o.years ?? undefined)].filter(Boolean).join(" · ");
  sec.innerHTML = `<h3 id="tg-0">${esc(what)}</h3><p class="tg-meta">${meta}</p>`;
  if (o.cloud?.length)
    sec.append(cloud(o.cloud, 30, "all of science",
      `Its words, sized by how much more its studies use them than all of science does (point at one for the numbers).`));
  lists(sec, 0, o.lenses, "topic", names, deps);
  return sec;
}

function card(g: Group, names: Map<number, string>, deps: Deps, canSplit: boolean, splitFrom: number): HTMLElement {
  const sec = document.createElement("section");
  sec.className = `tgroup g${((g.id - 1) % GROUP_COLOURS) + 1}`;
  sec.setAttribute("aria-labelledby", `tg-${g.id}`);
  const uses = Object.entries(g.cites_groups ?? {}).slice(0, 1)
    .map(([k]) => `builds on group ${k} (${esc(names.get(Number(k)) ?? "")})`);
  const meta = [`${g.size.toLocaleString()} studies`, span(g.years), ...uses].filter(Boolean).join(" · ");
  sec.innerHTML = `<h3 id="tg-${g.id}"><span class="tg-chip" aria-hidden="true"></span><span class="tg-num">Group ${g.id}</span> ${esc(g.name)}</h3>` +
    `<p class="tg-meta">${meta}</p>`;
  sec.onpointerenter = () => deps.focusGroup(g.id);
  sec.onpointerleave = () => deps.focusGroup(null);
  sec.addEventListener("focusin", () => deps.focusGroup(g.id));
  if (g.cloud?.length) sec.append(cloud(g.cloud, 24, "the other groups"));
  lists(sec, g.id, g.lenses, "group", names, deps);
  if (canSplit && g.size >= splitFrom) {
    const a = sec.appendChild(document.createElement("a"));
    a.className = "tg-split";
    a.href = deps.url({ group: g.id });
    a.textContent = `Split this group into smaller ones →`;
    a.onclick = (e) => { e.preventDefault(); deps.go(a.getAttribute("href")!); };
  }
  return sec;
}

const times = (x: number) => x >= 100 ? `${Number(x.toPrecision(2)).toLocaleString()}×` : `${x >= 10 ? Math.round(x) : x.toFixed(1)}×`;

/** A word cloud in Scimeetr's spirit: the words a set of studies uses more than others, the bigger the more
 *  telling (their relevance: how often, and how many times more often). A wrapped line of words, the biggest in
 *  the middle, so it reads as text (and to screen readers in order of size, from the centre out). */
function cloud(words: CloudWord[], k: number, vs: string, note?: string): HTMLElement {
  const w = words.slice(0, k);
  const lo = Math.min(...w.map((x) => x.score)), hi = Math.max(...w.map((x) => x.score));
  const big = k >= 30 ? 27 : 22;
  const order: CloudWord[] = [];                      // ranks 1, 2, 3, 4, 5 ... as 5 3 1 2 4: big in the middle
  w.forEach((x, i) => (i % 2 ? order.push(x) : order.unshift(x)));
  const div = document.createElement("div");
  div.className = "tg-cloud-box";
  if (note) div.insertAdjacentHTML("beforeend", `<p class="tg-help">${esc(note)}</p>`);
  const ul = div.appendChild(document.createElement("ul"));
  ul.className = "tg-cloud";
  ul.setAttribute("aria-label", `Words used more here than in ${vs}`);
  for (const x of order) {
    const t = hi > lo ? (x.score - lo) / (hi - lo) : 1;
    const li = ul.appendChild(document.createElement("li"));
    li.textContent = x.t;
    li.style.fontSize = `${(12.5 + (big - 12.5) * t).toFixed(1)}px`;
    if (t < 0.3) li.className = "small";
    li.title = `In ${Math.max(1, Math.round(100 * x.share))}% of these studies, ${times(x.lift)} as often as in ${vs}`;
  }
  return div;
}

/** Reading lists as tabs (a group's, or the whole topic's: id 0). */
function lists(sec: HTMLElement, id: number, lenses: Lenses, unit: string, names: Map<number, string>, deps: Deps) {
  const g = { id, lenses };
  const present = LENSES.filter(([k]) => (g.lenses[k] ?? []).length);
  const tabs = sec.appendChild(document.createElement("div"));
  tabs.className = "tg-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", id ? `Reading lists of group ${id}` : `Reading lists of the whole ${unit}`);
  const panel = sec.appendChild(document.createElement("div"));
  panel.className = "tg-panel";
  panel.setAttribute("role", "tabpanel");
  panel.id = `tgp-${g.id}`;
  const buttons: HTMLButtonElement[] = [];
  const pick = (i: number, focus = false) => {
    buttons.forEach((b, j) => { b.setAttribute("aria-selected", String(i === j)); b.tabIndex = i === j ? 0 : -1; });
    if (focus) buttons[i].focus();
    const [key, , help] = present[i];
    panel.setAttribute("aria-labelledby", buttons[i].id);
    panel.replaceChildren();
    panel.insertAdjacentHTML("beforeend", `<p class="tg-help">${esc(help(unit))}</p>`);
    if (key === "experts") for (const e of g.lenses.experts) panel.appendChild(expert(e, deps, unit));
    else for (const p of g.lenses[key] ?? []) panel.appendChild(deps.row(asHit(p), why(key, p, names)));
  };
  present.forEach(([, label], i) => {
    const b = tabs.appendChild(document.createElement("button"));
    b.type = "button";
    b.id = `tgt-${g.id}-${i}`;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-controls", panel.id);
    b.textContent = label;
    b.onclick = () => pick(i);
    b.onkeydown = (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        pick((i + (e.key === "ArrowRight" ? 1 : present.length - 1)) % present.length, true);
      }
    };
    buttons.push(b);
  });
  if (present.length) pick(0);
}

function why(key: string, p: TopicPaper, names: Map<number, string>): string {
  if (key === "newest") return "";
  if (key === "bridges") {
    const m = /group (\d+)/.exec(p.why ?? "");
    return m ? `${p.why!.replace(/ go to group \d+/, "")} go to group ${m[1]}: ${names.get(Number(m[1])) ?? ""}` : p.why ?? "";
  }
  const out = (p.why ?? "").replace(/^cited by/, "Cited by").replace(/^cites/, "Cites").replace(/^from/, "From");
  return p.in_map === false && key !== "start_here" ? `${out} · not among the mapped studies (no English abstract, or older)` : out;
}

function expert(e: Expert, deps: Deps, unit = "group"): HTMLElement {
  const div = document.createElement("div");
  div.className = "tg-expert";
  const paper = (p: TopicPaper | undefined) => p
    ? `<a href="?work=${encodeURIComponent(p.id)}" data-go>${esc(p.title)}</a> <span class="quiet">(${p.year ?? "n.d."})</span>` : "";
  div.innerHTML = `<span class="hit-title"><a href="https://openalex.org/${esc(e.author)}" target="_blank" rel="noopener">${esc(e.name || e.author)}</a></span>` +
    `<span class="hit-meta">h = ${e.h} inside this ${unit}` + (e.papers_in_group ? ` · author of ${e.papers_in_group} of its studies` : "") + `</span>` +
    `<span class="hit-why">Most cited here: ${paper(e.top)}` +
    (e.latest && e.latest.id !== e.top.id ? `<br>Latest here: ${paper(e.latest)}` : "") + `</span>`;
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
