import { MODELS, type ModelInfo } from "./models.ts";
import type { Keys } from "./pipeline.ts";   // the pipeline itself loads on first use (own-key rewrites)
import { lastSearch, looksExact, open, place, prefetch, search, similar, work, PaperError, type Hit, type Paper } from "./paper.ts";
import { focus, focusGroup, focusPoint, hot, initMap, mapView, setHere, setResults, setWalk, useContext, type Line, type Spot } from "./atlas.ts";
import { fetchTopics, renderPending, renderTopics, TopicError, type TopicMap, type TopicOptions } from "./topics.ts";
import { API } from "./jobs.ts";
import { Reader } from "./reader.ts";
import { activeJob, examples, follow, library, now, openable, savedRewrites, status, supportState, type LibraryItem, type Status, type Support } from "./jobs.ts";
import { ProgressPanel } from "./progress.ts";
import { track } from "./track.ts";

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector(sel) as T;
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

// ---------------------------------------------------------------- state
let paper: Paper | null = null;
let reader: Reader | null = null;
let chosen: ModelInfo = MODELS.find((m) => m.id === "our-9b")!;
let saved: Record<string, { parts: Record<string, string>; source: string; created?: string }> = {};
let home: Status | null = null;
let running: AbortController | null = null;
let leaving = false;   // the reader opened another paper: the old rewrite stops quietly (and goes on, on our GPU)
const progress = new ProgressPanel(document.querySelector("#progress") as HTMLElement);

// ---------------------------------------------------------------- keys (stay in this browser)
const KEYS = ["anthropic", "openai"] as const;
function keys(): Keys {
  const store = localStorage.getItem("remember") === "1" ? localStorage : sessionStorage;
  return Object.fromEntries(KEYS.map((k) => [k, store.getItem(`key-${k}`) ?? ""]));
}
function loadKeyForm() {
  const remember = localStorage.getItem("remember") === "1";
  ($("#remember") as HTMLInputElement).checked = remember;
  const k = keys();
  for (const name of KEYS) ($(`#key-${name}`) as HTMLInputElement).value = k[name] ?? "";
}
function saveKeyForm() {
  const remember = ($("#remember") as HTMLInputElement).checked;
  localStorage.setItem("remember", remember ? "1" : "0");
  for (const name of KEYS) {
    const v = ($(`#key-${name}`) as HTMLInputElement).value.trim();
    (remember ? localStorage : sessionStorage).setItem(`key-${name}`, v);
    (remember ? sessionStorage : localStorage).removeItem(`key-${name}`);
  }
}

// ---------------------------------------------------------------- views
function show(view: "home" | "results" | "reader" | "library" | "work" | "explore" | "api" | "support") {
  document.body.dataset.view = view;
  document.body.classList.remove("topics-mode");
  if (view !== "home") {   // leaving home closes its search tips, so Back finds the page as it was first seen
    document.getElementById("home-tips")?.setAttribute("hidden", "");
    document.querySelector("#home .help")?.setAttribute("aria-expanded", "false");
  }
  track("view", { view });
  window.scrollTo(0, 0);
  mapView(view);
  menu(false);
  const here: Record<string, string> = { library: "#open-library", api: "#open-api", support: "#open-support" };
  for (const id of ["#open-library", "#open-api", "#open-support"])
    if (id === here[view]) $(id).setAttribute("aria-current", "page"); else $(id).removeAttribute("aria-current");
}

/** After moving to another page, keyboard and screen-reader users start at its title (not at the top bar). Not on
 *  home (the search box takes focus) nor for results (focus stays in the box being typed in). */
function focusTitle(h: Element | null) {
  if (!h || !navigated) return;
  h.setAttribute("tabindex", "-1");
  (h as HTMLElement).focus({ preventScroll: true });
}
let navigated = false;   // the first page shown keeps the browser's own focus

/** Back and Forward return to where the page was scrolled (the results list, the library...). */
let returnY: number | null = null;
function restoreScroll() {
  if (returnY === null) return;
  const y = returnY;
  returnY = null;
  requestAnimationFrame(() => window.scrollTo(0, y));
}

/** Why a search result can't be opened, in the reader's words (the same refusals as opening it). */
const CANT: Record<string, string> = {
  licence: "Can't be opened here: its licence doesn't allow a full rewrite.",
  layout: "Can't be opened here: it isn't laid out as introduction, methods and results, the shape our models learned.",
  "no full text": "Can't be opened here: Europe PMC has no full text for it.",
};
// Results appear one by one as their check comes back, never moving once shown (no misclicks):
// in Europe PMC's order, each once it and the ones above it are checked; the ones that can't be
// opened gather greyed at the bottom. A paper still unchecked after HOLD_MS stops holding the
// rest back: it shows as usual and greys out in place if refused later.
const HOLD_MS = 1500;
let searchRun = 0;

// ---------------------------------------------------------------- search options
// "" = the reader hasn't chosen: the toggle follows what they type (quotes, AND/OR/NOT → exact words).
const opts: { by: "" | "meaning" | "words"; since: string; readable: boolean } = { by: "", since: "", readable: false };
const typed = () => ($("#search-results") as HTMLInputElement).value;

function drawOpts(text = typed()) {
  const shown = opts.by || (looksExact(text) ? "words" : "meaning");
  for (const b of document.querySelectorAll<HTMLButtonElement>(".seg button")) {
    b.setAttribute("aria-checked", String(b.dataset.by === shown));
    b.tabIndex = b.dataset.by === shown ? 0 : -1;   // one stop in the tab order; arrows move inside
  }
  for (const s of document.querySelectorAll<HTMLSelectElement>(".search-opts .since")) s.value = opts.since;
  $("#readable-only").setAttribute("aria-pressed", String(opts.readable));
}
function searchUrl(q: string) {
  const p = new URLSearchParams({ q });
  if (opts.by) p.set("by", opts.by);
  if (opts.since) p.set("since", opts.since);
  if (opts.readable) p.set("readable", "1");
  return `?${p}`;
}
function rerun() {
  if (document.body.classList.contains("topics-mode")) {
    const t = typed().trim() || new URLSearchParams(location.search).get("topics") || "";
    if (t) go(topicsUrl(t, { by: opts.by, since: opts.since }));
    return;
  }
  const q = typed().trim() || new URLSearchParams(location.search).get("q") || "";
  if (q && document.body.dataset.view === "results") runSearch(q);
}

let lastHits: Hit[] = [];

async function runSearch(q: string) {
  q = q.trim();
  if (!q) return;
  if (/^(10\.\d{4,9}\/|pmc\d+$|https?:\/\/(dx\.)?doi\.org\/)/i.test(q)) return go(`?paper=${encodeURIComponent(q)}`);
  go(searchUrl(q), false);
  ($("#search-results") as HTMLInputElement).value = q;
  drawOpts(q);
  show("results");
  document.title = `${q} · ScienceMadeReadable`;
  // phones: the keyboard goes away so the results can be seen
  if (matchMedia("(pointer: coarse)").matches) (document.activeElement as HTMLElement | null)?.blur();
  const run = ++searchRun;   // a newer search makes this one's late answers moot
  const box = $("#results");
  const live = $("#results-live");
  box.setAttribute("aria-busy", "true");
  live.textContent = "Searching…";
  box.innerHTML = `<div class="hit-skeleton" aria-hidden="true">${"<div><i></i><i></i><i></i></div>".repeat(5)}</div>`;
  let hits: Hit[];
  try {
    hits = await search(q, opts);
  } catch {
    if (run === searchRun) {
      box.removeAttribute("aria-busy");
      box.innerHTML = `<p class="quiet">The search didn't answer. <a href="#" id="retry">Try again</a>.</p>`;
      live.textContent = "The search didn't answer.";
      ($("#retry") as HTMLAnchorElement).onclick = (e) => { e.preventDefault(); runSearch(q); };
    }
    return;
  }
  if (run !== searchRun) return;
  box.removeAttribute("aria-busy");
  hits = collapse(hits);
  lastHits = hits;
  setResults(hits.filter((h) => h.x != null && h.id).map(spot), lastSearch.map);
  const switchTo = (by: "meaning" | "words", label: string) => {
    const a = document.createElement("a");
    a.href = "#";   // a link with an address: reachable with the keyboard
    a.textContent = label;
    a.onclick = (e) => { e.preventDefault(); opts.by = by; runSearch(q); };
    return a;
  };
  box.innerHTML = "";
  const how = box.appendChild(document.createElement("p"));
  how.className = "quiet search-how";
  const readableN = hits.filter((h) => h.readable).length;
  live.textContent = hits.length ? `${hits.length} studies found${readableN ? `, ${readableN} readable in plain words` : ""}.` : "Nothing found.";
  if (!hits.length) {
    track("search", { n: 0, src: lastSearch.by });
    if (lastSearch.by === "keyword") how.append("No study contains all these words. Try fewer words, or ", switchTo("meaning", "search by meaning"), ".");
    else how.textContent = opts.since ? "Nothing found in these years. Try “Any year” or other words." : "Nothing found. Try other words.";
    return;
  }
  const scope = lastSearch.scope === "readable" ? "readable" : "";
  if (lastSearch.by === "keyword")
    how.append(`${(lastSearch.matches ?? hits.length).toLocaleString()} ${scope} studies contain these words; best matches first · `,
      switchTo("meaning", "Search by meaning instead"));
  else if (lastSearch.by === "semantic")
    how.append("Closest in meaning first · ", switchTo("words", "Only studies with these exact words"));
  else how.textContent = "Our search is busy, so these come from Europe PMC's word search (papers we can make readable).";
  if (lastSearch.by !== "europepmc" && !opts.readable) {
    const map = document.createElement("a");
    map.href = topicsUrl(q, { by: opts.by, since: opts.since });
    map.textContent = "See the research groups in this topic";
    map.className = "to-topics";
    map.onclick = (e) => { e.preventDefault(); go(map.getAttribute("href")!); };
    how.append(" · ", map);
  }
  // the exception is news, the rule isn't: say how many can be read in plain words only when some can
  if (lastSearch.scope === "all" && lastSearch.by !== "europepmc" && readableN) {
    const only = document.createElement("a");
    only.href = "#";
    only.textContent = `${readableN} can be read in plain words`;
    only.onclick = (e) => { e.preventDefault(); opts.readable = true; drawOpts(); runSearch(q); };
    how.append(" · ", only);
  }

  const list = box.appendChild(document.createElement("div"));
  for (const h of hits) list.appendChild(row(h));
  track("search", { n: hits.length, ok: readableN, src: lastSearch.by });
  restoreScroll();

  // the readable ones: the same check as opening them (licence, layout), so a promise isn't broken later
  const pm = hits.filter((h) => h.readable && h.pmcid).map((h) => h.pmcid);
  if (pm.length) await openable(pm, (pmcid, v) => {
    if (run !== searchRun || v === "ok" || v === "?") return;
    const h = hits.find((x) => x.pmcid === pmcid);
    if (!h) return;
    h.readable = false;
    const a = list.querySelector<HTMLAnchorElement>(`a[data-key="${CSS.escape(key(h))}"]`);
    if (a) a.replaceWith(row(h, CANT[v]));
  });
}

// ---------------------------------------------------------------- the research groups of a topic
function topicsUrl(q: string, o: TopicOptions) {
  const p = new URLSearchParams({ topics: q });
  if (o.by) p.set("by", o.by);
  if (o.since) p.set("since", o.since);
  if (o.group) p.set("group", String(o.group));
  if (o.field) p.set("scope", "field");
  return `?${p}`;
}

let topicZoom: number | null = null;     // the zoom at which the shown topic map's studies fill the explorer
let topicShown = "";                      // the topic map whose studies are on the map now

async function openTopics(q: string, group: number | null, mapOnly = false, field = false) {
  q = q.trim();
  if (!q) return;
  const o: TopicOptions = { by: opts.by, since: opts.since, group, field };
  if (mapOnly) {
    const d = await fetchTopics(q, o).catch(() => null);
    if (d && !("pending" in d)) { drawTopicSpots(d, q, o); focus(focusPoint(), topicZoom ?? 3); }
    return;
  }
  ($("#search-results") as HTMLInputElement).value = q;
  drawOpts(q);
  show("results");
  document.body.classList.add("topics-mode");
  document.title = `Research groups: ${q} · ScienceMadeReadable`;
  const run = ++searchRun;   // a newer search or map makes this one's late answer moot
  const box = $("#results");
  const live = $("#results-live");
  box.setAttribute("aria-busy", "true");
  live.textContent = "Mapping the research groups…";
  box.innerHTML = `<p class="checking">Grouping about a thousand studies by what they cite and what they say. ` +
    `The first time, this takes up to half a minute.</p>` +
    `<div class="hit-skeleton" aria-hidden="true">${"<div><i></i><i></i><i></i></div>".repeat(4)}</div>`;
  setResults([], null);
  let d;
  try {
    d = await fetchTopics(q, o);
  } catch (e) {
    if (run !== searchRun) return;
    box.removeAttribute("aria-busy");
    const err = e instanceof TopicError ? e : null;
    const say = err?.status === 429 ? `Too many maps from here in the last minute. Try again in ${err.retry ?? 60} seconds.`
      : err?.status === 503 ? "Many maps are being made right now. Try again in a minute."
      : err?.status === 404 && err.detail ? err.detail
      : "The map didn't come back.";
    box.innerHTML = `<p class="quiet">${esc(say)} <a href="#" id="retry">Try again</a> · <a href="${esc(searchUrl(q))}" id="as-list">See the studies as a list</a></p>`;
    live.textContent = say;
    ($("#retry") as HTMLAnchorElement).onclick = (ev) => { ev.preventDefault(); openTopics(q, group); };
    ($("#as-list") as HTMLAnchorElement).onclick = (ev) => { ev.preventDefault(); go(searchUrl(q)); };
    return;
  }
  if (run !== searchRun) return;
  box.removeAttribute("aria-busy");
  if ("pending" in d) {        // a whole-field map being made: show its progress, ask again in 20 s
    renderPending(box, d.pending, { row, go, focusGroup, url: (x) => topicsUrl(q, { ...o, ...x }), listUrl: searchUrl(q) });
    live.textContent = "Mapping the whole field.";
    const here = location.search;
    setTimeout(() => { if (run === searchRun && location.search === here) openTopics(q, group, false, field); }, 20000);
    return;
  }
  live.textContent = `${d.groups.length} research groups.`;
  drawTopicSpots(d, q, o);
  renderTopics(box, d, { row, go, focusGroup, url: (x) => topicsUrl(q, { ...o, ...x }), listUrl: searchUrl(q) });
  track("topics", { groups: d.groups.length, cached: d.cached ? 1 : 0, sub: group ? 1 : 0 });
  restoreScroll();
}

function drawTopicSpots(d: TopicMap, q: string, o: TopicOptions) {
  const names = new Map(d.groups.map((g) => [g.id, g.name]));
  const spots: Spot[] = [];
  d.papers.id.forEach((id, i) => {
    const x = d.papers.x[i], y = d.papers.y[i], g = d.papers.group[i];
    if (x == null || y == null) return;
    spots.push({ id, x, y, group: g, kind: "topic", title: g ? `Group ${g}: ${names.get(g)}` : "In no group" });
  });
  if (spots.length > 4000) {   // tens of thousands of dots would stall the page: an even sample keeps every group's shape
    const step = spots.length / 4000;
    const kept: Spot[] = [];
    for (let i = 0; i < 4000; i++) kept.push(spots[Math.floor(i * step)]);
    spots.length = 0;
    spots.push(...kept);
  }
  spots.sort((a, b) => (a.group ? 1 : 0) - (b.group ? 1 : 0));   // grey ones underneath
  // "open the map" flies to the middle of these studies, close enough that the groups fill the screen
  const pct = (v: number[], q: number) => v.sort((a, b) => a - b)[Math.floor(q * (v.length - 1))];
  const xs = spots.map((s) => s.x), ys = spots.map((s) => s.y);
  const middle = spots.length ? { x: pct([...xs], 0.5), y: pct([...ys], 0.5) } : null;
  const spread = spots.length ? Math.max(pct([...xs], 0.9) - pct([...xs], 0.1), pct([...ys], 0.9) - pct([...ys], 0.1)) : 1;
  topicZoom = Math.min(40, Math.max(2, 0.6 / Math.max(spread, 1e-3)));
  topicShown = topicsUrl(q, o);
  setResults(spots, middle);
}

/** The same study is often indexed several times (repository copies, preprint and article): keep one per title,
 *  the readable copy if there is one, at the best rank. */
function collapse(hits: Hit[]): Hit[] {
  const seen = new Map<string, number>(), out: Hit[] = [];
  for (const h of hits) {
    const t = h.title.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    const i = t.length > 12 ? seen.get(t) : undefined;
    if (i === undefined) { if (t.length > 12) seen.set(t, out.length); out.push(h); }
    else if (h.readable && !out[i].readable) out[i] = h;
  }
  return out;
}

const key = (h: Hit) => h.id || h.pmcid || h.doi || h.title;
const spot = (h: Hit): Spot => ({ id: key(h), x: h.x!, y: h.y!, readable: h.readable, title: h.title,
  meta: [h.journal, h.year].filter(Boolean).join(" · ") });
/** Where a study opens: the reader when we can rewrite it, else its page (details, abstract, links). */
const href = (h: Hit) => h.readable ? `?paper=${encodeURIComponent(h.doi || h.pmcid)}` : `?work=${encodeURIComponent(h.id || h.doi || h.pmcid)}`;

/** "cited once", "cited 2,281 times" */
const cited = (n: unknown) => Number(n) === 1 ? "cited once" : `cited ${Number(n).toLocaleString()} times`;

const firstAuthors = (s?: string) => {
  const names = (s ?? "").split(", ").filter(Boolean);
  return names.length > 2 ? `${names[0]}, ${names[1]} et al.` : names.join(", ");
};

function row(h: Hit, why?: string): HTMLAnchorElement {
  const a = document.createElement("a");
  a.className = "hit";
  a.dataset.key = key(h);
  a.href = href(h);
  const badges = (h.retracted ? `<span class="badge-retracted" title="The journal withdrew this study: its results should not be relied on">Retracted</span>` : "") +
    (h.readable ? `<span class="badge-ok" title="Opens rewritten in plain words">Readable</span>` : "");
  const meta = [firstAuthors(h.authors), h.journal, h.year,
    h.citations ? cited(h.citations) : ""].filter(Boolean).join(" · ");
  a.innerHTML = `<span class="hit-title">${esc(h.title)}</span>` +
    `<span class="hit-meta">${esc(meta)}${badges ? ` ${badges}` : ""}</span>` +
    (h.abstract ? `<span class="hit-snippet">${esc(h.abstract)}</span>` : "") +
    (why ? `<span class="hit-why">${esc(why)}</span>` : "");
  a.onclick = (e) => { e.preventDefault(); go(a.getAttribute("href")!); };
  a.onmouseenter = () => hot(key(h), true);
  a.onmouseleave = () => hot(key(h), false);
  // a readable paper starts opening when pointed at (a short pause: not for every row the mouse crosses)
  if (h.readable && (h.doi || h.pmcid)) {
    let t = 0;
    const start = () => prefetch(h.doi || h.pmcid);
    a.addEventListener("pointerenter", () => { t = window.setTimeout(start, 120); });
    a.addEventListener("pointerleave", () => clearTimeout(t));
    a.addEventListener("touchstart", start, { passive: true });
    a.addEventListener("focus", start);
  }
  return a;
}

// ---------------------------------------------------------------- a study we can't rewrite (or not yet)
let workRun = 0;
async function openWork(ref: string) {
  show("work");
  const run = ++workRun;
  const box = $("#work");
  box.innerHTML = `<div class="page-skeleton" aria-hidden="true"><i class="h"></i><i class="h short"></i><i></i><i></i><i class="short"></i></div>
    <p class="sr-only" role="status">Opening the study…</p>`;
  setHere(null);
  let d: any;
  try {
    d = await work(ref);
  } catch (e: any) {
    // Not in our details (no English abstract in OpenAlex: some publishers withhold them). The citation index
    // still knows it: a short page from the walk, and the walk itself.
    try {
      const w = await fetchWalk(ref), n = w.nodes[w.focus];
      d = { id: n.id, title: plain(n.title) || "(no title)", year: n.year, type: n.type, citations: n.cited_by,
            authorsLine: "", venue: n.subfield || n.topic || "", openalex: `https://openalex.org/${n.id}`,
            abstract: "", smr: false, partial: true };
    } catch {
      if (run === workRun) box.innerHTML = `<p class="quiet center">${esc(e?.message ?? "We couldn't open this study.")}</p>`;
      return;
    }
  }
  if (run !== workRun) return;
  document.title = `${d.title} · ScienceMadeReadable`;
  track("open", { doi: d.doi ?? d.id, src: "work" });
  const original = d.doi ? `https://doi.org/${d.doi}` : d.url || d.openalex;
  const readable = Boolean(d.smr);
  const scholar = `https://scholar.google.com/scholar?q=${encodeURIComponent(d.title)}`;
  const cta = d.partial
    ? `<b>We only know this study through citations.</b> OpenAlex has no abstract for it in English, so we can't
       search it by meaning or rewrite it. Its citation walk is below.<br>
       <a class="btn" href="${esc(d.openalex)}" target="_blank" rel="noopener">See it on OpenAlex ↗</a>
       <a class="btn ghost" href="${esc(scholar)}" target="_blank" rel="noopener">Find it on Google Scholar ↗</a>`
    : readable
    ? `<b>This study can be read in plain words.</b> Every finding kept, nothing dumbed down.<br>
       <a class="btn" href="?paper=${encodeURIComponent(d.doi || d.pmcid)}" data-go>Read it in plain words</a>
       <a class="btn ghost" href="${esc(original)}" target="_blank" rel="noopener">The original ↗</a>`
    : `<b>We can't rewrite this one.</b> ${d.pmc_oa ? "Its licence doesn't allow an adapted copy." : "Its full text isn't openly available to us."}
       Here is what we know about it.<br>
       <a class="btn" href="${esc(original)}" target="_blank" rel="noopener">Read the original ↗</a>
       ${d.pdf_url ? `<a class="btn ghost" href="${esc(d.pdf_url)}" target="_blank" rel="noopener">PDF ↗</a>` : ""}`;
  const retracted = d.is_retracted === true || d.is_retracted === 1 || d.is_retracted === "1";
  const where = [d.venue, d.year, d.type && d.type !== "article" ? d.type.replace("-", " ") : "",
                 d.citations ? cited(d.citations) : ""].filter(Boolean).join(" · ");
  box.innerHTML = `
    <button class="back" type="button" id="w-back">← Back</button>
    <h1>${esc(d.title)}</h1>
    <div class="w-by">${esc(d.authorsLine)}</div>
    <div class="w-where">${esc(where)}</div>
    ${retracted ? `<p class="w-retracted" role="note"><b>This study was retracted.</b> The journal withdrew it, so its
      results should not be relied on.</p>` : ""}
    <div class="w-cta">${cta}</div>
    ${d.abstract ? `<p class="w-abs-head">Abstract</p><div class="w-abs">${esc(d.abstract)}</div>` : ""}
    <div class="w-links">
      ${d.doi ? `<a href="https://doi.org/${esc(d.doi)}" target="_blank" rel="noopener">DOI ↗</a>` : ""}
      <a href="${esc(d.openalex)}" target="_blank" rel="noopener">OpenAlex ↗</a>
      ${d.pmcid ? `<a href="https://pmc.ncbi.nlm.nih.gov/articles/${esc(d.pmcid)}/" target="_blank" rel="noopener">PubMed Central ↗</a>` : ""}
      ${d.license ? `<span class="quiet">Licence: ${esc(d.license)}</span>` : ""}
    </div>
    <section class="walk" id="walk"><h2>Its citation walk</h2><p class="loading-line">Following its citations…</p></section>
    <div class="w-related" id="w-related"></div>
    <p class="you-are-here">You are here, on the map of science.<br>Click the map to look around.</p>`;
  $("#w-back").onclick = () => history.length > 1 ? history.back() : go("");
  focusTitle(box.querySelector("h1"));
  for (const a of box.querySelectorAll<HTMLAnchorElement>("a[data-go]")) a.onclick = (e) => { e.preventDefault(); go(a.getAttribute("href")!); };
  setWalk([], []);
  const pos = await place([d.id]);
  if (run === workRun && pos[d.id]) setHere(pos[d.id]);
  const ok = await showWalk(d.id, () => run === workRun);
  if (ok || run !== workRun) return;
  // no walk (no references we know): the studies closest in meaning instead
  const rel = await similar(d.id, 6);
  if (run !== workRun || !rel.length) return;
  const r = $("#w-related");
  r.innerHTML = `<p class="w-abs-head">Closest in meaning</p>`;
  for (const h of rel) r.appendChild(row(h));
}

// ---------------------------------------------------------------- the citation walk
// From one study: the work it builds on, the studies citing the same work (its conversation), and what that
// conversation relies on that this study doesn't cite (read next). Each step opens another study: a walk.
const walkCache = new Map<string, Promise<any>>();
function fetchWalk(ref: string): Promise<any> {
  let p = walkCache.get(ref);
  if (!p) {
    p = fetch(`${API}/api/data/walk?id=${encodeURIComponent(ref)}&k=30`, { signal: AbortSignal.timeout(45000) })
      .then(async (r) => { const d = await r.json(); if (!r.ok || d.error) throw new Error(d.error || d.detail || "walk failed"); return d; });
    p.catch(() => walkCache.delete(ref));
    walkCache.set(ref, p);
  }
  return p;
}

const plain = (s?: string) => (s ?? "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
const asHit = (n: any): Hit => ({ id: n.id, pmcid: "", title: plain(n.title) || "(no title)", year: n.year ? String(n.year) : undefined,
  readable: Boolean(n.readable), x: n.x, y: n.y, citations: n.cited_by });

async function showWalk(ref: string, current: () => boolean): Promise<boolean> {
  const box = document.querySelector("#walk") as HTMLElement | null;
  let d: any;
  try { d = await fetchWalk(ref); } catch (e: any) {
    if (current() && box) box.remove();
    return false;
  }
  if (!current() || !box) return true;
  const N = d.nodes, col = (c: string) => (d.columns[c] ?? []).map((id: string) => N[id]).filter(Boolean);
  const k = (d.similar ?? []).length;
  const refs = [...col("refs"), ...col("refs_out")];
  const conv = col("similar");
  const next = [...col("candidates").sort((a: any, b: any) => Number(b.answer) - Number(a.answer)), ...col("candidates_out")];
  // on the map
  const spots: Spot[] = [];
  const add = (n: any, kind: Spot["kind"]) => { if (n.x != null) spots.push({ id: n.id, x: n.x, y: n.y, kind, readable: n.readable,
    strong: Boolean(n.answer), title: plain(n.title), meta: [n.year, n.cited_by ? cited(n.cited_by) : ""].filter(Boolean).join(" · ") }); };
  refs.forEach((n: any) => add(n, "ref")); conv.forEach((n: any) => add(n, "similar")); next.forEach((n: any) => add(n, "next"));
  const f = N[d.focus], lines: Line[] = [];
  for (const l of d.links ?? []) {
    const s = N[l.s], e = N[l.t];
    if (!s || !e || s.x == null || e.x == null) continue;
    if (l.kind === "x-ref" || (l.kind === "y-cand" && e.answer)) lines.push({ x1: s.x, y1: s.y, x2: e.x, y2: e.y, kind: l.kind });
  }
  setWalk(spots, lines);
  if (f?.x != null) setHere({ x: f.x, y: f.y });

  const group = (title: string, cls: string, why: string, list: any[], note: (n: any) => string) => {
    if (!list.length) return null;
    const g = document.createElement("div");
    g.className = "walk-group";
    g.innerHTML = `<h3><span class="lg ${cls}"></span>${esc(title)}</h3><p class="why">${esc(why)}</p>`;
    const items = document.createElement("div");
    g.appendChild(items);
    const draw = (n: number) => {
      items.replaceChildren();
      for (const node of list.slice(0, n)) {
        const a = row(asHit(node));
        a.href = `?work=${encodeURIComponent(node.id)}`;
        const meta = a.querySelector(".hit-meta")!;
        meta.insertAdjacentHTML("afterbegin", esc(note(node)) + " ");
        if (node.answer) meta.insertAdjacentHTML("beforeend", ` <span class="badge-best">most relied on</span>`);
        if (node.outlier) meta.insertAdjacentHTML("beforeend", ` <span class="badge-classic" title="${esc(node.outlier)}">widely cited classic</span>`);
        items.appendChild(a);
      }
      if (list.length > n) {
        const more = document.createElement("button");
        more.type = "button"; more.className = "more"; more.textContent = `Show all ${list.length}`;
        more.onclick = () => draw(list.length);
        items.appendChild(more);
      }
    };
    draw(5);
    return g;
  };
  const of = (n: number | undefined) => n ? `cited by ${n} of the ${k} ·` : "";
  box.innerHTML = `<h2>Its citation walk</h2>
    <p class="walk-intro">${k} studies cite much of the same work as this one: its conversation. Here is what they build on,
      and what they lead to. Open any of them to keep walking.</p>
    <div class="walk-actions"><button type="button" id="walk-map">See the walk on the map</button></div>`;
  ($("#walk-map") as HTMLButtonElement).onclick = () => go(`?map&work=${encodeURIComponent(ref)}`);
  const parts = [
    group("Builds on", "k-ref", "The work this study cites that its conversation also cites, most shared first.", refs,
      (n) => of(n.ref_support).replace(/ · $/, "")),
    group("Same conversation", "k-similar", "Studies citing the same work as this one, most overlap first.", conv,
      (n) => n.shared ? `shares ${n.shared} references ·` : ""),
    group("Read next", "k-next", "What the conversation relies on that this study doesn't cite.", next,
      (n) => of(n.support).replace(/ · $/, "")),
  ];
  for (const p of parts) if (p) box.appendChild(p);
  if (f?.subfield) fieldTrend(box, f.subfield, f.year);
  return true;
}

let countsP: Promise<any> | null = null;
async function fieldTrend(box: HTMLElement, sub: string, year?: number) {
  countsP ??= fetch(`${API}/api/data/map/counts`).then((r) => r.json()).catch(() => { countsP = null; return null; });
  const c = await countsP;
  const i = c?.subfields?.indexOf(sub) ?? -1;
  if (i < 0) return;
  const from = c.years.indexOf(1950), years: number[] = c.years.slice(from), vals: number[] = c.counts.slice(from).map((r: number[]) => r[i]);
  const last = c.years.length - 2;   // the current year is partial
  const total = c.counts.reduce((s: number, r: number[]) => s + r[i], 0);
  const max = Math.max(...vals.slice(0, -1), 1);
  const pts = vals.slice(0, -1).map((v, j) => `${(j / (vals.length - 2)) * 100},${30 - (v / max) * 30}`).join(" ");
  const yi = year ? years.indexOf(year) : -1;
  const el = document.createElement("div");
  el.className = "field-trend";
  el.innerHTML = `<svg viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true"><polyline points="${pts}"/>` +
    (yi >= 0 && yi < vals.length - 1 ? `<circle cx="${(yi / (vals.length - 2)) * 100}" cy="${30 - (vals[yi] / max) * 30}" r="2.2"/>` : "") +
    `</svg><span>${esc(sub)}: about ${Math.round(total / 1000).toLocaleString()} thousand studies, ` +
    `${c.counts[last][i].toLocaleString()} in ${c.years[last]}. Since 1950 · ● this study's year.</span>`;
  box.querySelector(".walk-intro")?.after(el);
}

// ---------------------------------------------------------------- the map, whole
function openExplore(work?: string | null) {
  show("explore");
  if (work && document.querySelector("#walk") === null) {
    // a shared link to a walk on the map: fetch it, then centre on the study
    fetchWalk(work).then((d) => { showWalkOnMapOnly(d); focus(focusPoint()); }).catch(() => focus(focusPoint()));
    return;
  }
  focus(focusPoint());
}
function showWalkOnMapOnly(d: any) {
  const host = document.createElement("section");
  host.id = "walk";
  host.hidden = true;
  document.body.appendChild(host);
  return showWalk(d.focus, () => true).finally(() => host.remove());
}

async function openPaper(id: string) {
  show("reader");
  if (running) { leaving = true; running.abort(); }
  progress.hide();
  $("#paper").innerHTML = `<div class="page-skeleton reader-skeleton" aria-hidden="true"><i class="h"></i><i class="h"></i><i class="h short"></i><i></i><i></i><i></i><i class="short"></i></div>
    <p class="sr-only" role="status">Opening the paper…</p>`;
  setStatus("");
  try {
    paper = await open(id);
  } catch (e: any) {
    $("#paper").innerHTML = `<p class="quiet center">${esc(e instanceof PaperError ? e.message : "We couldn't open this paper.")}</p>`;
    const msg = String(e?.message ?? "");
    track("open", { err: /licen/i.test(msg) ? "licence" : /laid out|standard/i.test(msg) ? "layout"
      : /couldn't find|isn't a DOI/i.test(msg) ? "not found" : /openly|full text/i.test(msg) ? "no full text" : "other" });
    return;
  }
  document.title = `${paper.title} · ScienceMadeReadable`;
  track("open", { doi: paper.doi });
  setHere(null);
  setWalk([], []);
  place([paper.doi]).then((pos) => { if (paper && pos[paper.doi]) setHere(pos[paper.doi]); });
  $("#thanks").hidden = true;
  reader = new Reader($("#paper"), paper);
  reader.set({}, []);
  $("#credit").innerHTML = `Original: <a href="${esc(paper.url)}" target="_blank" rel="noopener">${esc(paper.title)}</a>, ` +
    `${esc(paper.authors)} (${esc(paper.journal)}, ${esc(paper.year)}), CC BY. The rewrite is an adaptation: the language was ` +
    `simplified by a language model and not checked by a person, so it may contain mistakes.` +
    `<p class="walk-link"><a href="?work=${encodeURIComponent(paper.doi)}" id="to-walk">Its citation walk: what it builds on, ` +
    `and what to read next →</a></p>`;
  ($("#to-walk") as HTMLAnchorElement).onclick = (e) => { e.preventDefault(); go(`?work=${encodeURIComponent(paper!.doi)}`); };
  saved = await savedRewrites(paper.doi);
  for (const m of MODELS) {
    const mine = localStorage.getItem(`rewrite:${paper.doi}:${m.id}`);
    if (mine && !saved[m.id]) saved[m.id] = JSON.parse(mine);
  }
  // a saved rewrite by another model, when the chosen one has none
  if (!saved[chosen.id]) chosen = MODELS.find((m) => saved[m.id] && m.provider === "ours") ?? chosen;
  renderModels();
  showChosen();
  attach();
  focusTitle($("#paper h1"));
}

/** Someone already asked our model for this paper: follow that rewrite instead of starting another. */
async function attach() {
  if (!paper || chosen.provider !== "ours" || saved[chosen.id] || running) return;
  const id = await activeJob(paper.doi, chosen.lm);
  if (id && !running) makeReadable(id);
}

function showChosen() {
  if (!paper || !reader) return;
  const r = saved[chosen.id];
  if (r) {
    reader.set(r.parts, Object.keys(r.parts));
    if (!running) track("read", { doi: paper.doi, model: chosen.id, src: r.source });
    setStatus(r.source === "benchmark" ? "The rewrite our judge scored." : "");
    $("#go").hidden = true;
  } else {
    reader.set({}, []);
    const offline = chosen.provider === "ours" && (!home?.worker_online || !home.models.includes(chosen.lm));
    $("#go").hidden = offline;
    $("#go").textContent = "Make it readable";
    setStatus(hint(chosen));
  }
}

function hint(m: ModelInfo): string {
  if (m.provider === "ours") {
    if (!home?.worker_online) return "New rewrites are paused: our graphics card is offline. Papers already made readable (the Library) still open.";
    if (!home.models.includes(m.lm)) return `New rewrites by ${m.name} are paused: it isn't running on our graphics card right now. Papers already made readable (the Library) still open.`;
    return "Usually under a minute.";
  }
  const k = keys()[m.provider as "anthropic" | "openai"];
  return k ? `Uses your ${m.provider === "anthropic" ? "Anthropic" : "OpenAI"} key · about ${m.seconds} s · ~$${m.usdPerPaper.toFixed(2)}`
    : `Needs your ${m.provider === "anthropic" ? "Anthropic" : "OpenAI"} key (Settings).`;
}

function renderModels() {
  const sel = $("#model") as HTMLSelectElement;
  sel.innerHTML = "";
  for (const [prov, label] of [["ours", "Our small open models"], ["anthropic", "Anthropic (your key)"], ["openai", "OpenAI (your key)"]]) {
    const g = document.createElement("optgroup");
    g.label = label;
    for (const m of MODELS.filter((x) => x.provider === prov)) {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = m.name + (saved[m.id] ? "  ✓" : "");
      o.selected = m.id === chosen.id;
      g.appendChild(o);
    }
    sel.appendChild(g);
  }
  const s = $("#model-score");
  s.textContent = `rated ${chosen.score.toFixed(1)} out of 10`;
  s.title = `On our test papers, a judge model rated ${chosen.name}'s rewrites ${chosen.score.toFixed(1)}/10 for being faithful, ` +
    `clear and pleasant to read, and preferred them over another model's ${chosen.winRate}% of the time.`;
}

function setStatus(t: string, error = false) {
  const s = $("#status");
  s.textContent = t;
  s.classList.toggle("error", error);
}

// ---------------------------------------------------------------- making a rewrite
async function makeReadable(existing?: string) {
  if (!paper || !reader) return;
  if (running) { running.abort(); return; }
  leaving = false;
  const m = chosen, p = paper, r = reader;
  running = new AbortController();
  $("#go").textContent = "Stop";
  const started = performance.now();
  const parts: Record<string, string> = {};
  const done: string[] = [];
  let timer = 0;
  let sharing = 0;
  const redraw = () => { timer = 0; r.set(parts, done); progress.update(parts, done, sharing); };
  progress.start(p.sections);
  if (!existing) track("make", { doi: p.doi, model: m.id });
  r.setPending(true);
  $("#go").hidden = false;
  const later = () => { if (!timer) timer = window.setTimeout(redraw, 120); };
  try {
    if (m.provider === "ours") {
      setStatus("");
      const res = await follow(p.doi, m.lm, (v) => {
        Object.assign(parts, v.parts);
        done.splice(0, done.length, ...v.done);
        sharing = v.running_others ?? 0;
        if (v.status === "queued") progress.queued(v.ahead, v.worker_online);
        later();
      }, running.signal, existing);
      if ("saved" in res) { saved = await savedRewrites(p.doi); showChosen(); return; }
      saved[m.id] = { parts: res.parts, source: "demo" };
    } else {
      setStatus("Loading…");
      const { rewrite } = await import("./pipeline.ts");
      await rewrite(m, p, keys(), {
        part: (k, t, isDone) => { parts[k] = t; if (isDone) done.push(k); later(); },
        status: (t) => setStatus(t),
        usage: () => {},
      }, running.signal);
      const rec = { parts, source: "browser", created: new Date().toISOString().slice(0, 10) };
      try { localStorage.setItem(`rewrite:${p.doi}:${m.id}`, JSON.stringify(rec)); } catch { /* storage full */ }
      saved[m.id] = rec;
    }
    renderModels();
    r.setPending(false);
    showChosen();
    progress.finish((performance.now() - started) / 1000);
    if (m.provider !== "ours") track("done", { doi: p.doi, model: m.id, s: Math.round((performance.now() - started) / 1000) });   // ours: the queue counts it
    setStatus("");
  } catch (e: any) {
    const msg = String(e?.message ?? e);
    const plain = /failed to fetch|networkerror|load failed|fetch request failed/i.test(msg)
      ? (m.provider === "ours" ? "we couldn't reach our GPU. It may have just gone offline; try again in a minute."
                               : "we couldn't reach the provider from this browser (network, or an extension blocking it).")
      : msg;
    r.setPending(false);
    if (leaving) return;
    if (running?.signal.aborted) { progress.hide(); setStatus(m.provider === "ours" ? "Stopped following it (it keeps going on our GPU, and will be saved)." : "Stopped."); }
    else { progress.fail(`${m.name} couldn't finish: ${plain}`); setStatus(""); track("fail", { doi: p.doi, model: m.id }); }
  } finally {
    if (timer) clearTimeout(timer);
    running = null;
    if (leaving) { leaving = false; return; }
    $("#go").textContent = saved[m.id] ? "Make it readable" : "Try again";
    if (saved[m.id]) $("#go").hidden = true;
  }
}

// ---------------------------------------------------------------- home page
async function loadExamples() {
  try {
    const list = await examples();
    const box = $("#examples");
    box.innerHTML = "";
    for (const e of list) {
      const a = document.createElement("a");
      a.className = "example";
      a.href = `?paper=${encodeURIComponent(e.doi)}`;
      a.innerHTML = `<span class="ex-plain">${esc(e.plain_title || e.title)}</span><span class="ex-orig">${esc(e.title)}</span>`;
      a.onclick = (ev) => { ev.preventDefault(); go(a.getAttribute("href")!); };
      box.appendChild(a);
    }
    $("#examples-wrap").hidden = !list.length;
  } catch { /* no examples */ }
}

// ---------------------------------------------------------------- right now, and the library
const SHORT: Record<string, string> = Object.fromEntries(MODELS.map((m) => [m.id, m.name.replace(/^(Claude|GPT-[\d.]+) /, "")]));

function paperLink(doi: string, inner: string, cls: string): HTMLAnchorElement {
  const a = document.createElement("a");
  a.className = cls;
  a.href = `?paper=${encodeURIComponent(doi)}`;
  a.innerHTML = inner;
  a.onclick = (e) => { e.preventDefault(); go(a.getAttribute("href")!); };
  return a;
}

let nowTimer = 0;
async function refreshNow() {
  const d = await now();
  if (!d) { $("#now-wrap").hidden = $("#recent-wrap").hidden = true; return; }
  const box = $("#now");
  box.innerHTML = "";
  for (const j of d.live.slice(0, 6)) {
    const title = j.plain_title || j.title || j.doi;
    const state = j.status === "queued" ? "in line" : `${Math.round(j.progress * 100)}%`;
    const a = paperLink(j.doi, `<span class="now-title${j.plain_title ? "" : " dim"}">${esc(title)}</span>` +
      `<span class="now-meta">${esc(SHORT[j.model] ?? j.model)} · ${state}</span>` +
      `<span class="now-bar"><i class="w${Math.round(j.progress * 20)}"></i></span>`, "now-item");
    box.appendChild(a);
  }
  $("#now-wrap").hidden = !d.live.length;
  const rec = $("#recent");
  rec.innerHTML = "";
  for (const r of d.recent.slice(0, 4)) {
    rec.appendChild(paperLink(r.doi, `<span class="ex-plain">${esc(r.plain_title || r.title || r.doi)}</span>` +
      `<span class="ex-orig">${esc(r.title || "")}</span>`, "example"));
  }
  $("#recent-wrap").hidden = !d.recent.length;
}

let libraryItems: LibraryItem[] = [];
function drawLibrary() {
  const words = ($("#lib-filter") as HTMLInputElement).value.toLowerCase().split(/\s+/).filter(Boolean);
  const box = $("#library");
  box.innerHTML = "";
  const shown = libraryItems.filter((x) => {
    const hay = `${x.plain_title ?? ""} ${x.title} ${x.journal ?? ""}`.toLowerCase();
    return words.every((w) => hay.includes(w));
  });
  for (const x of shown) {
    const badges = x.models.length > 3 ? `<span class="badge all">all ${x.models.length} models</span>`
      : x.models.map((m) => `<span class="badge">${esc(SHORT[m] ?? m)}</span>`).join("");
    box.appendChild(paperLink(x.doi, `<span class="ex-plain">${esc(x.plain_title || x.title)}</span>` +
      `<span class="ex-orig">${esc(x.title)}</span>` +
      `<span class="lib-meta">${esc([x.journal, x.year].filter(Boolean).join(" · "))}${badges}</span>`, "example lib-item"));
  }
  if (!shown.length) box.innerHTML = `<p class="quiet">${libraryItems.length ? "No paper matches those words." : "Nothing here yet."}</p>`;
}
async function openLibrary() {
  show("library");
  $("#library").innerHTML = `<p class="quiet">Loading…</p>`;
  libraryItems = await library();
  $("#lib-sub-count")?.remove();
  $(".lib-sub").textContent = `${libraryItems.length} papers already made readable. They open instantly.`;
  document.title = "Library · ScienceMadeReadable";
  drawLibrary();
  focusTitle($("#library-view h1"));
  restoreScroll();
}

// ---------------------------------------------------------------- routing (?q=… or ?paper=…)
function go(url: string, load = true) {
  history.replaceState({ ...(history.state ?? {}), y: window.scrollY }, "");   // where Back should return to
  history.pushState(null, "", url || location.pathname);
  navigated = true;
  if (load) route();
}
history.scrollRestoration = "manual";   // the page is rebuilt on Back: the browser's own guess lands too early
function route() {
  const q = new URLSearchParams(location.search);
  clearInterval(nowTimer);
  document.title = "ScienceMadeReadable";   // a paper or study page puts its own title in front
  if (q.has("support")) return openSupport();
  if (q.has("library")) return openLibrary();
  if (q.has("api")) { show("api"); document.title = "Search API for agents · ScienceMadeReadable"; focusTitle($("#api-view h1")); restoreScroll(); return; }
  if (q.has("map")) {
    if (q.get("topics")) {
      useContext("results");
      const p = new URLSearchParams(q); p.delete("map");
      if (topicShown !== `?${p}`) {        // a shared link: draw that topic's studies first
        show("explore");
        return openTopics(q.get("topics")!, Number(q.get("group")) || null, true, q.get("scope") === "field");
      }
      show("explore");
      return focus(focusPoint(), topicZoom ?? 3);
    }
    if (q.get("work") || q.get("paper")) useContext("walk"); else if (q.get("q")) useContext("results");
    return openExplore(q.get("work"));
  }
  if (q.get("topics")) {
    const by = q.get("by");
    opts.by = by === "words" || by === "meaning" ? by : "";
    opts.since = /^\d{4}$/.test(q.get("since") ?? "") ? q.get("since")! : "";
    opts.readable = false;
    return openTopics(q.get("topics")!, Number(q.get("group")) || null, false, q.get("scope") === "field");
  }
  if (q.get("work")) return openWork(q.get("work")!);
  if (q.get("paper")) return openPaper(q.get("paper")!);
  if (q.get("q")) {
    const by = q.get("by");
    opts.by = by === "words" || by === "meaning" ? by : "";
    opts.since = /^\d{4}$/.test(q.get("since") ?? "") ? q.get("since")! : "";
    opts.readable = q.get("readable") === "1";
    // the same search again (back from a study): keep the results and the map as they were
    if (document.body.dataset.view !== "results" && lastHits.length && ($("#search-results") as HTMLInputElement).value === q.get("q")) {
      drawOpts(q.get("q")!);
      setHere(null);
      show("results");
      document.title = `${q.get("q")} · ScienceMadeReadable`;
      restoreScroll();
      return;
    }
    return runSearch(q.get("q")!);
  }
  show("home");
  ($("form.search.big input") as HTMLInputElement).focus();
}

// ---------------------------------------------------------------- start
for (const f of document.querySelectorAll<HTMLFormElement>("form.search")) {
  const input = f.querySelector("input") as HTMLInputElement;
  f.onsubmit = (e) => { e.preventDefault(); runSearch(input.value); };
  input.addEventListener("input", () => { if (!opts.by) drawOpts(input.value); });
}
for (const b of document.querySelectorAll<HTMLButtonElement>(".seg button"))
  b.onclick = () => {
    opts.by = b.dataset.by as "meaning" | "words";
    const home = document.body.dataset.view === "home";
    drawOpts(home ? ($("form.search.big input") as HTMLInputElement).value : typed());
    if (home) { const q = ($("form.search.big input") as HTMLInputElement).value.trim(); if (q) runSearch(q); }
    else rerun();
  };
for (const s of document.querySelectorAll<HTMLSelectElement>(".search-opts .since"))
  s.onchange = () => { opts.since = s.value; drawOpts(); rerun(); };
for (const t of document.querySelectorAll<HTMLButtonElement>(".tips-toggle"))
  t.onclick = () => {
    const panel = t.parentElement!.nextElementSibling as HTMLElement;
    panel.hidden = !panel.hidden;
    t.setAttribute("aria-expanded", String(!panel.hidden));
  };
// home: the "?" in the search box opens the same tips (a copy, so the two pages stay in step)
const homeTips = $("#results-view .search-tips").cloneNode(true) as HTMLElement;
homeTips.id = "home-tips";
$("#home").append(homeTips);
const homeHelp = $("#home .help") as HTMLButtonElement;
function homeTipsOpen(open: boolean) {
  homeTips.hidden = !open;
  homeHelp.setAttribute("aria-expanded", String(open));
  if (open) fitHomeTips();
}
// the home page does not scroll: the panel ends above the bottom of the screen and scrolls inside itself
function fitHomeTips() {
  homeTips.style.maxHeight = `${Math.max(160, innerHeight - homeTips.getBoundingClientRect().top - 16)}px`;
}
addEventListener("resize", () => { if (!homeTips.hidden) fitHomeTips(); });
homeHelp.onclick = () => homeTipsOpen(homeTips.hidden);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !homeTips.hidden) { homeTipsOpen(false); homeHelp.focus(); }
});
document.addEventListener("click", (e) => {
  const t = e.target as Node;
  if (!homeTips.hidden && !homeTips.contains(t) && !homeHelp.contains(t)) homeTipsOpen(false);
});
for (const tip of document.querySelectorAll<HTMLButtonElement>(".tip"))
  tip.onclick = () => {
    homeTipsOpen(false);
    opts.by = tip.dataset.by as "meaning" | "words";
    ($("form.search.big input") as HTMLInputElement).value = tip.dataset.q!;
    runSearch(tip.dataset.q!);
  };
drawOpts("");
// phones: a hint that fits the box
if (matchMedia("(max-width: 560px)").matches)
  for (const i of document.querySelectorAll<HTMLInputElement>("form.search input")) i.placeholder = "Curious about…?";
$("#readable-only").onclick = () => { opts.readable = !opts.readable; drawOpts(); rerun(); };
initMap({
  api: API,
  // keep what was on screen in the address (?q=, ?work=, ?paper=), so Back and a shared link return to it
  explore: () => go(`?map${location.search.length > 1 ? "&" + location.search.slice(1) : ""}`),
  open: (s) => { const h = lastHits.find((x) => key(x) === s.id); go(h ? href(h) : `?work=${encodeURIComponent(s.id)}`); },
});
$("#brand").onclick = (e) => { e.preventDefault(); go(""); };
for (const id of ["#open-library"]) $(id).onclick = (e) => { e.preventDefault(); go("?library"); };
for (const id of ["#open-api", "#open-api-2", "#open-api-3"])
  $(id).onclick = (e) => { e.preventDefault(); ($("#about") as HTMLDialogElement).close(); go("?api"); };
// the API page: a Copy button on every code box (counted, to know whether anyone sets up an agent)
for (const pre of document.querySelectorAll<HTMLElement>("#api-view pre")) {
  const box = document.createElement("div");
  box.className = "codebox";
  pre.replaceWith(box);
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy";
  copy.textContent = "Copy";
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(pre.textContent!.trim()); }
    catch {   // no clipboard access: select the text, so Ctrl+C / long-press copies it
      getSelection()?.selectAllChildren(pre);
      return;
    }
    copy.textContent = "Copied";
    copy.classList.add("done");
    setTimeout(() => { copy.textContent = "Copy"; copy.classList.remove("done"); }, 1600);
    track("copy", { src: pre.dataset.copy });
  };
  pre.tabIndex = 0;   // a long line scrolls sideways: reachable with the keyboard too
  box.append(pre, copy);
}

// ---------------------------------------------------------------- the top bar on phones: a menu
function menu(open: boolean) {
  $("nav.top").classList.toggle("menu-open", open);
  $("#menu-btn").setAttribute("aria-expanded", String(open));
}
$("#menu-btn").onclick = () => {
  const open = !$("nav.top").classList.contains("menu-open");
  menu(open);
  if (open) $("#nav-links a")?.focus();
};
document.addEventListener("click", (e) => {
  if ($("nav.top").classList.contains("menu-open") && !$("nav.top").contains(e.target as Node)) menu(false);
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && $("nav.top").classList.contains("menu-open")) { menu(false); $("#menu-btn").focus(); }
});
// "Skip to content": to the page's title, past the top bar
$("#skip").onclick = (e) => {
  e.preventDefault();
  const main = [...document.querySelectorAll<HTMLElement>("main")].find((m) => getComputedStyle(m).display !== "none");
  const target = main?.querySelector<HTMLElement>("h1, input") ?? main;
  if (!target) return;
  if (target.tagName !== "INPUT") target.setAttribute("tabindex", "-1");
  target.focus();
};
// "By meaning / Exact words": a radio group, so the arrow keys move between the two
for (const seg of document.querySelectorAll<HTMLElement>(".seg"))
  seg.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    const bs = [...seg.querySelectorAll<HTMLButtonElement>("button")];
    const i = bs.indexOf(document.activeElement as HTMLButtonElement);
    const next = bs[(i + (e.key === "ArrowLeft" || e.key === "ArrowUp" ? bs.length - 1 : 1)) % bs.length];
    next.focus();
    next.click();
  });
$("#lib-filter").oninput = drawLibrary;
($("#model") as HTMLSelectElement).onchange = (e) => {
  chosen = MODELS.find((m) => m.id === (e.target as HTMLSelectElement).value)!;
  renderModels();
  showChosen();
  attach();
}
;
$("#go").onclick = () => makeReadable();
($("#side") as HTMLInputElement).onchange = (e) => document.body.classList.toggle("solo", !(e.target as HTMLInputElement).checked);
for (const [btn, dlg] of [["#open-settings", "#settings"], ["#open-about", "#about"], ["#open-about-2", "#about"]]) {
  $(btn).onclick = (e) => {
    e.preventDefault();
    menu(false);
    if (dlg === "#settings") loadKeyForm();
    ($(dlg) as HTMLDialogElement).showModal();
  };
}

// ---------------------------------------------------------------- support
// Support goes to Stripe (card, Apple Pay, Google Pay; no account; one link per amount, made by
// tools/stripe_setup.py) or GitHub Sponsors (for developers; no fee). The links come from the queue
// (support.json) or config.js, when the queue can't be reached.
// Voluntary support only (2026-10-12): no funding meter, no cost-per-hour pitch, no paid "sponsor a day".
// Those read as crowdfunding to Stripe, which restricts it.
const SPONSORS = (window.SRL_CONFIG?.support?.github ?? "https://github.com/sponsors/MaximeRivest").replace(/\/$/, "");
const githubLink = (amount: number, frequency: "one-time" | "recurring") =>
  `${SPONSORS}/sponsorships?preview=false&frequency=${frequency}` + (amount ? `&amount=${amount}` : "");
let sup: Support | null = null;
let amount = 10;               // 0: the supporter chooses
const usd = (n: number) => `$${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
const stripeLinks = (): Record<string, string> => ({ ...(window.SRL_CONFIG?.support?.stripe ?? {}), ...(sup?.stripe ?? {}) });

function drawSupport() {
  const st = stripeLinks();
  const label = amount ? `Support with ${usd(amount)}` : "Support with any amount";
  const card = st[amount ? String(amount) : "custom"];
  const gh = $("#give-github") as HTMLAnchorElement;
  gh.href = githubLink(amount, "one-time");
  if (card) {
    Object.assign($("#give-card") as HTMLAnchorElement, { href: card, textContent: label, hidden: false });
    if (gh.parentElement!.id !== "gh-alt") { $("#gh-alt").append(" · ", gh); }
    gh.className = "ghost-link";
    gh.textContent = "Developers: GitHub Sponsors";
    ($("#give-monthly") as HTMLAnchorElement).href = st.monthly ?? githubLink(3, "recurring");
    $("#pay-note").textContent = "Card, Apple Pay or Google Pay, through Stripe: no account needed. " +
      "On GitHub Sponsors (needs a GitHub account), GitHub keeps no fee.";
  } else {
    $("#give-card").hidden = true;
    gh.className = "primary-btn";
    gh.textContent = `${label} on GitHub`;
    ($("#give-monthly") as HTMLAnchorElement).href = githubLink(3, "recurring");
    $("#pay-note").textContent = "GitHub Sponsors needs a GitHub account (free), and GitHub keeps no fee.";
  }
  for (const b of document.querySelectorAll<HTMLButtonElement>(".amounts button")) b.classList.toggle("on", +b.dataset.amount! === amount);
}

// The support page (?support). `supportFrom` says which link led there; a shared or typed address is "link".
let supportFrom = "link";
for (const id of ["#open-support", "#open-support-2", "#open-support-3"])
  $(id).onclick = (e) => { e.preventDefault(); supportFrom = id.slice(1); go("?support"); };
$("#sup-read").onclick = (e) => { e.preventDefault(); go(""); };
$("#sup-api").onclick = (e) => { e.preventDefault(); go("?api"); };

async function openSupport() {
  track("support", { src: supportFrom });
  supportFrom = "link";
  show("support");
  document.title = "Support · ScienceMadeReadable";
  drawSupport();
  focusTitle($("#support-view h1"));
  restoreScroll();
  sup = (await supportState()) ?? sup;
  drawSupport();
}

for (const b of document.querySelectorAll<HTMLButtonElement>(".amounts button"))
  b.onclick = () => { amount = +b.dataset.amount!; drawSupport(); };
const clicked: Record<string, () => number> = { "#give-card": () => amount, "#give-github": () => amount, "#give-monthly": () => 3 };
for (const [a, n] of Object.entries(clicked))
  $(a).addEventListener("click", () => track("give", { src: a.slice(6), amount: n() }));
supportState().then((s) => { sup = s; });
if (new URLSearchParams(location.search).has("thanks")) {   // back from Stripe
  track("supported", {});
  history.replaceState(null, "", location.pathname);
  ($("#thanked") as HTMLDialogElement).showModal();
}

// A quiet note at the end of the paper, once per visit, when a reader has read a rewrite to the end.
new IntersectionObserver((entries) => {
  if (!entries.some((e) => e.isIntersecting) || sessionStorage.getItem("thanked")) return;
  if (!paper || !saved[chosen.id] || document.body.dataset.view !== "reader") return;
  sessionStorage.setItem("thanked", "1");
  $("#thanks").hidden = false;
  track("thanks_shown", {});
}, { rootMargin: "0px 0px -20% 0px" }).observe($("#credit"));
$("#settings form").onsubmit = () => { saveKeyForm(); if (paper) showChosen(); };
window.onpopstate = (e) => { returnY = e.state?.y ?? null; navigated = true; route(); };
// "Rewriting on" only when our main model is really there to take a paper (the worker can be up with no model)
const checkHome = () => status().then((s) => {
  home = s;
  const on = !!s?.worker_online && (s.models ?? []).includes("our-9b");
  $("#gpu").textContent = on ? "Rewriting on" : "Rewriting paused";
  $("#gpu").classList.toggle("on", on);
  if (paper && !running) showChosen();
});
checkHome();
setInterval(checkHome, 30000);
route();
