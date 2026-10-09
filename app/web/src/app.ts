import { MODELS, type ModelInfo } from "./models.ts";
import { rewrite, type Keys } from "./pipeline.ts";
import { lastSearch, looksExact, open, place, search, similar, work, PaperError, type Hit, type Paper } from "./paper.ts";
import { currentHere, currentQuestion, focus, hot, initMap, mapView, setHere, setResults, type Spot } from "./atlas.ts";
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
function show(view: "home" | "results" | "reader" | "library" | "work" | "explore") {
  document.body.dataset.view = view;
  track("view", { view });
  window.scrollTo(0, 0);
  mapView(view);
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
  for (const b of document.querySelectorAll<HTMLButtonElement>(".seg button"))
    b.setAttribute("aria-checked", String(b.dataset.by === shown));
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
  const run = ++searchRun;   // a newer search makes this one's late answers moot
  const box = $("#results");
  box.innerHTML = `<p class="quiet">Looking across science…</p>`;
  let hits: Hit[];
  try {
    hits = await search(q, opts);
  } catch {
    if (run === searchRun) box.innerHTML = `<p class="quiet">The search didn't answer. Try again in a moment.</p>`;
    return;
  }
  if (run !== searchRun) return;
  hits = collapse(hits);
  lastHits = hits;
  setResults(hits.filter((h) => h.x != null && h.id).map(spot), lastSearch.map);
  const switchTo = (by: "meaning" | "words", label: string) => {
    const a = document.createElement("a");
    a.textContent = label;
    a.onclick = () => { opts.by = by; runSearch(q); };
    return a;
  };
  box.innerHTML = "";
  const how = box.appendChild(document.createElement("p"));
  how.className = "quiet search-how";
  const readableN = hits.filter((h) => h.readable).length;
  if (!hits.length) {
    track("search", { n: 0, src: lastSearch.by });
    if (lastSearch.by === "keyword") how.append("No study contains all these words. Try fewer words, or ", switchTo("meaning", "search by meaning"), ".");
    else how.textContent = opts.since ? "Nothing found in these years. Try “Any year” or other words." : "Nothing found. Try other words.";
    return;
  }
  const scope = lastSearch.scope === "readable" ? "readable" : "";
  if (lastSearch.by === "keyword")
    how.append(`${(lastSearch.matches ?? hits.length).toLocaleString()} ${scope} studies contain these words; best matches first. `,
      switchTo("meaning", "Search by meaning instead"));
  else if (lastSearch.by === "semantic")
    how.append("Closest in meaning first. ", switchTo("words", "Only studies with these exact words"));
  else how.textContent = "Our search is busy, so these come from Europe PMC's word search (papers we can make readable).";
  if (lastSearch.scope === "all" && lastSearch.by !== "europepmc")
    how.append(` · ${readableN} of ${hits.length} can be read in plain words`);

  const list = box.appendChild(document.createElement("div"));
  for (const h of hits) list.appendChild(row(h));
  track("search", { n: hits.length, ok: readableN, src: lastSearch.by });

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

function row(h: Hit, why?: string): HTMLAnchorElement {
  const a = document.createElement("a");
  a.className = "hit";
  a.dataset.key = key(h);
  a.href = href(h);
  const badge = h.readable ? `<span class="badge-ok" title="We can rewrite this one in plain words">Readable</span>`
    : `<span class="badge-orig"${why ? ` title="${esc(why)}"` : ""}>Original only</span>`;
  a.innerHTML = `<span class="hit-title">${esc(h.title)}</span>` +
    `<span class="hit-meta">${esc([h.journal, h.year].filter(Boolean).join(" · "))} ${badge}${h.x != null ? ' <span class="pin" title="on the map">●</span>' : ""}</span>`;
  a.onclick = (e) => { e.preventDefault(); go(a.getAttribute("href")!); };
  a.onmouseenter = () => hot(key(h), true);
  a.onmouseleave = () => hot(key(h), false);
  return a;
}

// ---------------------------------------------------------------- a study we can't rewrite (or not yet)
let workRun = 0;
async function openWork(ref: string) {
  show("work");
  const run = ++workRun;
  const box = $("#work");
  box.innerHTML = `<p class="quiet center">Opening the study…</p>`;
  setHere(null);
  let d: any;
  try {
    d = await work(ref);
  } catch (e: any) {
    if (run === workRun) box.innerHTML = `<p class="quiet center">${esc(e?.message ?? "We couldn't open this study.")}</p>`;
    return;
  }
  if (run !== workRun) return;
  document.title = `${d.title} · made readable`;
  track("open", { doi: d.doi ?? d.id, src: "work" });
  const original = d.doi ? `https://doi.org/${d.doi}` : d.url || d.openalex;
  const readable = Boolean(d.smr);
  const cta = readable
    ? `<b>This study can be read in plain words.</b> Every finding kept, nothing dumbed down.<br>
       <a class="btn" href="?paper=${encodeURIComponent(d.doi || d.pmcid)}" data-go>Read it in plain words</a>
       <a class="btn ghost" href="${esc(original)}" target="_blank" rel="noopener">The original ↗</a>`
    : `<b>We can't rewrite this one.</b> ${d.pmc_oa ? "Its licence doesn't allow an adapted copy." : "Its full text isn't openly available to us."}
       Here is what we know about it.<br>
       <a class="btn" href="${esc(original)}" target="_blank" rel="noopener">Read the original ↗</a>
       ${d.pdf_url ? `<a class="btn ghost" href="${esc(d.pdf_url)}" target="_blank" rel="noopener">PDF ↗</a>` : ""}`;
  const where = [d.venue, d.year, d.type && d.type !== "article" ? d.type.replace("-", " ") : "",
                 d.citations ? `cited ${Number(d.citations).toLocaleString()} times` : ""].filter(Boolean).join(" · ");
  box.innerHTML = `
    <button class="back" type="button" id="w-back">← Back</button>
    <h1>${esc(d.title)}</h1>
    <div class="w-by">${esc(d.authorsLine)}</div>
    <div class="w-where">${esc(where)}</div>
    <div class="w-cta">${cta}</div>
    ${d.abstract ? `<p class="w-abs-head">Abstract</p><div class="w-abs">${esc(d.abstract)}</div>` : ""}
    <div class="w-links">
      ${d.doi ? `<a href="https://doi.org/${esc(d.doi)}" target="_blank" rel="noopener">DOI ↗</a>` : ""}
      <a href="${esc(d.openalex)}" target="_blank" rel="noopener">OpenAlex ↗</a>
      ${d.pmcid ? `<a href="https://pmc.ncbi.nlm.nih.gov/articles/${esc(d.pmcid)}/" target="_blank" rel="noopener">PubMed Central ↗</a>` : ""}
      ${d.license ? `<span class="quiet">Licence: ${esc(d.license)}</span>` : ""}
    </div>
    <div class="w-related" id="w-related"></div>
    <p class="you-are-here">You are here, on the map of science.<br>Click the map to look around.</p>`;
  $("#w-back").onclick = () => history.length > 1 ? history.back() : go("");
  for (const a of box.querySelectorAll<HTMLAnchorElement>("a[data-go]")) a.onclick = (e) => { e.preventDefault(); go(a.getAttribute("href")!); };
  const pos = await place([d.id]);
  if (run === workRun && pos[d.id]) setHere(pos[d.id]);
  const rel = await similar(d.id, 6);
  if (run !== workRun || !rel.length) return;
  const r = $("#w-related");
  r.innerHTML = `<p class="w-abs-head">Closest in meaning</p>`;
  for (const h of rel) r.appendChild(row(h));
}

// ---------------------------------------------------------------- the map, whole
function openExplore() {
  show("explore");
  focus(currentQuestion() ?? currentHere());
}

async function openPaper(id: string) {
  show("reader");
  if (running) { leaving = true; running.abort(); }
  progress.hide();
  $("#paper").innerHTML = `<p class="quiet center">Opening the paper…</p>`;
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
  document.title = `${paper.title} · made readable`;
  track("open", { doi: paper.doi });
  setHere(null);
  place([paper.doi]).then((pos) => { if (paper && pos[paper.doi]) setHere(pos[paper.doi]); });
  $("#thanks").hidden = true;
  reader = new Reader($("#paper"), paper);
  reader.set({}, []);
  $("#credit").innerHTML = `Original: <a href="${esc(paper.url)}" target="_blank" rel="noopener">${esc(paper.title)}</a>, ` +
    `${esc(paper.authors)} (${esc(paper.journal)}, ${esc(paper.year)}), CC BY. The rewrite is an adaptation: the language was ` +
    `simplified by a language model and not checked by a person, so it may contain mistakes.`;
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
    if (!home?.worker_online) return "Our GPU is offline right now, so it can't take new papers. Papers in the Library and the examples still work.";
    if (!home.models.includes(m.lm)) return `${m.name} isn't running on our GPU right now.`;
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
  $("#model-score").textContent = `score ${chosen.score.toFixed(1)}/10 · wins ${chosen.winRate}%`;
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
  drawLibrary();
}

// ---------------------------------------------------------------- routing (?q=… or ?paper=…)
function go(url: string, load = true) {
  history.pushState(null, "", url || location.pathname);
  if (load) route();
}
function route() {
  const q = new URLSearchParams(location.search);
  clearInterval(nowTimer);
  if (q.has("support")) setTimeout(() => openSupport("link"), 300);
  if (q.has("library")) return openLibrary();
  if (q.has("map")) return openExplore();
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
      return show("results");
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
for (const tip of document.querySelectorAll<HTMLButtonElement>(".tip"))
  tip.onclick = () => {
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
  explore: () => go(`?map${location.search.includes("q=") ? "&" + location.search.slice(1).replace(/(^|&)map(&|$)/, "$1") : ""}`),
  open: (s) => { const h = lastHits.find((x) => key(x) === s.id); go(h ? href(h) : `?work=${encodeURIComponent(s.id)}`); },
});
$("#brand").onclick = (e) => { e.preventDefault(); go(""); };
for (const id of ["#open-library"]) $(id).onclick = (e) => { e.preventDefault(); go("?library"); };
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
for (const [btn, dlg] of [["#open-settings", "#settings"], ["#open-about", "#about"], ["#open-about-2", "#about"],
                          ["#open-support", "#support"], ["#open-support-2", "#support"], ["#open-support-3", "#support"]]) {
  $(btn).onclick = (e) => {
    e.preventDefault();
    if (dlg === "#support") return openSupport(btn.slice(1));
    if (dlg === "#settings") loadKeyForm();
    ($(dlg) as HTMLDialogElement).showModal();
  };
}

// ---------------------------------------------------------------- support
// Support goes to Stripe (card, Apple Pay, Google Pay; no account; one link per amount, made by
// tools/stripe_setup.py) or GitHub Sponsors (for developers; no fee). The links come from the queue
// (support.json) or config.js, when the queue can't be reached. The meter: what the GPU costs a day,
// and the support of the last 24 hours.
const SPONSORS = (window.SRL_CONFIG?.support?.github ?? "https://github.com/sponsors/MaximeRivest").replace(/\/$/, "");
const githubLink = (amount: number, frequency: "one-time" | "recurring") =>
  `${SPONSORS}/sponsorships?preview=false&frequency=${frequency}` + (amount ? `&amount=${amount}` : "");
const PAPERS_PER_HOUR = 800;   // the 9B on one H100, many papers at once (training/speed/throughput-h100-9b.json)
let sup: Support | null = null;
let amount = 10;               // 0: the supporter chooses
const usd = (n: number) => `$${n.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
const stripeLinks = (): Record<string, string> => ({ ...(window.SRL_CONFIG?.support?.stripe ?? {}), ...(sup?.stripe ?? {}) });

function drawSupport() {
  const cost = sup?.daily_cost ?? 108;
  const st = stripeLinks();
  const hours = (amount || 10) / (cost / 24);
  $("#impact").textContent = amount === 0 ? "Any amount helps: $4.50 pays for an hour of a rented H100."
    : `${usd(amount)} pays for about ${hours < 1.5 ? `${Math.round(hours * 60)} minutes` : `${Math.round(hours)} hours`} of a rented H100: ` +
      `time to make up to ${(Math.round(hours * PAPERS_PER_HOUR / 100) * 100).toLocaleString("en-US")} papers readable.`;
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
  ($("#give-day") as HTMLAnchorElement).href = st.day ?? githubLink(110, "one-time");
  $("#day-note").innerHTML = st.day ? ": you'll be asked for the name to show and the day."
    : `, then tell me the name and the day on X, <a href="https://x.com/MaximeRivest" target="_blank" rel="noopener">@MaximeRivest</a>.`;
  for (const b of document.querySelectorAll<HTMLButtonElement>(".amounts button")) b.classList.toggle("on", +b.dataset.amount! === amount);
  const meter = sup?.daily_cost && sup.day_dollars !== undefined;
  $("#meter").hidden = !meter;
  if (meter) {
    const got = sup!.day_dollars!, n = sup!.day_count!;
    $("#meter-text").textContent = got > 0 ? `${usd(got)} from ${n} ${n === 1 ? "supporter" : "supporters"} in the last 24 hours`
      : "No support yet in the last 24 hours";
    $("#meter-cost").textContent = `a rented-GPU day costs ${usd(cost)}`;
    ($("#meter-fill") as HTMLElement).style.width = `${Math.min(100, Math.max(got > 0 ? 3 : 0, (got / cost) * 100))}%`;
    const notes: string[] = [];
    if (sup!.monthly_supporters) notes.push(`${sup!.monthly_supporters} ${sup!.monthly_supporters === 1 ? "person supports" : "people support"} it every month.`);
    if (sup!.recent?.length) notes.push(`Thank you, ${sup!.recent.slice(0, 5).map((l) => "@" + l).join(", ")}.`);
    if (got === 0) notes.push("You could be the first today.");
    $("#meter-note").textContent = notes.join(" ");
  }
  drawSponsorOfTheDay();
}

function drawSponsorOfTheDay() {
  const d = sup?.sponsor_of_the_day;   // the queue only sends today's, once approved (tools/sponsor_day.py)
  const html = d?.name ? `Today's rewrites are sponsored by ${d.url ? `<a href="${esc(d.url)}" target="_blank" rel="noopener sponsored">${esc(d.name)}</a>`
    : `<b>${esc(d.name)}</b>`}. Thank you!` : "";
  for (const id of ["#sotd", "#sotd-bar"]) { $(id).innerHTML = html; $(id).hidden = !html; }
}

async function openSupport(src: string) {
  track("support", { src });
  drawSupport();
  ($("#support") as HTMLDialogElement).showModal();
  sup = (await supportState()) ?? sup;
  drawSupport();
}

for (const b of document.querySelectorAll<HTMLButtonElement>(".amounts button"))
  b.onclick = () => { amount = +b.dataset.amount!; drawSupport(); };
const clicked: Record<string, () => number> = { "#give-card": () => amount, "#give-github": () => amount, "#give-monthly": () => 3, "#give-day": () => 110 };
for (const [a, n] of Object.entries(clicked))
  $(a).addEventListener("click", () => track("give", { src: a.slice(6), amount: n() }));
supportState().then((s) => { sup = s; drawSponsorOfTheDay(); });
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
  if (sup?.daily_cost) $("#thanks-text").textContent =
    `This rewrite was free for you. Busy days need a rented GPU, about ${usd(sup.daily_cost)} a day, and readers pay for them.`;
  $("#thanks").hidden = false;
  track("thanks_shown", {});
}, { rootMargin: "0px 0px -20% 0px" }).observe($("#credit"));
$("#settings form").onsubmit = () => { saveKeyForm(); if (paper) showChosen(); };
window.onpopstate = route;
const checkHome = () => status().then((s) => { home = s; $("#gpu").textContent = s?.worker_online ? "GPU online" : "GPU offline"; $("#gpu").classList.toggle("on", !!s?.worker_online); if (paper && !running) showChosen(); });
checkHome();
setInterval(checkHome, 30000);
route();
