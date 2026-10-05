import { MODELS, type ModelInfo } from "./models.ts";
import { rewrite, type Keys } from "./pipeline.ts";
import { open, search, PaperError, type Paper } from "./paper.ts";
import { Reader } from "./reader.ts";
import { activeJob, examples, follow, library, now, savedRewrites, status, type LibraryItem, type Status } from "./jobs.ts";
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
function show(view: "home" | "results" | "reader" | "library") {
  document.body.dataset.view = view;
  track("view", { view });
  window.scrollTo(0, 0);
}

async function runSearch(q: string) {
  q = q.trim();
  if (!q) return;
  if (/^(10\.\d{4,9}\/|pmc\d+$|https?:\/\/(dx\.)?doi\.org\/)/i.test(q)) return go(`?paper=${encodeURIComponent(q)}`);
  go(`?q=${encodeURIComponent(q)}`, false);
  show("results");
  const box = $("#results");
  box.innerHTML = `<p class="quiet">Looking for open papers…</p>`;
  try {
    const hits = await search(q);
    track("search", { n: hits.length });
    box.innerHTML = hits.length ? "" : `<p class="quiet">Nothing open to read for that. Try other words.</p>`;
    for (const h of hits) {
      const a = document.createElement("a");
      a.className = "hit";
      a.href = `?paper=${encodeURIComponent(h.doi || h.pmcid)}`;
      a.innerHTML = `<span class="hit-title">${esc(h.title)}</span><span class="hit-meta">${esc([h.journal, h.year].filter(Boolean).join(" · "))}</span>`;
      a.onclick = (e) => { e.preventDefault(); go(a.getAttribute("href")!); };
      box.appendChild(a);
    }
  } catch {
    box.innerHTML = `<p class="quiet">The search didn't answer. Try again in a moment.</p>`;
  }
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
    if (!sessionStorage.getItem("thanked")) {
      sessionStorage.setItem("thanked", "1");
      $("#thanks").hidden = false;
    }
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
  if (q.has("library")) return openLibrary();
  if (q.get("paper")) return openPaper(q.get("paper")!);
  if (q.get("q")) { ($("#search-results") as HTMLInputElement).value = q.get("q")!; return runSearch(q.get("q")!); }
  show("home");
  refreshNow();
  nowTimer = window.setInterval(() => { if (document.visibilityState === "visible") refreshNow(); }, 4000);
}

// ---------------------------------------------------------------- start
for (const f of document.querySelectorAll<HTMLFormElement>("form.search")) {
  f.onsubmit = (e) => { e.preventDefault(); runSearch((f.querySelector("input") as HTMLInputElement).value); };
}
$("#brand").onclick = (e) => { e.preventDefault(); go(""); };
for (const id of ["#open-library", "#library-link"]) $(id).onclick = (e) => { e.preventDefault(); go("?library"); };
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
    if (dlg === "#settings") loadKeyForm();
    if (dlg === "#support") track("support", {});
    ($(dlg) as HTMLDialogElement).showModal();
  };
}
// where gifts go (config.js): GitHub Sponsors, and a card option without an account when there is one
const support = window.SRL_CONFIG?.support ?? {};
($("#give-github") as HTMLAnchorElement).href = support.github ?? "https://github.com/sponsors/MaximeRivest";
if (support.card) { ($("#give-card") as HTMLAnchorElement).href = support.card; $("#give-card").hidden = false; }
for (const a of ["#give-github", "#give-card"]) $(a).addEventListener("click", () => track("give", { src: a.slice(6) }));
$("#settings form").onsubmit = () => { saveKeyForm(); if (paper) showChosen(); };
window.onpopstate = route;
const checkHome = () => status().then((s) => { home = s; $("#gpu").textContent = s?.worker_online ? "GPU online" : "GPU offline"; $("#gpu").classList.toggle("on", !!s?.worker_online); if (paper && !running) showChosen(); });
checkHome();
setInterval(checkHome, 30000);
loadExamples();
route();
