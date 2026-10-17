// The live admin dashboard: reads /admin/data (admin.py) every 10 s while the tab is visible and draws it.
// No library: the charts are small SVGs. The page's security policy forbids inline styles, so colours are classes.
"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (n) => (n ?? 0).toLocaleString();
const params = new URLSearchParams(location.search);
let range = ["today", "yesterday", "7d", "30d"].includes(params.get("range")) ? params.get("range") : "today";
let feedFilter = "all";
let D = null;
const opened = new Set();        // visits unfolded by the reader: kept open across refreshes
let loading = false;

// ------------------------------------------------------------------ helpers
function hue(v) {
  let h = 0;
  for (const c of v || "") h = (h * 31 + c.charCodeAt(0)) % 360;
  return h;
}
const HUES = 12;                 // visitor colours: 12 classes .v0 … .v11 (no inline styles allowed)
const who = (v) => v ? `<span class="who v${hue(v) % HUES}" title="visitor ${esc(v)}">${esc(v.slice(0, 4))}</span>` : `<span class="who">—</span>`;

function clock(t, withDay) {
  const d = new Date(t * 1000);
  const hm = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return withDay ? `${d.toLocaleDateString([], { weekday: "short", day: "numeric" })} ${hm}` : hm;
}
function ago(t) {
  const s = Math.max(0, Math.round(D.now - t));
  return s < 60 ? `${s} s ago` : s < 3600 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`;
}
function length(s) {
  s = Math.round(s);
  return s < 60 ? `${s} s` : s < 3600 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`;
}
const link = (text, href) => href ? `<a href="${esc(href)}" target="_blank" rel="noopener">${esc(text)}</a>` : esc(text);

/** A ranked list as a table with a bar for each count. rows: [text, count, link]. */
function table(el, rows, empty = "Nothing yet.") {
  if (!rows || !rows.length) { $(el).innerHTML = `<p class="none">${empty}</p>`; return; }
  const max = Math.max(...rows.map((r) => r[1]), 1);
  $(el).innerHTML = `<table>${rows.map(([t, n, href]) =>
    `<tr><td class="txt">${link(t, href)}</td><td class="bar"><svg viewBox="0 0 100 8" preserveAspectRatio="none" aria-hidden="true">` +
    `<rect width="${(100 * n / max).toFixed(1)}" height="8" rx="2"/></svg></td><td class="n">${num(n)}</td></tr>`).join("")}</table>`;
}

// ------------------------------------------------------------------ charts
/** Bars (stacked) or lines over the period's hours or days. series: [{name, data, cls}]. */
function chart(el, labels, series, stacked, totals = true) {
  const W = 440, H = 170, L = 30, B = 20, T = 8, R = 4;
  const n = labels.length, w = (W - L - R) / n;
  const tops = labels.map((_, i) => stacked ? series.reduce((a, s) => a + s.data[i], 0) : Math.max(...series.map((s) => s.data[i])));
  const max = niceMax(Math.max(...tops, 1));
  const y = (v) => T + (H - T - B) * (1 - v / max);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="chart">`;
  for (const f of [0, 0.5, 1]) {
    svg += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(max * f)}" y2="${y(max * f)}"/>` +
      `<text class="ax" x="${L - 5}" y="${y(max * f) + 4}" text-anchor="end">${num(Math.round(max * f))}</text>`;
  }
  const every = Math.ceil(n / 8);
  labels.forEach((lab, i) => {
    if (i % every === 0) svg += `<text class="ax" x="${L + w * (i + 0.5)}" y="${H - 6}" text-anchor="middle">${esc(lab)}</text>`;
  });
  if (stacked) {
    labels.forEach((lab, i) => {
      let base = 0;
      for (const s of series) {
        const v = s.data[i];
        if (v) svg += `<rect class="${s.cls}" x="${L + w * i + w * 0.12}" width="${w * 0.76}" y="${y(base + v)}" height="${y(base) - y(base + v)}"/>`;
        base += v;
      }
    });
  } else {
    for (const s of series) {
      const pts = s.data.map((v, i) => `${(L + w * (i + 0.5)).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      svg += `<polyline class="line ${s.cls}" points="${pts}"/>`;
    }
  }
  labels.forEach((lab, i) => {     // hover: one invisible column per hour or day, saying every value
    const tip = `${lab}\n` + series.map((s) => `${s.name}: ${num(s.data[i])}`).join("\n");
    svg += `<rect class="hit" x="${L + w * i}" y="${T}" width="${w}" height="${H - T - B}"><title>${esc(tip)}</title></rect>`;
  });
  svg += "</svg>";
  const legend = series.map((s) => `<span><i class="key ${s.cls}"></i>${esc(s.name)}` +
    (totals ? ` <b>${num(s.data.reduce((a, b) => a + b, 0))}</b>` : "") + "</span>").join("");
  $(el).innerHTML = svg + `<p class="legend">${legend}</p>`;
}
function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v)), m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

// ------------------------------------------------------------------ drawing
function tile(label, now, prev, sub = "", lower = false) {
  let delta = "";
  if (!prev && now) delta = `<em class="up" title="none ${esc(D.previous)}">new</em>`;
  else if (prev) {
    const p = Math.round(100 * (now - prev) / prev);
    const good = lower ? p < 0 : p > 0;
    delta = p === 0 ? `<em title="${num(prev)} ${esc(D.previous)}">same</em>`
      : `<em class="${good ? "up" : "down"}" title="${num(prev)} ${esc(D.previous)}">${p > 0 ? "▲" : "▼"} ${Math.abs(p)}%</em>`;
  }
  return `<div class="tile"><b>${num(now)}</b>${delta}<span>${esc(label)}</span>${sub ? `<small>${sub}</small>` : ""}</div>`;
}

function render() {
  const c = D.now_counts, p = D.prev_counts;
  document.title = `${c.visitors} visitors · Admin`;
  for (const b of document.querySelectorAll("#ranges button")) b.setAttribute("aria-pressed", String(b.dataset.range === D.range));
  $("#updated").textContent = `Updated ${clock(D.now)} · compared with ${D.previous}`;
  const h = D.health;
  const pill = (ok, yes, no) => `<span class="pill ${ok ? "ok" : "bad"}">${ok ? yes : no}</span>`;
  $("#health").innerHTML = pill(h.search, "search up", "search DOWN") + pill(h.worker, `GPU worker on (${(h.models || []).join(", ")})`, "GPU worker off") +
    `<span class="pill">${h.running} writing · ${h.queued} waiting</span>`;
  const early = D.start < D.first_detail;
  $("#early").hidden = !early;
  if (early) $("#early").textContent = "Searches, topic maps, author pages, visit paths and what agents asked are recorded from Oct 17, 2026, about 3:40 pm. Before that this period shows visits, papers and rewrites, and agents' tool calls (from the server log, without their questions).";

  // right now
  const L = D.live;
  $("#active-n").textContent = L.active.length;
  $("#active").innerHTML = L.active.length ? L.active.map((a) =>
    `<li><button class="row" data-visit="${esc(a.key)}">${who(a.v)}<span class="what">${esc(a.page)}<small>${esc(a.doing)}</small></span>` +
    `<span class="when">${a.pages} page${a.pages === 1 ? "" : "s"} · ${length(a.last - a.since)}<small>${ago(a.last)}</small></span></button></li>`).join("")
    : `<li class="none">Nobody right now.</li>`;
  $("#agents-n").textContent = L.agents_10min;
  $("#agent-apps").textContent = L.agent_apps_10min.map(([a, n]) => `${a} ${n}`).join(" · ") || "None.";
  drawFeed();

  // tiles
  const money = D.money;
  const acc = D.accounts;
  $("#tiles").innerHTML = [
    tile("visitors", c.visitors, p.visitors, `${c.from_x} from X`),
    tile("searches", c.searches, p.searches, c.empty ? `${c.empty} found nothing` : ""),
    tile("topic maps", c.topics, p.topics),
    tile("author & institution pages", c.people, p.people),
    tile("papers opened", c.opened, p.opened),
    tile("rewrites read", c.reads, p.reads, `${c.made} new written`),
    tile("agent tool calls", c.agent_calls, p.agent_calls, `MCP ${num(c.mcp_calls)} · API ${num(c.api_calls)} · Ask ${num(c.ask)}${c.agents ? ` · ${c.agents} addresses` : ""}`),
    tile("MCP connections", c.mcp_connects, p.mcp_connects),
    tile("agent setups copied", c.copies, p.copies),
    tile("support clicks", c.give, p.give, `${c.support} opened the support window`),
    `<div class="tile"><b>${money ? `$${num(money.dollars)}` : "…"}</b><span>paid (Stripe)</span><small>${money ? (money.error ? "Stripe can't be read" : `${money.paid} payments · ${money.started} checkouts started`) : "reading Stripe…"}</small></div>`,
    `<div class="tile"><b>${acc ? num(acc.total) : "?"}</b><span>accounts</span><small>${acc ? `${acc.new} new · ${acc.paid} paid · ${acc.granted} granted · ${num(acc.key_requests)} key requests` : ""}</small></div>`,
    tile("errors shown", c.errors, p.errors, "", true),
  ].join("");

  // charts
  const C = { ...D.chart };
  if (!D.hourly) {               // days before anything was recorded (before launch) are left out
    const keys = Object.keys(C).filter((k) => k !== "labels");
    const first = C.labels.findIndex((_, i) => keys.some((k) => C[k][i]));
    if (first > 0) for (const k of ["labels", ...keys]) C[k] = C[k].slice(first);
  }
  chart("#c-people", C.labels, [{ name: "from X", data: C.x, cls: "s1" }, { name: "other", data: C.other, cls: "s2" }], true, false);
  chart("#c-did", C.labels, [{ name: "searches", data: C.searches, cls: "s2" }, { name: "papers opened", data: C.opened, cls: "s4" },
    { name: "topic maps", data: C.topics, cls: "s5" }, { name: "author pages", data: C.people, cls: "s6" }, { name: "rewrites read", data: C.reads, cls: "s3" }], false);
  chart("#c-agents", C.labels, [{ name: "MCP", data: C.mcp, cls: "s5" }, { name: "API", data: C.api, cls: "s2" }, { name: "Ask", data: C.ask, cls: "s4" }], true);

  // what people look for
  const lk = D.looked, sp = D.speed;
  table("#t-searches", lk.searches, early ? "Recorded from Oct 17." : "No searches.");
  table("#t-empty", lk.empty, "None.");
  table("#t-topics", lk.topics, early ? "Recorded from Oct 17." : "No topic maps.");
  table("#t-people", lk.people, early ? "Recorded from Oct 17." : "None.");
  table("#t-papers", lk.papers);
  table("#t-cant", lk.cant_open, "None.");
  $("#speed-search").textContent = sp.search_median != null ? `Wait: typically ${sp.search_median} s, slowest tenth over ${sp.search_p90} s. Visitors per search.` : "Visitors per search.";
  $("#speed-map").textContent = sp.map_median != null ? `New maps took typically ${sp.map_median} s (slowest tenth over ${sp.map_p90} s).` : "";

  // flow
  const f = D.flow;
  $("#flow-line").innerHTML = `<b>${num(f.visits)}</b> visits · <b>${f.visits ? Math.round(100 * f.bounced / f.visits) : 0}%</b> left after one page without doing anything` +
    (f.median_min != null ? ` · a visit of more than one step lasts typically <b>${f.median_min} min</b>` : "");
  table("#t-entries", f.entries); table("#t-exits", f.exits); table("#t-after", f.after_search);
  table("#t-paths", f.paths); table("#t-busy", f.left_busy, "None."); table("#t-errors", f.errors, "None.");
  table("#t-refs", f.refs, "Direct visits only."); table("#t-devices", f.devices);

  // agents
  const A = D.agents;
  $("#agent-line").innerHTML = `<b>${num(c.agent_calls)}</b> calls from <b>${num(c.agents)}</b> addresses` +
    (A.limited ? ` · ${num(A.limited)} refused for the rate limit` : "") +
    (A.busiest.length ? ` · the busiest address made ${num(A.busiest[0][1])}` : "");
  table("#t-tools", A.tools); table("#t-apps", A.apps); table("#t-programs", A.programs, "None.");
  table("#t-queries", A.queries, early ? "Recorded from Oct 17." : "None.");

  drawVisits();
  $("#took").textContent = `Computed in ${D.took_ms} ms.`;
}

function drawFeed() {
  const rows = D.live.feed.filter((f) => feedFilter === "all" || f[1] === feedFilter);
  for (const b of document.querySelectorAll("#feed-filter button")) b.setAttribute("aria-pressed", String(b.dataset.f === feedFilter));
  $("#feed").innerHTML = rows.length ? rows.map(([at, kind, v, text, mark, href]) =>
    `<li class="${kind} ${mark}"><time>${clock(at)}</time>${kind === "agent" ? `<span class="who bot">AI</span>` : who(v)}<span>${link(text, href)}</span></li>`).join("")
    : `<li class="none">Quiet.</li>`;
}

function drawVisits() {
  const withDay = D.range !== "today";
  $("#visits").innerHTML = D.visits.length ? D.visits.map((v) => {
    const on = opened.has(v.key);
    const head = `<button class="row" aria-expanded="${on}" data-visit="${esc(v.key)}">${who(v.v)}` +
      `<span class="what">${esc(v.summary || "(no page)")}<small>${[v.device, v.ref && `from ${v.ref}`, v.did.length ? `did: ${v.did.join(", ")}` : "did nothing"].filter(Boolean).map(esc).join(" · ")}</small></span>` +
      `<span class="when">${clock(v.start, withDay)}<small>${v.pages} page${v.pages === 1 ? "" : "s"} · ${length(v.length)}</small></span></button>`;
    const steps = on ? `<ol class="steps">${v.steps.map(([at, text, mark, href]) =>
      `<li class="${mark}"><time>${clock(at)}</time><span>${link(text, href)}</span></li>`).join("")}</ol>` : "";
    return `<li id="visit-${esc(v.key)}">${head}${steps}</li>`;
  }).join("") : `<li class="none">No visits in this period.</li>`;
}

// ------------------------------------------------------------------ loading and controls
async function load() {
  if (loading) return;
  loading = true;
  try {
    const r = await fetch(`data?range=${range}`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    D = await r.json();
    render();
    document.body.classList.remove("stale");
  } catch (e) {
    document.body.classList.add("stale");
    $("#updated").textContent = `Couldn't refresh (${e.message}); trying again.`;
  } finally {
    loading = false;
  }
}

$("#ranges").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  range = b.dataset.range;
  history.replaceState(null, "", range === "today" ? "./" : `?range=${range}`);
  opened.clear();
  load();
});
$("#feed-filter").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b || !D) return;
  feedFilter = b.dataset.f;
  drawFeed();
});
document.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-visit]");
  if (!b || !D) return;
  const key = b.dataset.visit;
  const fromNow = b.closest("#active");
  if (fromNow) {                                   // a visitor on the site now: unfold their visit below
    if (!D.visits.some((v) => v.key === key) && D.range !== "today") { range = "today"; history.replaceState(null, "", "./"); opened.add(key); return load(); }
    opened.add(key);
  } else if (opened.has(key)) opened.delete(key); else opened.add(key);
  drawVisits();
  if (fromNow) document.getElementById(`visit-${key}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
});

load();
setInterval(() => { if (!document.hidden) load(); }, 10000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) load(); });
