/**
 * A study's page: what it is, who wrote it and from where, where it belongs in science, what it builds on and what
 * built on it, and where to walk next. Written for someone who lands here from a shared link and knows nothing
 * about the site, the field or how papers work.
 *
 * Why it is laid out this way (the human-computer interaction behind it):
 * - Overview first, then details on demand (Shneiderman 1996, doi:10.1109/vl.1996.545307): the head says what the
 *   study is in one glance; its timeline shows its whole citation life; the lists are the details.
 * - People judge a page in a fraction of a second, and what looks good is judged easier to use (Lindgaard et al.
 *   2006, doi:10.1080/01449290500330448; Tractinsky et al. 2000, doi:10.1016/s0953-5438(00)00031-x): one calm column,
 *   a clear hierarchy, nothing that isn't needed.
 * - Information scent (Pirolli & Card; Chi et al. 2001, doi:10.1145/365024.365325): every link says what is behind
 *   it (an author's studies and h-index, a group's size and what its researchers do, a study's year and citations).
 * - Berrypicking (Bates 1989, doi:10.1108/eb024320): readers move by footnote chasing (what it cites), citation
 *   searching (what cites it), author searching, area scanning (its field's map) and subject searches (its
 *   keywords). The page offers each move, each a click.
 * - Fields, bridges and foundations (Choe et al. 2024, doi:10.48550/arxiv.2405.07267): researchers browsing citation
 *   networks look for the field a paper is in and its foundations, and pick better by connections than by layout:
 *   so the field map's group is named, and lists say how each study is connected.
 * - Getting lost in hypertext (Otter & Johnson 2000, doi:10.1016/s0953-5438(00)00030-8): the path walked so far stays
 *   on the page (trail.ts), and studies already opened are marked (as CiteSee marks known papers, Chang et al. 2023,
 *   doi:10.1145/3544548.3580847); the map in the corner keeps "you are here" (overview + detail: Hornbæk et al.
 *   2002, doi:10.1145/586081.586086).
 */
import { API } from "./jobs.ts";
import { fieldUrl } from "./fields.ts";
import type { Hit } from "./paper.ts";
import type { Line, Spot } from "./atlas.ts";
import { renderTrail } from "./trail.ts";

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
const n = (x: number) => x.toLocaleString("en");
const plural = (k: number, one: string, many = `${one}s`) => `${n(k)} ${k === 1 ? one : many}`;
const plain = (s?: string | null) => (s ?? "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();

export interface Item {
  id: string; title: string; year: number | null; citations: number; type?: string; first_author?: string;
  authors_count?: number; x?: number; y?: number; readable?: boolean;
}
interface Author { id: string; name: string; position: number; works?: number; h?: number; at?: number[] }
interface Org { id: string; name: string; country: string | null; kind: string | null }
interface Unit { id: number; name: string; about?: string | null; size: number }
interface InMap { key: string; topic: string; definition: string | null; papers: number; via: "topic" | "cited"; group?: Unit; sub?: Unit }
interface Side { count: number; years: Record<string, number>; plot: [number, number, string][]; top: Item[]; newest?: Item[] }
export interface Context {
  id: string; in_index: boolean; maps: InMap[]; year?: number | null; title?: string; type?: string | null; citations?: number;
  references_listed?: number; authors?: Author[]; authors_count?: number; institutions?: Org[];
  affiliations?: "exact" | "none"; references?: Side; cited_by?: Side;
}

export interface Deps {
  go: (url: string) => void;
  row: (h: Hit, why?: string) => HTMLAnchorElement;
  hot: (key: string, on: boolean) => void;
  setWalk: (spots: Spot[], lines: Line[]) => void;
  walk: (ref: string) => Promise<any>;
  current: () => boolean;           // still the page shown (the reader hasn't moved on)
}

const ctxCache = new Map<string, Promise<Context>>();
export function fetchContext(ref: string): Promise<Context> {
  let p = ctxCache.get(ref);
  if (!p) {
    p = fetch(`${API}/api/data/works/${encodeURIComponent(ref)}/context`, { signal: AbortSignal.timeout(20000) })
      .then(async (r) => { if (!r.ok) throw new Error(String(r.status)); return r.json(); });
    p.catch(() => ctxCache.delete(ref));
    ctxCache.set(ref, p);
    if (ctxCache.size > 40) ctxCache.delete(ctxCache.keys().next().value!);
  }
  return p;
}

async function fetchLinks(id: string, direction: "references" | "cited_by", sort: string, offset: number, limit: number) {
  const p = new URLSearchParams({ direction, sort, offset: String(offset), limit: String(limit) });
  const r = await fetch(`${API}/api/data/works/${encodeURIComponent(id)}/links?${p}`, { signal: AbortSignal.timeout(20000) });
  if (!r.ok) throw new Error(String(r.status));
  return (await r.json()) as { total: number; items: Item[] };
}

const TYPES: Record<string, string> = {
  article: "Research article", review: "Review", preprint: "Preprint", book: "Book", "book-chapter": "Book chapter",
  "conference-paper": "Conference paper", dissertation: "Thesis", report: "Report", letter: "Letter", editorial: "Editorial",
  dataset: "Dataset", "peer-review": "Peer review", erratum: "Correction", paratext: "Front matter", other: "Publication",
};
const TYPE_HELP: Record<string, string> = {
  article: "A study reporting new research, usually checked by other scientists before publication (peer review).",
  review: "A review sums up many earlier studies on one question: often the best place to start.",
  preprint: "A preprint is shared before peer review: other scientists haven't checked it yet.",
};
const regionName = (() => {
  try { const d = new Intl.DisplayNames(["en"], { type: "region" }); return (c: string) => d.of(c) ?? c; }
  catch { return (c: string) => c; }
})();

/** An item of our lists as a search hit, for the shared row (title, authors, journal, year, citations, badges). */
export const asHit = (x: Item): Hit => ({
  id: x.id, pmcid: "", title: plain(x.title) || "(no title)", year: x.year ? String(x.year) : undefined,
  authors: x.first_author ? (x.authors_count && x.authors_count > 1 ? `${x.first_author} et al.` : x.first_author) : undefined,
  readable: Boolean(x.readable), citations: x.citations > 0 ? x.citations : undefined, x: x.x, y: x.y,
  type: x.type,
});

// ================================================================ the page
/** The study's head and abstract from its record (`d`, the data API's /works), then everything around it. */
export function renderStudy(box: HTMLElement, d: any, deps: Deps) {
  const retracted = d.is_retracted === true || d.is_retracted === 1 || d.is_retracted === "1";
  const type = d.type ?? "article";
  const kicker = `<span class="st-type" title="${esc(TYPE_HELP[type] ?? "")}">${esc(TYPES[type] ?? type.replace(/-/g, " "))}</span>` +
    (d.year ? ` · ${esc(String(d.year))}` : "") + (d.venue ? ` · <i class="st-venue">${esc(d.venue)}</i>` : "");
  const original = d.doi ? `https://doi.org/${d.doi}` : d.url || d.openalex;
  const scholar = `https://scholar.google.com/scholar?q=${encodeURIComponent(d.title)}`;
  const readable = Boolean(d.smr);
  const act = d.partial
    ? `<a class="btn ghost" href="${esc(d.openalex)}" target="_blank" rel="noopener">See it on OpenAlex ↗</a>
       <a class="btn ghost" href="${esc(scholar)}" target="_blank" rel="noopener">Find it on Google Scholar ↗</a>`
    : readable
    ? `<a class="btn" href="?paper=${encodeURIComponent(d.doi || d.pmcid)}" data-go>Read it in plain words</a>
       <a class="btn ghost" href="${esc(original)}" target="_blank" rel="noopener">The original ↗</a>`
    : `<a class="btn" href="${esc(original)}" target="_blank" rel="noopener">Read the original ↗</a>
       ${d.pdf_url ? `<a class="btn ghost" href="${esc(d.pdf_url)}" target="_blank" rel="noopener">PDF ↗</a>` : ""}`;
  const note = d.partial
    ? "We only know this study through the studies that cite it: OpenAlex has no English abstract for it, so we can't search it by meaning or rewrite it."
    : readable ? "Its full text is open: we can rewrite it in plain words, every finding kept."
    : d.pmc_oa ? "Its licence doesn't allow us to share a rewritten copy." : "Its full text isn't openly available to us, so we can't rewrite it.";
  box.innerHTML = `
    <nav class="trail" aria-label="Your path"></nav>
    <article class="st">
      <header class="st-head">
        <p class="st-kicker">${kicker}</p>
        <h1>${esc(d.title)}</h1>
        <div class="st-authors" aria-label="Authors">${esc(d.authorsLine || "")}</div>
        <div class="st-orgs"></div>
        ${retracted ? `<p class="w-retracted" role="note"><b>This study was retracted.</b> The journal withdrew it, so its
          results should not be relied on.</p>` : ""}
        <div class="st-act">${act}<span class="st-note">${esc(note)}</span></div>
      </header>
      <section class="st-facts" aria-label="At a glance"></section>
      ${d.abstract ? `<section class="st-abs" aria-labelledby="st-abs-h"><h2 id="st-abs-h" class="st-label">Abstract</h2>
        <div class="st-abs-text">${esc(d.abstract)}</div>
        <p class="st-abs-note">In the authors' words. ${readable ? "The plain-words version keeps every finding." : ""}</p></section>` : ""}
      <section class="st-where" aria-labelledby="st-where-h" hidden></section>
      <section class="st-time" aria-labelledby="st-time-h" hidden></section>
      <section class="st-lists" aria-label="Its references and the studies citing it" hidden></section>
      <section class="st-walk" aria-labelledby="st-walk-h" hidden></section>
      <div class="st-links">
        ${d.doi ? `<a href="https://doi.org/${esc(d.doi)}" target="_blank" rel="noopener">DOI ${esc(d.doi)} ↗</a>` : ""}
        <a href="${esc(d.openalex)}" target="_blank" rel="noopener">OpenAlex ↗</a>
        <a href="${esc(scholar)}" target="_blank" rel="noopener">Google Scholar ↗</a>
        ${d.pmcid ? `<a href="https://pmc.ncbi.nlm.nih.gov/articles/${esc(d.pmcid)}/" target="_blank" rel="noopener">PubMed Central ↗</a>` : ""}
        ${d.license ? `<span class="quiet">Licence: ${esc(d.license)}</span>` : ""}
        ${d.publisher ? `<span class="quiet">Publisher: ${esc(d.publisher)}</span>` : ""}
      </div>
    </article>
    <p class="you-are-here">You are here, on the map of science: blue, what it builds on; gold, what cites it.<br>Click the map to look around.</p>`;
  wire(box, deps);
  renderTrail(box.querySelector(".trail")!, deps.go, `?work=${encodeURIComponent(d.id)}`);
  facts(box, d, null);
  topicsFallback(box, d, deps);      // keywords right away; replaced by the field maps holding it once known
}

/** Everything the data API knows around the study (authors, organisations, citations, field maps). */
export function fillContext(box: HTMLElement, d: any, c: Context, deps: Deps) {
  authors(box, d, c, deps);
  facts(box, d, c);
  where(box, d, c, deps);
  timeline(box, d, c, deps);
  lists(box, d, c, deps);
  onMap(c, deps);
  walkTeaser(box, d, deps);
}

function wire(el: HTMLElement, deps: Deps) {
  for (const a of el.querySelectorAll<HTMLAnchorElement>("a[data-go]")) {
    if (a.dataset.wired) continue;
    a.dataset.wired = "1";
    a.onclick = (e) => { if (e.metaKey || e.ctrlKey || e.shiftKey || e.button) return; e.preventDefault(); deps.go(a.getAttribute("href")!); };
  }
}

// ---------------------------------------------------------------- authors and where they signed from
const SHOWN = 8;
function authors(box: HTMLElement, d: any, c: Context, deps: Deps) {
  const host = box.querySelector<HTMLElement>(".st-authors"), orgsBox = box.querySelector<HTMLElement>(".st-orgs");
  if (!host || !orgsBox) return;
  // names from the record (first 12, in order) fill positions OpenAlex could not tie to a person
  const named: string[] = (d.authors as string | undefined)?.split(" | ").filter(Boolean) ?? [];
  const byPos = new Map((c.authors ?? []).map((a) => [a.position, a]));
  const total = Math.max(c.authors_count ?? 0, named.length, c.authors?.length ?? 0);
  const people: (Author | { name: string; position: number })[] = [];
  for (let p = 1; p <= total; p++) {
    const a = byPos.get(p) ?? (named[p - 1] ? { name: named[p - 1], position: p } : null);
    if (a) people.push(a);
  }
  if (!people.length) return;
  const orgs = c.institutions ?? [];
  const exact = c.affiliations === "exact";
  const one = (a: Author | { name: string; position: number }) => {
    const at = exact && "at" in a ? (a.at ?? []) : [];
    const sup = at.length ? `<sup>${at.map((i) => i + 1).join(",")}</sup>` : "";
    if (!("id" in a)) return `<span class="st-au">${esc(a.name)}${sup}</span>`;
    const tip = [a.works ? plural(a.works, "study", "studies") : "", a.h ? `h-index ${a.h}` : "",
                 at.length ? orgs[at[0]]?.name ?? "" : ""].filter(Boolean).join(" · ");
    return `<a class="st-au" href="?author=${esc(a.id)}" data-go title="${esc(tip)}">${esc(a.name)}</a>${sup}`;
  };
  // long lists: the first authors, then the last (often the lab's head), the middle on demand
  const fold = people.length > SHOWN + 2;
  const head = fold ? people.slice(0, SHOWN) : people, middle = fold ? people.slice(SHOWN, -1) : [], last = fold ? people[people.length - 1] : null;
  const missing = total - people.length;          // listed by OpenAlex, but beyond the names we have
  host.innerHTML = head.map(one).join(", ") +
    (fold ? `<span class="st-au-more" hidden>, ${middle.map(one).join(", ")}</span>` +
      `<span class="st-gap">, <button type="button" class="st-more">${middle.length} more</button></span>, ${one(last!)}` : "") +
    (missing > 0 ? ` <span class="quiet">and ${plural(missing, "other")}</span>` : "");
  host.querySelector<HTMLButtonElement>(".st-more")?.addEventListener("click", () => {
    host.querySelector<HTMLElement>(".st-au-more")!.hidden = false;
    host.querySelector(".st-gap")?.remove();
  });
  // organisations: numbered when we know who signed from where (as on the paper itself), else simply listed
  if (orgs.length) {
    const label = (o: Org, i: number) => `${exact ? `<sup>${i + 1}</sup>` : ""}<a href="?institution=${esc(o.id)}" data-go ` +
      `title="${esc([o.kind ? o.kind.replace(/^\w/, (x) => x.toUpperCase()) : "", o.country ? regionName(o.country) : ""].filter(Boolean).join(", "))}">` +
      `${esc(o.name)}</a>${o.country ? `<span class="st-cc">, ${esc(regionName(o.country))}</span>` : ""}`;
    const list = orgs.map((o, i) => ({ o, i }));
    const head = list.slice(0, 6), rest = list.slice(6);
    orgsBox.innerHTML = (exact ? "" : `<span class="st-from">From</span> `) +
      head.map(({ o, i }) => label(o, i)).join(`<span class="st-sep"> · </span>`) +
      (rest.length ? `<span class="st-org-more" hidden><span class="st-sep"> · </span>${rest.map(({ o, i }) => label(o, i)).join(`<span class="st-sep"> · </span>`)}</span>` +
        ` <button type="button" class="st-more">and ${plural(rest.length, "more organisation")}</button>` : "");
    orgsBox.querySelector<HTMLButtonElement>(".st-more")?.addEventListener("click", (e) => {
      orgsBox.querySelector<HTMLElement>(".st-org-more")!.hidden = false;
      (e.currentTarget as HTMLElement).remove();
    });
    if (!exact) orgsBox.title = "The organisations on this study. We don't yet know which author signed from which.";
  }
  wire(host, deps); wire(orgsBox, deps);
}

// ---------------------------------------------------------------- at a glance
function facts(box: HTMLElement, d: any, c: Context | null) {
  const host = box.querySelector<HTMLElement>(".st-facts");
  if (!host) return;
  const cites = Number(d.citations ?? c?.citations ?? 0);
  const year = Number(d.year ?? c?.year ?? 0);
  const now = new Date().getFullYear();
  const perYear = year && now - year >= 2 && cites ? Math.round(cites / (now - year + 0.5)) : 0;
  const peak = c?.cited_by && Object.keys(c.cited_by.years).length > 2
    ? Object.entries(c.cited_by.years).filter(([y]) => Number(y) < now).sort((a, b) => b[1] - a[1])[0] : null;
  const tile = (big: string, small: string, help = "") =>
    `<div class="st-fact"${help ? ` title="${esc(help)}"` : ""}><b>${big}</b><span>${small}</span></div>`;
  const out = [
    tile(cites ? n(cites) : "0", cites === 1 ? "citation" : "citations",
         "How many later studies list this one in their references (OpenAlex's count)." +
         (perYear ? ` About ${n(perYear)} a year.` : "")),
  ];
  if (peak) out.push(tile(peak[0], "year cited most", `${n(peak[1])} studies from ${peak[0]} cite it, more than from any other full year.`));
  const listed = c?.references_listed ?? -1, linked = c?.references?.count ?? 0;
  if (c) out.push(tile(n(listed > 0 ? listed : linked), listed > 0 ? "references" : "references known",
    listed > linked && linked ? `It lists ${n(listed)} references; ${n(linked)} of them are studies we can link to.` :
    "The earlier studies it lists in its references."));
  const people = c?.authors_count ?? 0;
  if (people) out.push(tile(n(people), people === 1 ? "author" : "authors",
    c?.institutions?.length ? `From ${plural(c.institutions.length, "organisation")}.` : ""));
  if (d.field) out.push(tile(`<span class="st-fact-word">${esc(d.field)}</span>`, "its field (OpenAlex)", `OpenAlex's label for it: ${[d.field, d.topic].filter(Boolean).join(" › ")}. ` +
    "These labels are automatic and often rough."));
  host.innerHTML = out.join("");
}

// ---------------------------------------------------------------- where it belongs in science
function mapLink(m: InMap) {
  const at = m.group ? `${m.group.id}${m.sub ? `.${m.sub.id}` : ""}` : "";
  return fieldUrl(m.topic, m.definition) + (at ? `&at=${at}` : "");
}

/** Keywords and OpenAlex's subject (before, or without, a field map): each opens the research groups of that topic. */
function topicsFallback(box: HTMLElement, d: any, deps: Deps) {
  const host = box.querySelector<HTMLElement>(".st-where");
  if (!host) return;
  const kws: string[] = [...new Set<string>([d.topic, ...((d.keywords as string | undefined)?.split(" | ") ?? [])]
    .filter(Boolean).map((x: string) => x.trim()))].slice(0, 10);
  if (!kws.length) return;
  host.hidden = false;
  host.innerHTML = `<h2 id="st-where-h" class="st-h">Its topics</h2>
    <p class="st-sub">Each topic opens a map of its research: the groups of studies inside it and what each reads.</p>
    <div class="st-chips">${kws.map((k) => `<a class="st-chip" href="?topics=${encodeURIComponent(k)}" data-go ` +
      `title="${esc(`The research groups of “${k}”`)}">${esc(k)}</a>`).join("")}</div>`;
  wire(host, deps);
}

function where(box: HTMLElement, d: any, c: Context, deps: Deps) {
  const host = box.querySelector<HTMLElement>(".st-where");
  if (!host || !c.maps.length) return;
  const chips = host.querySelector(".st-chips")?.outerHTML ?? "";
  const card = (m: InMap) => {
    const unit = m.sub ?? m.group;
    const path = [m.group, m.sub].filter(Boolean) as Unit[];
    const lead = m.via === "cited"
      ? `Cited as a foundation by the field of <b>${esc(m.topic)}</b>`
      : `Part of the field of <b>${esc(m.topic)}</b>`;
    return `<a class="st-map" href="${esc(mapLink(m))}" data-go>
      <span class="st-map-lead">${lead} <span class="quiet">· ${n(m.papers)} studies</span></span>
      ${path.length ? `<span class="st-map-path">${path.map((u, i) => `<span class="${i === path.length - 1 ? "here" : ""}">${esc(u.name)}</span>`)
        .join(`<span class="st-map-sep" aria-hidden="true">›</span>`)}</span>` : ""}
      ${unit?.about ? `<span class="st-map-about">${esc(unit.about)}</span>` : ""}
      <span class="st-map-go">${unit ? `Open this group of ${n(unit.size)} studies` : "Open the field's map"} →</span>
    </a>`;
  };
  const first = c.maps.slice(0, 2), rest = c.maps.slice(2);
  host.hidden = false;
  host.innerHTML = `<h2 id="st-where-h" class="st-h">Where it belongs</h2>
    <p class="st-sub">Its research community in our maps of whole fields, where studies are grouped by what they cite and the
      words they use. Descriptions written by an AI from each group's studies.</p>
    ${first.map(card).join("")}
    ${rest.length ? `<details class="st-more-maps"><summary>Also in ${plural(rest.length, "other field map")}</summary>${rest.map(card).join("")}</details>` : ""}
    ${chips ? `<p class="st-sub st-sub2">Or map one of its topics:</p>${chips}` : ""}`;
  wire(host, deps);
}

// ---------------------------------------------------------------- its life in citations: a timeline
function timeline(box: HTMLElement, d: any, c: Context, deps: Deps) {
  const host = box.querySelector<HTMLElement>(".st-time");
  const refs = c.references?.plot ?? [], citing = c.cited_by?.plot ?? [];
  const year = Number(d.year ?? c.year ?? 0);
  if (!host || !year || refs.length + citing.length < 3) return;
  host.hidden = false;
  // drawn at the column's real width, so its text stays readable on a phone; again when the width changes
  let drawnW = 0;
  const draw = (W: number) => {
    const now = new Date().getFullYear();
    const ry0 = refs.map((p) => p[0]).filter((y) => y > 1500 && y <= year + 1).sort((a, b) => a - b);
    const yrs = [...ry0, ...citing.map((p) => p[0])].sort((a, b) => a - b);
    // the oldest few references (an 1859 classic) would squeeze the rest: the axis starts at the references' 10th
    // percentile, older ones sit at its left edge (said under the axis)
    const lo = Math.min(year - 3, ry0.length ? ry0[Math.floor(ry0.length * 0.1)] : year - 10);
    const older = ry0.filter((y) => y < lo).length;
    const hi = Math.max(now, year + 3);
    const H = 230, padL = 8, padR = 8, top = 22, base = 168, barH = 40;
    const x = (y: number) => padL + ((Math.max(lo, Math.min(hi, y)) - lo + 0.5) / (hi - lo + 1)) * (W - padL - padR);
    const maxC = Math.max(10, d.citations ?? 0, ...refs.map((p) => p[1]), ...citing.map((p) => p[1]));
    const y = (k: number) => base - 8 - (Math.log10(Math.max(0, k) + 1) / Math.log10(maxC + 1)) * (base - 8 - top);
    // dots of one year spread across that year's width (a stable spread: the same study always lands in the same place)
    const jitter = (id: string) => { let h = 7; for (const ch of id) h = (h * 31 + ch.charCodeAt(0)) >>> 0; return ((h % 1000) / 1000 - 0.5) * ((W - padL - padR) / (hi - lo + 1)) * 0.8; };
    // citations per year (gold) and references per year (blue), as small bars under the axis
    const cy = c.cited_by?.years ?? {}, ry = c.references?.years ?? {};
    // each side on its own scale: a study lists tens of references but can gather thousands of citations
    const maxOf = (ys: Record<string, number>) => Math.max(1, ...Object.values(ys));
    const bw = Math.max(1.5, ((W - padL - padR) / (hi - lo + 1)) * 0.62);
    const bars = (ys: Record<string, number>, cls: string) => Object.entries(ys).filter(([k]) => Number(k) >= lo && Number(k) <= hi)
      .map(([k, v]) => `<rect class="${cls}" x="${(x(Number(k)) - bw / 2).toFixed(1)}" y="${base + 6}" width="${bw.toFixed(1)}" ` +
        `height="${Math.max(1, (v / maxOf(ys)) * barH).toFixed(1)}"><title>${n(v)} ${cls === "b-cite" ? `citing studies from ${k}` : `references from ${k}`}</title></rect>`).join("");
    const dots = (pts: [number, number, string][], cls: string) => pts.map(([yy, k, id]) =>
      `<circle class="${cls}" data-id="${id}" cx="${(x(yy) + jitter(id)).toFixed(1)}" cy="${y(k).toFixed(1)}" r="${k >= maxC / 20 ? 4.2 : 3}"/>`).join("");
    const ticks: number[] = [];
    const span = hi - lo, perTick = span * 64 / W;   // a label every ~64 px at least
    const step = [1, 2, 5, 10, 20, 25, 50, 100].find((k) => k >= perTick) ?? 200;
    for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
    const cites = Number(d.citations ?? 0);
    host.innerHTML = `<h2 id="st-time-h" class="st-h">Its place in time</h2>
      <p class="st-sub">Each dot is a study, placed by its year; higher means cited more. <span class="k-ref-t">Blue</span>: what
        this study builds on. <span class="k-cite-t">Gold</span>: the studies that cite it${c.cited_by && c.cited_by.count > citing.length
          ? ` (the ${n(citing.length)} most cited of ${n(c.cited_by.count)})` : ""}. The bars count them year by year.
        ${now} is not over yet. Point at a dot to see it; click to go there.</p>
      <div class="st-svg-wrap">
        <svg class="st-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(`Timeline: ${n(c.references?.count ?? 0)} references from ` +
          `${yrs[0] ?? year} on, the study in ${year}, then ${n(c.cited_by?.count ?? 0)} citing studies up to ${now}.`)}">
          <line class="axis" x1="${padL}" x2="${W - padR}" y1="${base}" y2="${base}"/>
          ${ticks.map((t) => `<line class="tick" x1="${x(t)}" x2="${x(t)}" y1="${top - 6}" y2="${base}"/><text class="tick-t" x="${x(t)}" y="${H - 4}">${t}</text>`).join("")}
          ${bars(ry, "b-ref")}${bars(cy, "b-cite")}
          <line class="now-line" x1="${x(year)}" x2="${x(year)}" y1="${top - 10}" y2="${base + 6 + barH}"/>
          ${dots(refs, "d-ref")}${dots(citing, "d-cite")}
          <circle class="d-self" cx="${x(year)}" cy="${y(cites)}" r="7"/>
          ${older ? `<text class="tick-t older-t" x="${padL}" y="${base - 2}" text-anchor="start">← ${n(older)} older</text>` : ""}
          <text class="self-t" x="${x(year)}" y="${top - 12}" text-anchor="${x(year) > W - 120 ? "end" : x(year) < 120 ? "start" : "middle"}">this study, ${year}</text>
        </svg>
        <div class="st-tip" role="status" hidden></div>
      </div>`;
    const svg = host.querySelector("svg")!, tip = host.querySelector<HTMLElement>(".st-tip")!;
    const index = new Map<string, { title?: string; year: number; cites: number }>();
    for (const [yy, k, id] of refs) index.set(id, { year: yy, cites: k });
    for (const [yy, k, id] of citing) index.set(id, { year: yy, cites: k });
    // titles for the tooltip: the lists' items carry them; the rest come from the citation index on hover
    for (const it of [...(c.references?.top ?? []), ...(c.cited_by?.top ?? []), ...(c.cited_by?.newest ?? [])])
      if (index.has(it.id)) index.get(it.id)!.title = plain(it.title);
    let lastTip = "";
    const showTip = async (el: SVGCircleElement) => {
      const id = el.dataset.id!, info = index.get(id);
      if (!info) return;
      lastTip = id;
      const r = el.getBoundingClientRect(), wrap = host.querySelector(".st-svg-wrap")!.getBoundingClientRect();
      tip.style.left = `${Math.min(wrap.width - 260, Math.max(0, r.left - wrap.left - 120))}px`;
      tip.style.top = `${r.top - wrap.top - 12}px`;
      const draw = () => {
        tip.innerHTML = `<b>${esc(info.title ?? "…")}</b><span>${info.year} · ${info.cites === 1 ? "cited once" : `cited ${n(info.cites)} times`}</span>`;
        tip.hidden = false;
      };
      draw();
      if (!info.title) {
        const t = await titleOf(id);
        if (lastTip === id && t) { info.title = t; draw(); }
      }
    };
    svg.addEventListener("pointerover", (e) => { const el = e.target as SVGCircleElement; if (el.dataset?.id) { showTip(el); deps.hot(el.dataset.id, true); } });
    svg.addEventListener("pointerout", (e) => { const el = e.target as SVGCircleElement; if (el.dataset?.id) { tip.hidden = true; lastTip = ""; deps.hot(el.dataset.id, false); } });
    svg.addEventListener("click", (e) => { const el = e.target as SVGCircleElement; if (el.dataset?.id) deps.go(`?work=${el.dataset.id}`); });
  };
  const fit = () => {
    const w = Math.round(host.clientWidth);
    if (w > 0 && Math.abs(w - drawnW) > 24) { drawnW = w; draw(Math.max(300, w)); }
  };
  fit();
  if ("ResizeObserver" in window) {
    const ro = new ResizeObserver(() => { if (!host.isConnected) ro.disconnect(); else fit(); });
    ro.observe(host);
  }
}

const titles = new Map<string, Promise<string>>();
function titleOf(id: string): Promise<string> {
  let p = titles.get(id);
  if (!p) {
    p = fetch(`${API}/api/data/works/${id}`, { signal: AbortSignal.timeout(8000) })
      .then((r) => r.ok ? r.json() : null).then((x) => plain(x?.title)).catch(() => "");
    titles.set(id, p);
  }
  return p;
}

// ---------------------------------------------------------------- what it builds on, what built on it
type Dir = "references" | "cited_by";
function lists(box: HTMLElement, d: any, c: Context, deps: Deps) {
  const host = box.querySelector<HTMLElement>(".st-lists");
  if (!host) return;
  const count: Record<Dir, number> = { references: c.references?.count ?? 0, cited_by: c.cited_by?.count ?? 0 };
  if (!count.references && !count.cited_by) return;
  host.hidden = false;
  let dir: Dir = count.cited_by ? "cited_by" : "references";
  let sort = "cited";
  const help: Record<Dir, string> = {
    references: "Earlier studies this one lists in its references: the work it stands on. Following them is how researchers trace an idea back.",
    cited_by: "Later studies that list this one in their references: where its ideas went next.",
  };
  host.innerHTML = `<h2 class="st-h">What it builds on, and what built on it</h2>
    <div class="st-tabs" role="group" aria-label="Which list">
      <button type="button" data-dir="references" ${count.references ? "" : "disabled"}><span class="lg k-ref"></span>Builds on <b>${n(count.references)}</b></button>
      <button type="button" data-dir="cited_by" ${count.cited_by ? "" : "disabled"}><span class="lg k-cite"></span>Cited by <b>${n(count.cited_by)}</b></button>
      <select aria-label="Order"><option value="cited">Most cited first</option><option value="newest">Newest first</option><option value="oldest">Oldest first</option></select>
    </div>
    <p class="st-sub st-list-help"></p>
    <div class="st-list" aria-live="polite"></div>`;
  const list = host.querySelector<HTMLElement>(".st-list")!, helpEl = host.querySelector<HTMLElement>(".st-list-help")!;
  const sel = host.querySelector("select")!;
  let run = 0;
  const draw = async (more = false) => {
    const me = ++run;
    for (const b of host.querySelectorAll<HTMLButtonElement>("[data-dir]")) b.setAttribute("aria-pressed", String(b.dataset.dir === dir));
    helpEl.textContent = help[dir];
    const have = more ? list.querySelectorAll(".hit").length : 0;
    const first = !more && sort === "cited" ? (dir === "references" ? c.references?.top : c.cited_by?.top) : null;
    let items: Item[], total = count[dir];
    if (first && (first.length >= Math.min(10, total))) items = first.slice(0, 10);
    else {
      list.querySelector(".more")?.setAttribute("disabled", "");
      try { const r = await fetchLinks(c.id, dir, sort, have, have ? 25 : 10); items = r.items; total = r.total; }
      catch { if (me === run && !more) list.innerHTML = `<p class="quiet">The list didn't come back. Try again in a moment.</p>`; return; }
    }
    if (me !== run) return;
    if (!more) list.replaceChildren();
    list.querySelector(".more")?.remove();
    for (const it of items) list.appendChild(itemRow(it, deps));
    const shown = list.querySelectorAll(".hit").length;
    if (shown < total) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "more";
      b.textContent = `Show more (${n(shown)} of ${n(total)})`;
      b.onclick = () => draw(true);
      list.appendChild(b);
    }
  };
  for (const b of host.querySelectorAll<HTMLButtonElement>("[data-dir]"))
    b.onclick = () => { if (b.dataset.dir !== dir) { dir = b.dataset.dir as Dir; draw(); } };
  sel.onchange = () => { sort = sel.value; draw(); };
  draw();
}

/** One study in a list: the shared result row, opening its page, marked when already opened. */
export function itemRow(it: Item, deps: Deps, why?: string): HTMLAnchorElement {
  const a = deps.row(asHit(it), why);
  a.href = `?work=${encodeURIComponent(it.id)}`;     // its page (which offers the plain-words version when there is one)
  return a;
}

// ---------------------------------------------------------------- the map in the corner
function onMap(c: Context, deps: Deps) {
  const spots: Spot[] = [];
  const add = (it: Item, kind: Spot["kind"]) => {
    if (it.x == null || it.y == null) return;
    spots.push({ id: it.id, x: it.x, y: it.y, kind, readable: it.readable, title: plain(it.title),
                 meta: [it.year, it.citations ? `cited ${n(it.citations)} times` : ""].filter(Boolean).join(" · ") });
  };
  for (const it of c.references?.top ?? []) add(it, "ref");
  for (const it of [...(c.cited_by?.top ?? []), ...(c.cited_by?.newest ?? [])]) add(it, "similar");
  deps.setWalk(spots, []);
}

// ---------------------------------------------------------------- keep walking
/** A taste of the citation walk (its own page): loaded when the reader gets near it (the walk takes 0.5-5 s). */
function walkTeaser(box: HTMLElement, d: any, deps: Deps) {
  const host = box.querySelector<HTMLElement>(".st-walk");
  if (!host) return;
  host.hidden = false;
  const url = `?walk=${encodeURIComponent(d.id)}`;
  host.innerHTML = `<h2 id="st-walk-h" class="st-h">Keep walking</h2>
    <p class="st-sub">The studies that cite much of the same work as this one form its <b>conversation</b>. What they lean on
      that this study doesn't cite is often the best thing to read next.</p>
    <div class="st-walk-list"></div>
    <p><a class="btn ghost" href="${esc(url)}" data-go>Open its citation walk →</a></p>`;
  wire(host, deps);
  const load = () => {
    host.querySelector<HTMLElement>(".st-walk-list")!.innerHTML = `<p class="loading-line">Following its citations…</p>`;
    return walkNow();
  };
  const walkNow = () => deps.walk(d.id).then((w) => {
    if (!deps.current()) return;
    const N = w.nodes, col = (k: string) => (w.columns[k] ?? []).map((id: string) => N[id]).filter(Boolean);
    const next = [...col("candidates").sort((a: any, b: any) => Number(b.answer) - Number(a.answer)), ...col("candidates_out")].slice(0, 3);
    const k = (w.similar ?? []).length;
    const listEl = host.querySelector<HTMLElement>(".st-walk-list")!;
    listEl.replaceChildren();
    if (!next.length) { listEl.innerHTML = `<p class="quiet">Nothing stands out yet: open the walk to see its conversation.</p>`; return; }
    const intro = document.createElement("p");
    intro.className = "st-sub";
    intro.textContent = `Read next, from its conversation of ${k} studies:`;
    listEl.appendChild(intro);
    for (const nd of next) {
      const it: Item = { id: nd.id, title: plain(nd.title), year: nd.year ?? null, citations: nd.cited_by ?? 0, x: nd.x, y: nd.y, readable: nd.readable };
      listEl.appendChild(itemRow(it, deps, nd.support ? `Cited by ${nd.support} of those ${k} studies, not by this one.` : undefined));
    }
  }).catch(() => {
    if (!deps.current()) return;
    host.querySelector<HTMLElement>(".st-walk-list")!.innerHTML =
      `<p class="quiet">We can't walk from this study: none of its references are in our data.</p>`;
    host.querySelector(".btn")?.remove();
  });
  if (!("IntersectionObserver" in window)) { load(); return; }
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); load(); } }, { rootMargin: "400px" });
  io.observe(host);
}
