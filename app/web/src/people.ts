/**
 * People: an author's page, and finding authors by name (the data API's /authors; method in
 * search/srl_search/people.py). An author's page is the topic map's one-screen overview (overview.ts) over their
 * own work: boxes of their fields (OpenAlex's label on each paper; subfields inside, or subfields with topics
 * inside when most of their work is in one field), their studies lit on the globe, the people they publish with
 * under it, and lists (most cited, as first author, newest, co-authors) for the whole or the field picked.
 * Over the overview: who they are in numbers (studies, citations, h-index, years, where they have published from).
 */
import { API } from "./jobs.ts";
import { renderOverview } from "./overview.ts";
import { lists, span, TopicError, type CloudWord, type Deps, type More, type TopicMap } from "./topics.ts";

export interface Person {
  id: string; name: string; works: number; citations: number; h: number; years: [number, number] | null;
  topic?: { name: string; subfield: string; field: string } | null;
  institutions?: { id: string; name: string; country?: string | null; kind?: string | null; papers: number; years: [number, number] | null }[];
  per_year?: Record<string, number>; positions?: { first: number; second: number; other: number };
  // institutions
  country?: string | null; kind?: string | null; ror?: string | null; authors?: number;
  cited_years?: [number, number] | null;                       // when the work carrying most citations was done
  records?: { id: string; name: string; works: number; citations: number }[];   // several records seen together
}
/** Another author record with the same name (/authors/{id}/namesakes). */
export interface Namesake {
  id: string; name: string; works: number; citations: number; h: number; years: [number, number] | null;
  topic?: { name: string } | null; institution?: { name: string } | null; top?: { id: string; title: string; year?: number; citations?: number };
}
/** An author page's address: one record, or several seen together ("A1,A2,A3": ?author=A1&with=A2,A3). */
export const authorUrl = (ids: string) => {
  const [first, ...rest] = ids.split(",");
  return `?author=${encodeURIComponent(first)}` + (rest.length ? `&with=${rest.map(encodeURIComponent).join(",")}` : "");
};
const apiRef = (id: string) => {
  const [first, ...rest] = id.split(",");
  return { path: encodeURIComponent(first), with: rest.join(",") };
};
export type Kind = "author" | "institution";
export interface AuthorMap extends TopicMap { kind: Kind; person: Person; level: "fields" | "subfields" }
const PATH: Record<Kind, string> = { author: "authors", institution: "institutions" };
export interface AuthorHit {
  id: string; name: string; works: number; citations?: number; h?: number; years?: [number, number] | null;
  topic?: { name: string; subfield: string; field: string } | null; institution?: { id: string; name: string; country?: string | null };
  country?: string | null; kind?: string | null;   // institutions
  match: "full" | "part";
}

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

export async function fetchAuthor(id: string, kind: Kind = "author", signal?: AbortSignal): Promise<AuthorMap> {
  const a = apiRef(id);
  const r = await fetch(`${API}/api/data/${PATH[kind]}/${a.path}${a.with ? `?with=${encodeURIComponent(a.with)}` : ""}`,
                        { signal: signal ?? AbortSignal.timeout(60000) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new TopicError(r.status, d.detail ?? d.error ?? "");
  return d as AuthorMap;
}

/** A list of the author's page read further (the overview's pager). */
export function moreOfAuthor(id: string, kind: Kind = "author"): More {
  return async (at, lens, offset, limit) => {
    const p = new URLSearchParams({ list: lens, offset: String(offset), limit: String(limit) });
    p.set("read", at.g == null ? "" : at.s == null ? `${at.g}` : `${at.g}.${at.s}`);
    const a = apiRef(id);
    if (a.with) p.set("with", a.with);
    const r = await fetch(`${API}/api/data/${PATH[kind]}/${a.path}?${p}`, { signal: AbortSignal.timeout(30000) });
    if (!r.ok) throw new TopicError(r.status, "");
    return r.json();
  };
}

export async function searchAuthors(q: string, k = 10, signal?: AbortSignal, kind: Kind = "author"): Promise<AuthorHit[]> {
  try {
    const r = await fetch(`${API}/api/data/${PATH[kind]}?q=${encodeURIComponent(q)}&k=${k}`, { signal: signal ?? AbortSignal.timeout(8000) });
    if (!r.ok) return [];
    return ((await r.json()).results ?? []) as AuthorHit[];
  } catch { return []; }
}

export async function namesakesOf(id: string): Promise<Namesake[]> {
  try {
    const r = await fetch(`${API}/api/data/authors/${encodeURIComponent(id)}/namesakes`, { signal: AbortSignal.timeout(15000) });
    if (!r.ok) return [];
    return ((await r.json()).results ?? []) as Namesake[];
  } catch { return []; }
}

/** Who wrote a study, with their ids (the citation index's authorship index; some positions have no person). */
export async function authorsOf(work: string): Promise<{ id: string; name: string; position: number }[]> {
  try {
    const r = await fetch(`${API}/api/data/works/${encodeURIComponent(work)}/authors`, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) return [];
    return ((await r.json()).authors ?? []);
  } catch { return []; }
}

const n = (x: number) => x.toLocaleString();
const plural = (x: number, one: string, many = `${one}s`) => `${n(x)} ${x === 1 ? one : many}`;

const KINDS: Record<string, string> = { education: "university", healthcare: "hospital", company: "company", facility: "research facility",
  government: "government body", nonprofit: "non-profit", funder: "funder", archive: "archive", other: "organisation" };

/** The author's (or institution's) page: the numbers, then the overview of their work. */
export function renderAuthor(box: HTMLElement, d: AuthorMap, deps: Deps) {
  box.replaceChildren();
  const p = d.person;
  const head = box.appendChild(document.createElement("header"));
  head.className = "au-head";
  if (d.kind === "institution") {
    const facts = [plural(p.works, "study", "studies"), p.authors ? plural(p.authors, "author") : "", p.citations ? `cited ${n(p.citations)} times` : "",
      p.h ? `h-index ${p.h}` : "", p.years ? `${p.years[0]}–${p.years[1]}` : ""].filter(Boolean).join(" · ");
    head.innerHTML = `<h1 class="au-name">${esc(p.name)} <span class="au-facts">${esc(facts)}</span></h1>` +
      `<p class="au-where">${[KINDS[p.kind ?? ""] ?? p.kind, p.country].filter(Boolean).map((x) => esc(x!)).join(", ")}` +
      `${p.kind || p.country ? " · " : ""}<span class="au-bars-slot"></span>` +
      `<a href="https://openalex.org/${esc(p.id)}" target="_blank" rel="noopener">OpenAlex ↗</a>` +
      (p.ror ? ` · <a href="${esc(p.ror)}" target="_blank" rel="noopener">ROR ↗</a>` : "") + `</p>`;
    const slot = head.querySelector<HTMLElement>(".au-bars-slot")!;
    const b = bars(p);
    if (b) slot.replaceWith(b, " · "); else slot.remove();
    const info = document.createElement("div");
    const level = d.level === "subfields" ? "subfields, with their topics inside" : "fields, with their subfields inside";
    info.innerHTML = `<p><b>${esc(p.name)}: ${plural(p.works, "study", "studies")}, in ${d.groups.length} ${d.level}.</b> ` +
      `Each box is one of its ${level}: the label OpenAlex gives each study, sized by how many of its studies carry it. ` +
      `Point at a box to see who publishes most there and light those studies on the map; click it for its lists: ` +
      `most cited, newest, its people, and the organisations it works with.</p>` +
      `<p class="quiet">A study counts for an organisation when OpenAlex matched one of its authors' affiliations to it: ` +
      `on recent papers that is most of them, on older ones far fewer, so counts are lower bounds. People are counted by ` +
      `OpenAlex author record; co-authors on papers with more than 100 authors are left out. The dots on the map are ` +
      `the most cited and newest studies, up to 1,200, among those with an English abstract.</p>` +
      (d.coverage.other ? `<p class="quiet">Grey: ${n(d.coverage.other)} studies with no field label, or in fields too small to show.</p>` : "");
    renderOverview(box, d, deps, {
      cloud: (w, k) => peopleCloud(w, k, deps), lists, span, info, act: null, splitFrom: Infinity,
      legend: `<p class="tv-legend">Who publishes most here · bigger: on more of these studies</p>`,
    });
    return;
  }
  const where = (p.institutions ?? []).slice(0, 3).map((i) =>
    `<a href="?institution=${esc(i.id)}" data-go title="${esc(`${i.papers} ${i.papers === 1 ? "study" : "studies"} from there` +
      (i.years ? `, ${i.years[0] === i.years[1] ? i.years[0] : `${i.years[0]}–${i.years[1]}`}` : ""))}">${esc(i.name)}</a>` +
    (i.country ? ` <span class="quiet">(${esc(i.country)})</span>` : "")).join(" · ");
  const yr = (y?: [number, number] | null) => y ? (y[0] === y[1] ? `${y[0]}` : `${y[0]}–${y[1]}`) : "";
  // a historical figure: when the work that carries their citations was done, rather than first-to-last record
  // (which runs to this year's reprints)
  const era = p.cited_years && p.years && (p.cited_years[1] < p.years[1] - 15 || p.cited_years[0] > p.years[0] + 15)
    ? `most cited work ${yr(p.cited_years)}` : yr(p.years);
  const facts = [plural(p.works, "study", "studies"), p.citations ? `cited ${n(p.citations)} times` : "", p.h ? `h-index ${p.h}` : "",
    era].filter(Boolean).join(" · ");
  head.innerHTML = `<h1 class="au-name">${esc(p.name)} <span class="au-facts" title="${esc(p.years ? `Studies listed ${yr(p.years)}` : "")}">${esc(facts)}</span></h1>` +
    `<p class="au-where">${where ? `${where} · ` : ""}${p.topic ? `mostly <b>${esc(p.topic.name)}</b> · ` : ""}` +
    `<span class="au-bars-slot"></span><a href="https://openalex.org/${esc(p.id)}" target="_blank" rel="noopener">OpenAlex ↗</a>` +
    ` · <a href="https://scholar.google.com/scholar?q=${encodeURIComponent(`author:"${p.name}"`)}" target="_blank" rel="noopener">Google Scholar ↗</a></p>`;
  const slot = head.querySelector<HTMLElement>(".au-bars-slot")!;
  const b = bars(p);
  if (b) slot.replaceWith(b, " · "); else slot.remove();
  for (const a of head.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
    a.onclick = (e) => { e.preventDefault(); deps.go(a.getAttribute("href")!); };
  head.append(records(p, deps));

  const info = document.createElement("div");
  const level = d.level === "subfields" ? "subfields, with their topics inside" : "fields, with their subfields inside";
  info.innerHTML = `<p><b>${esc(p.name)}: ${plural(p.works, "study", "studies")}, in ${d.groups.length} ${d.level}.</b> ` +
    `Each box is one of their ${level}: the label OpenAlex gives each study (its main topic, in a subfield, in a field), ` +
    `sized by how many of their studies carry it. Point at a box to see who they work with there and light those studies ` +
    `on the map; click it for its lists.</p>` +
    (p.records ? `<p class="quiet">These are ${p.records.length} OpenAlex author records seen together, as chosen on this page: ` +
      `that they are one person is a reader's judgement, not checked by us.</p>` : "") +
    `<p class="quiet">${p.records ? "Each record" : "This is one OpenAlex author record."} OpenAlex sometimes splits one person into several records, and ` +
    `about a quarter of author names on recent papers are tied to no record at all, so some of their work may be missing ` +
    `here. Citation counts are OpenAlex's. Co-authors on papers with more than 100 authors are not counted. The dots on ` +
    `the map are their studies with an English abstract (the ones we can place).</p>` +
    (d.coverage.other ? `<p class="quiet">Grey: ${n(d.coverage.other)} studies with no field label, or in fields too small to show.</p>` : "");
  for (const a of info.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
    a.onclick = (e) => { e.preventDefault(); deps.go(a.getAttribute("href")!); };
  renderOverview(box, d, deps, {
    cloud: (w, k) => peopleCloud(w, k, deps), lists, span, info, act: null, splitFrom: Infinity,
    legend: `<p class="tv-legend">The people they publish with here · bigger: on more of these studies</p>`,
  });
}

/** Under the name: the records seen together (each can be taken out), and the other records with the same name, to
 *  choose which to see together. OpenAlex splits one person often, historical figures most of all (reprints,
 *  letters, collected works each under a new record); nothing here is merged unless the reader asks. */
function records(p: Person, deps: Deps): HTMLElement {
  const box = document.createElement("div");
  box.className = "au-records";
  const ids = p.id.split(",");
  if (p.records) {
    const line = box.appendChild(document.createElement("p"));
    line.className = "au-together";
    line.append(`${p.records.length} records seen together: `);
    p.records.forEach((r, i) => {
      if (i) line.append(" · ");
      const keep = ids.filter((x) => x !== r.id).join(",");
      const a = line.appendChild(document.createElement("a"));
      a.href = authorUrl(r.id); a.textContent = `${r.id} (${plural(r.works, "study", "studies")})`;
      a.onclick = (e) => { e.preventDefault(); deps.go(a.getAttribute("href")!); };
      const x = line.appendChild(document.createElement("button"));
      x.type = "button"; x.className = "au-x"; x.textContent = "×";
      x.title = `See the page without ${r.id}`; x.setAttribute("aria-label", `Take ${r.id} out`);
      x.onclick = () => deps.go(authorUrl(keep));
    });
  }
  const more = box.appendChild(document.createElement("details"));
  more.className = "au-namesakes";
  more.hidden = true;
  namesakesOf(ids[0]).then((list) => {
    const others = list.filter((x) => !ids.includes(x.id));
    if (!others.length) return;
    more.hidden = false;
    more.innerHTML = `<summary>${others.length === 1 ? "1 other record has" : `${others.length} other records have`} the same name: ` +
      `the same person, or someone else?</summary>` +
      `<p class="quiet small">OpenAlex often splits one person into several records, historical figures most of all ` +
      `(reprints, letters and collected works each under a new one). Tick the ones you recognise by their most cited ` +
      `work, and see them together. Nothing is merged for anyone else.</p><div class="au-ns-list"></div>` +
      `<button type="button" class="au-ns-go" disabled>See them together</button>`;
    const host = more.querySelector<HTMLElement>(".au-ns-list")!, btn = more.querySelector<HTMLButtonElement>(".au-ns-go")!;
    const ticked = new Set<string>();
    for (const o of others) {
      const row = host.appendChild(document.createElement("label"));
      row.className = "au-ns";
      const facts = [plural(o.works, "study", "studies"), o.citations ? `cited ${n(o.citations)} times` : "not cited",
        o.years ? (o.years[0] === o.years[1] ? `${o.years[0]}` : `${o.years[0]}–${o.years[1]}`) : "", o.topic?.name, o.institution?.name]
        .filter(Boolean).join(" · ");
      row.innerHTML = `<input type="checkbox"> <span><b>${esc(o.id)}</b> <span class="quiet">${esc(facts)}</span>` +
        (o.top ? `<br><span class="au-ns-top">Most cited: ${esc(o.top.title)}${o.top.year ? ` (${o.top.year})` : ""}</span>` : "") + `</span>`;
      const box = row.querySelector("input")!;
      box.onchange = () => {
        if (box.checked) ticked.add(o.id); else ticked.delete(o.id);
        btn.disabled = !ticked.size;
        btn.textContent = ticked.size ? `See ${ticked.size + ids.length} records together` : "See them together";
      };
    }
    btn.onclick = () => deps.go(authorUrl([...ids, ...ticked].join(",")));
  });
  return box;
}

/** Studies a year, as a row of small bars (the page's policy allows no inline styles: set from here). */
function bars(p: Person): HTMLElement | null {
  const per = p.per_year ?? {};
  const years = Object.keys(per).map(Number).filter((y) => y > 1900).sort((a, b) => a - b);
  if (years.length < 2) return null;
  const y1 = years[years.length - 1], y0 = Math.max(years[0], y1 - 59);   // the last sixty years at most
  const max = Math.max(...years.filter((y) => y >= y0).map((y) => per[y]));
  const box = document.createElement("span");
  box.className = "au-bars";
  box.setAttribute("role", "img");
  box.setAttribute("aria-label", `Studies a year, ${y0} to ${y1}, at most ${max} in a year`);
  box.title = `Studies a year, ${y0}–${y1}`;
  for (let y = y0; y <= y1; y++) {
    const c = per[y] ?? 0;
    const i = box.appendChild(document.createElement("i"));
    i.style.height = `${Math.max(c ? 2 : 0, Math.round(16 * c / max))}px`;
    i.title = `${y}: ${plural(c, "study", "studies")}`;
  }
  const wrap = document.createElement("span");
  wrap.className = "au-bars-wrap";
  wrap.append(box, ` ${y0}–${y1}`);
  return wrap;
}

/** Co-authors as a cloud, each a link to their page; sized by how many of the unit's studies they are on. */
function peopleCloud(words: CloudWord[], k: number, deps: Deps): HTMLElement {
  const w = words.slice(0, k);
  const lo = Math.min(...w.map((x) => x.share)), hi = Math.max(...w.map((x) => x.share));
  const order: CloudWord[] = [];
  [...w].sort((a, b) => b.share - a.share).forEach((x, i) => (i % 2 ? order.push(x) : order.unshift(x)));
  const div = document.createElement("div");
  div.className = "tg-cloud-box";
  const ul = div.appendChild(document.createElement("ul"));
  ul.className = "tg-cloud au-cloud";
  ul.setAttribute("aria-label", "The people they publish with most here");
  for (const x of order) {
    const t = hi > lo ? Math.sqrt((x.share - lo) / (hi - lo)) : 1;
    const li = ul.appendChild(document.createElement("li"));
    const a = li.appendChild(document.createElement("a"));
    a.href = x.institution ? `?institution=${encodeURIComponent(x.institution)}` : `?author=${encodeURIComponent(x.author ?? "")}`;
    a.textContent = x.t;
    a.onclick = (e) => { e.preventDefault(); deps.go(a.getAttribute("href")!); };
    li.style.fontSize = `${(12.5 + (22 - 12.5) * t).toFixed(1)}px`;
    li.title = `${x.t}: on ${plural(x.score, "study", "studies")} together here`;
  }
  return div;
}

// ---------------------------------------------------------------- finding people
const who = (a: AuthorHit) => [plural(a.works, "study", "studies"), a.citations ? `cited ${n(a.citations)} times` : "",
  a.years ? (a.years[0] === a.years[1] ? `${a.years[0]}` : `${a.years[0]}–${a.years[1]}`) : ""].filter(Boolean).join(" · ");

/** One author (or institution) as a row (the people page, and the strip over search results). */
export function authorRow(a: AuthorHit, go: (url: string) => void, kind: Kind = "author"): HTMLAnchorElement {
  const el = document.createElement("a");
  el.className = "hit au-hit";
  el.href = `?${kind}=${encodeURIComponent(a.id)}`;
  const why = kind === "author" ? [a.topic?.name, a.institution?.name] : [KINDS[a.kind ?? ""] ?? a.kind, a.country];
  el.innerHTML = `<span class="hit-title">${esc(a.name)}</span>` +
    `<span class="hit-meta">${esc(who(a))}</span>` +
    `<span class="hit-why">${why.filter((x): x is string => !!x).map(esc).join(" · ")}</span>`;
  el.onclick = (e) => { e.preventDefault(); go(el.getAttribute("href")!); };
  return el;
}

/** Does a search look like it could be someone's name? Up to four words of letters, no search syntax. */
export const looksLikeName = (q: string, max = 4) => {
  const t = q.trim().split(/\s+/);
  return t.length >= 1 && t.length <= max && t.every((w) => /^[\p{L}][\p{L}'’.&-]*$/u.test(w)) && q.trim().length >= 4;
};

/** Over the results of a search that could be a name: the authors it names (full matches only, the most
 *  published first), or nothing. */
export async function peopleStrip(host: HTMLElement, q: string, go: (url: string) => void, signal?: AbortSignal) {
  if (!looksLikeName(q, 6)) return;
  const words = q.trim().split(/\s+/).length;
  const [people, orgs] = await Promise.all([
    words <= 4 ? searchAuthors(q, 12, signal) : Promise.resolve([]),
    searchAuthors(q, 4, signal, "institution"),
  ]);
  if (signal?.aborted) return;
  // one word (a surname alone): only people with some body of work, so a topic word that is also a rare surname
  // stays quiet; organisations: whole-name matches with a body of work
  const all = people.filter((a) => a.match === "full" && (words > 1 || a.works >= 10));
  // records with one name after the best known one: one line ("and 26 other records named Albert Einstein")
  const nameKey = (a: AuthorHit) => a.name.normalize("NFKD").toLowerCase().replace(/[^\p{L}]/gu, "");
  const twins = new Map<string, number>();
  const hits: AuthorHit[] = [];
  for (const a of all) {
    const k = nameKey(a);
    if (twins.has(k)) twins.set(k, twins.get(k)! + 1); else { twins.set(k, 0); hits.push(a); }
  }
  const show = hits.slice(0, 3);
  const orgShow = orgs.filter((o) => o.match === "full" && o.works >= 100).slice(0, 2);
  if (!show.length && !orgShow.length) return;
  const strip = document.createElement("section");
  strip.className = "au-strip";
  strip.setAttribute("aria-label", "People and organisations with this name");
  const orgs_ = () => {
    strip.insertAdjacentHTML("beforeend", `<h2 class="au-strip-h">Organisations named ${esc(q.trim())}</h2>`);
    for (const o of orgShow) strip.append(authorRow(o, go, "institution"));
  };
  // a search that is exactly someone's name puts the people first ("Albert Einstein" before the college)
  const exact = show.some((a) => nameKey(a) === q.normalize("NFKD").toLowerCase().replace(/[^\p{L}]/gu, ""));
  if (orgShow.length && !exact) orgs_();
  if (show.length) {
    strip.insertAdjacentHTML("beforeend", `<h2 class="au-strip-h">People named ${esc(q.trim())}</h2>`);
    for (const a of show) {
      const row = authorRow(a, go);
      const t = twins.get(nameKey(a)) ?? 0;
      if (t) row.insertAdjacentHTML("beforeend", `<span class="hit-why au-twin">and ${t}${t >= 5 ? "+" : ""} other ${t === 1 ? "record" : "records"} ` +
        `with this name: the page lets you see them together</span>`);
      strip.append(row);
    }
    if (hits.length > show.length || [...twins.values()].some((x) => x > 0)) {
      const more = strip.appendChild(document.createElement("a"));
      more.className = "au-more";
      more.href = `?people=${encodeURIComponent(q.trim())}`;
      more.textContent = "More people with this name →";
      more.onclick = (e) => { e.preventDefault(); go(more.getAttribute("href")!); };
    }
  }
  if (orgShow.length && exact) orgs_();
  host.prepend(strip);
}

/** The people page: everyone matching a name. */
export async function renderPeople(box: HTMLElement, q: string, go: (url: string) => void, signal?: AbortSignal) {
  box.innerHTML = `<h1 class="topics-title">People named ${esc(q)}</h1><p class="checking">Looking them up…</p>`;
  const hits = await searchAuthors(q, 50, signal);
  if (signal?.aborted) return;
  box.innerHTML = `<h1 class="topics-title">People named ${esc(q)}</h1>`;
  if (!hits.length) {
    box.insertAdjacentHTML("beforeend", `<p class="quiet">No author by that name. Try a surname alone, or fewer accents and initials.</p>`);
    return;
  }
  const full = hits.filter((a) => a.match === "full"), part = hits.filter((a) => a.match !== "full");
  box.insertAdjacentHTML("beforeend", `<p class="topics-intro">${full.length ? `${n(full.length)} author ${full.length === 1 ? "record" : "records"} ` +
    `with every word of the name, the most published first.` : "No record has every word of the name; these have some of them."} ` +
    `OpenAlex sometimes keeps one person under several records: look at the fields and years to tell them apart.</p>`);
  const list = box.appendChild(document.createElement("div"));
  // the same name, institution and topic twice: very likely one person under two OpenAlex records (said, not merged)
  const seen = new Map<string, AuthorHit>();
  for (const a of full) {
    const row = authorRow(a, go);
    const key = [a.name.toLowerCase(), a.institution?.id, a.topic?.subfield].join("|");
    const twin = a.institution && a.topic ? seen.get(key) : undefined;
    if (twin) row.insertAdjacentHTML("beforeend", `<span class="hit-why au-twin">Same name, institution and field as ${esc(twin.name)} above: probably the same person under two records.</span>`);
    else seen.set(key, a);
    list.append(row);
  }
  if (part.length) {
    if (full.length) box.insertAdjacentHTML("beforeend", `<h2 class="au-strip-h">Part of the name</h2>`);
    const l2 = box.appendChild(document.createElement("div"));
    for (const a of part) l2.append(authorRow(a, go));
  }
}
