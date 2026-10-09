/**
 * Finding and opening papers. Search: our own semantic index (the queue server's /api/search), Europe PMC
 * when ours doesn't answer. Metadata and the full text (JATS XML): Europe PMC, from the browser; figure
 * images: PubMed Central's open-access copy on AWS.
 */
import { API } from "./jobs.ts";
import { layout, licenseOk, sectionNodes, splitSections, type Mark } from "./jats.ts";

const EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest";
const IMAGES = "https://pmc-oa-opendata.s3.amazonaws.com";

export interface Hit {
  pmcid: string; doi?: string; title: string; authors?: string; journal?: string; year?: string;
  id?: string;           // OpenAlex W-id (all-of-science search)
  readable?: boolean;    // we can rewrite and show it (PMC full text, CC BY research article)
  x?: number; y?: number; // on the map of science
  type?: string; citations?: number;
}

export interface Paper {
  doi: string; pmcid: string; title: string; authors: string; journal: string; year: string; license: string;
  url: string; sections: Record<string, string>; marks: Record<string, Mark[]>;
}

export class PaperError extends Error {}

/** Europe PMC titles carry markup, sometimes escaped (&lt;i&gt;): read it as HTML, keep the text. */
const clean = (s?: string) => {
  const once = new DOMParser().parseFromString(s ?? "", "text/html").body.textContent ?? "";
  return (new DOMParser().parseFromString(once, "text/html").body.textContent ?? "").replace(/\s+/g, " ").trim();
};

/** How the last search was answered: our index by meaning or by exact words, or Europe PMC. */
export const lastSearch: { by: "semantic" | "keyword" | "europepmc"; matches: number | null; scope: "all" | "readable";
  map: { x: number; y: number } | null } = { by: "semantic", matches: null, scope: "all", map: null };

/** Open-access, CC BY research papers with full text: the ones we can rewrite and show. Ours first
 *  (by meaning; by exact words when the query has quotes, AND / OR / NOT, brackets or word*),
 *  Europe PMC's keyword search if ours fails or is slow. */
export interface SearchOptions { by?: "meaning" | "words" | ""; since?: string; readable?: boolean }

const authorsLine = (s?: string, n = 0) => {
  const names = (s ?? "").split(" | ").filter(Boolean);
  return names.slice(0, 6).join(", ") + (n > 6 ? ", et al." : "");
};
const plainText = (s?: string) => (s ?? "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();

/** All of science (our data API through the queue server): every embedded study, the readable ones marked,
 *  each placed on the map. Throws when the data API can't be reached. */
async function searchAll(q: string, opts: SearchOptions, signal?: AbortSignal): Promise<Hit[]> {
  const params = new URLSearchParams({ q, k: "20", map: "true", abstracts: "false",
    mode: opts.by === "words" ? "keyword" : opts.by === "meaning" ? "semantic" : "auto" });
  if (opts.since) params.set("year_from", opts.since);
  if (opts.readable) params.set("collection", "smr");
  const r = await fetch(`${API}/api/data/search?${params}`,
    { signal: AbortSignal.any([AbortSignal.timeout(9000), ...(signal ? [signal] : [])]) });
  if (!r.ok) throw new Error(`data API ${r.status}`);
  const d = await r.json();
  lastSearch.by = d.mode === "keyword" ? "keyword" : "semantic";
  lastSearch.matches = d.matches ?? null;
  lastSearch.scope = opts.readable ? "readable" : "all";
  lastSearch.map = d.map ?? null;
  return (d.results ?? []).map((x: any) => ({
    id: x.id, pmcid: x.pmcid ?? "", doi: x.doi ?? undefined, title: plainText(x.title).replace(/\.$/, "") || "(no title)",
    authors: authorsLine(x.authors, x.authors_count), journal: x.venue ?? undefined, year: x.year ? String(x.year) : undefined,
    readable: Boolean(x.smr), x: x.x, y: x.y, type: x.type, citations: x.citations,
  }));
}

/** One study's details (any study we hold, readable or not). */
export async function work(ref: string): Promise<any> {
  const r = await fetch(`${API}/api/data/works/${encodeURIComponent(ref)}`, { signal: AbortSignal.timeout(9000) });
  if (r.status === 404) throw new PaperError("We don't have this study (yet).");
  if (!r.ok) throw new PaperError("Our search is busy right now. Try again in a moment.");
  const d = await r.json();
  return { ...d, title: plainText(d.title), abstract: plainText(d.abstract), authorsLine: authorsLine(d.authors, d.authors_count) };
}

/** Studies closest in meaning to one study. */
export async function similar(ref: string, k = 6): Promise<Hit[]> {
  try {
    const r = await fetch(`${API}/api/data/similar/${encodeURIComponent(ref)}?k=${k}&abstracts=false`, { signal: AbortSignal.timeout(9000) });
    if (!r.ok) return [];
    return ((await r.json()).results ?? []).map((x: any) => ({
      id: x.id, pmcid: x.pmcid ?? "", doi: x.doi ?? undefined, title: plainText(x.title) || "(no title)",
      journal: x.venue ?? undefined, year: x.year ? String(x.year) : undefined, readable: Boolean(x.smr),
    }));
  } catch { return []; }
}

/** Where studies sit on the map (by DOI, PMCID or W-id). */
export async function place(refs: string[]): Promise<Record<string, { x: number; y: number }>> {
  try {
    const r = await fetch(`${API}/api/data/map/place?ids=${encodeURIComponent(refs.join(","))}`, { signal: AbortSignal.timeout(6000) });
    return r.ok ? (await r.json()).papers ?? {} : {};
  } catch { return {}; }
}

/** Did the reader type search syntax (quotes, AND/OR/NOT, brackets, word*, -word, title:)? Same rule as the server. */
export const looksExact = (q: string) => /"|\b(AND|OR|NOT)\b|[()*]|(^|\s)[-+]\w|\b(title|abstract|keywords):/.test(q);

export async function search(q: string, opts: SearchOptions = {}, signal?: AbortSignal): Promise<Hit[]> {
  try {
    return await searchAll(q, opts, signal);
  } catch (e) {
    if (signal?.aborted) throw e;
  }
  return searchReadable(q, opts, signal);
}

async function searchReadable(q: string, opts: SearchOptions = {}, signal?: AbortSignal): Promise<Hit[]> {
  lastSearch.scope = "readable";
  lastSearch.map = null;
  const params = new URLSearchParams({ q, mode: opts.by === "words" ? "keyword" : opts.by === "meaning" ? "semantic" : "auto" });
  if (opts.since) params.set("year_from", opts.since);
  try {
    const r = await fetch(`${API}/api/search?${params}`,
      { signal: AbortSignal.any([AbortSignal.timeout(7000), ...(signal ? [signal] : [])]) });
    if (r.ok) {
      const d = await r.json();
      lastSearch.by = d.mode === "keyword" ? "keyword" : "semantic";
      lastSearch.matches = d.matches ?? null;
      return (d.hits ?? []).map((h: Hit) => ({ ...h, readable: true }));
    }
  } catch (e) {
    if (signal?.aborted) throw e;
  }
  lastSearch.by = "europepmc";
  lastSearch.matches = null;
  return searchEuropePMC(q, opts.since, signal);
}

async function searchEuropePMC(q: string, since?: string, signal?: AbortSignal): Promise<Hit[]> {
  const years = since ? ` AND PUB_YEAR:[${since} TO 3000]` : "";
  const query = `(${q}) AND OPEN_ACCESS:y AND HAS_FT:y AND LICENSE:"cc by" AND PUB_TYPE:"research-article"${years}`;
  const url = `${EPMC}/search?${new URLSearchParams({ query, format: "json", resultType: "lite", pageSize: "15" })}`;
  const d = await (await fetch(url, { signal })).json();
  return (d.resultList?.result ?? []).filter((r: any) => r.pmcid).map((r: any) => ({
    pmcid: r.pmcid, doi: r.doi, title: clean(r.title).replace(/\.$/, ""), authors: r.authorString,
    journal: r.journalTitle, year: r.pubYear, readable: true,
  }));
}

/** A DOI, a PMCID, or a doi.org link → the paper's record. */
async function lookup(id: string): Promise<any> {
  const raw = id.trim().replace(/^(https?:\/\/)?(dx\.)?doi\.org\//i, "").replace(/^doi:\s*/i, "");
  const query = /^pmc\d+$/i.test(raw) ? `PMCID:${raw.toUpperCase()}` : /^10\.\d{4,9}\//.test(raw) ? `DOI:"${raw}"` : null;
  if (!query) throw new PaperError("That isn't a DOI or a PubMed Central ID. Try searching by topic instead.");
  const d = await (await fetch(`${EPMC}/search?${new URLSearchParams({ query, format: "json", resultType: "core" })}`)).json();
  const r = (d.resultList?.result ?? []).find((x: any) => x.pmcid);
  if (!r) throw new PaperError("We couldn't find an open copy of this paper in PubMed Central.");
  return r;
}

export async function open(id: string): Promise<Paper> {
  const r = await lookup(id);
  if (r.isOpenAccess !== "Y") throw new PaperError("This paper's full text isn't openly available, so we can't rewrite it.");
  const res = await fetch(`${EPMC}/${r.pmcid}/fullTextXML`);
  if (!res.ok) throw new PaperError("Europe PMC has no full text for this paper.");
  const doc = new DOMParser().parseFromString(await res.text(), "text/xml");
  const root = doc.documentElement;
  if (!licenseOk(root)) throw new PaperError(
    "This paper isn't under a CC BY licence. A full rewrite republishes the paper, which only CC BY clearly allows.");
  const sections = splitSections(root);
  if (!sections) throw new PaperError(
    "This paper isn't laid out as a standard research article (introduction, methods, results), which is what our models learned.");
  const nodes = sectionNodes(root);
  const marks: Record<string, Mark[]> = {};
  for (const [k, v] of Object.entries(sections)) if (v) marks[k] = layout(nodes[k] as any, v.split("\n\n"));
  const doi = (r.doi ?? "").toLowerCase();
  return {
    doi, pmcid: r.pmcid, title: clean(r.title).replace(/\.$/, ""), authors: r.authorString ?? "",
    journal: r.journalInfo?.journal?.title ?? r.journalTitle ?? "", year: r.pubYear ?? "", license: "CC BY",
    url: doi ? `https://doi.org/${doi}` : `https://europepmc.org/article/PMC/${r.pmcid}`, sections, marks,
  };
}

/** An <img> for a figure file; tries the next stored version of the article if the first is missing. */
export function image(pmcid: string, href: string, alt: string): HTMLImageElement {
  const file = /\.[a-z0-9]{2,4}$/i.test(href) ? href : `${href}.jpg`;
  const img = document.createElement("img");
  img.alt = alt;
  img.loading = "lazy";
  img.referrerPolicy = "no-referrer";
  let version = 1;
  img.onerror = () => { if (++version <= 4) img.src = `${IMAGES}/${pmcid}.${version}/${file}`; else img.remove(); };
  img.src = `${IMAGES}/${pmcid}.${version}/${file}`;
  return img;
}
