/**
 * Finding and opening papers. Search: our own semantic index (the queue server's /api/search), Europe PMC
 * when ours doesn't answer. Metadata and the full text (JATS XML): Europe PMC, from the browser; figure
 * images: PubMed Central's open-access copy on AWS.
 */
import { API } from "./jobs.ts";
import { layout, licenseOk, sectionNodes, splitSections, type Mark } from "./jats.ts";

const EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest";
const IMAGES = "https://pmc-oa-opendata.s3.amazonaws.com";

export interface Hit { pmcid: string; doi?: string; title: string; authors?: string; journal?: string; year?: string }

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
export const lastSearch: { by: "semantic" | "keyword" | "europepmc"; matches: number | null } = { by: "semantic", matches: null };

/** Open-access, CC BY research papers with full text: the ones we can rewrite and show. Ours first
 *  (by meaning; by exact words when the query has quotes, AND / OR / NOT, brackets or word*),
 *  Europe PMC's keyword search if ours fails or is slow. */
export async function search(q: string, signal?: AbortSignal): Promise<Hit[]> {
  try {
    const r = await fetch(`${API}/api/search?${new URLSearchParams({ q })}`,
      { signal: AbortSignal.any([AbortSignal.timeout(7000), ...(signal ? [signal] : [])]) });
    if (r.ok) {
      const d = await r.json();
      lastSearch.by = d.mode === "keyword" ? "keyword" : "semantic";
      lastSearch.matches = d.matches ?? null;
      return d.hits ?? [];
    }
  } catch (e) {
    if (signal?.aborted) throw e;
  }
  lastSearch.by = "europepmc";
  lastSearch.matches = null;
  return searchEuropePMC(q, signal);
}

async function searchEuropePMC(q: string, signal?: AbortSignal): Promise<Hit[]> {
  const query = `(${q}) AND OPEN_ACCESS:y AND HAS_FT:y AND LICENSE:"cc by" AND PUB_TYPE:"research-article"`;
  const url = `${EPMC}/search?${new URLSearchParams({ query, format: "json", resultType: "lite", pageSize: "15" })}`;
  const d = await (await fetch(url, { signal })).json();
  return (d.resultList?.result ?? []).filter((r: any) => r.pmcid).map((r: any) => ({
    pmcid: r.pmcid, doi: r.doi, title: clean(r.title).replace(/\.$/, ""), authors: r.authorString,
    journal: r.journalTitle, year: r.pubYear,
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
