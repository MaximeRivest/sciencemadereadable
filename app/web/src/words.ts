/**
 * A word of a word cloud, clicked: what it says about the group, the group's studies that use it, and two ways on:
 * map the word within the topic (its own research groups), or find it across all of science.
 *
 * Word clouds are best for getting an impression of a set of documents and for browsing, weaker for finding one
 * thing (Rivadeneira et al. 2007, doi:10.1145/1240624.1240775); readers expect a clicked tag to narrow to the items
 * that carry it, as tag clouds have done since Flickr and Delicious (Sinclair & Cardew-Hall 2008,
 * doi:10.1177/0165551506078083). So a click first narrows (the group's studies with that word), then offers to widen.
 */
import { API } from "./jobs.ts";

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
const plain = (s?: string | null) => (s ?? "").replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();
const pct = (x: number) => x >= 0.095 ? `${Math.round(100 * x)}%` : x >= 0.0095 ? `${(100 * x).toFixed(1)}%` : `${Number((100 * x).toPrecision(1))}%`;

export interface WordAnswer {
  word: string; total: number; scanned: number; matched: number;
  papers: { id: string; title: string; year: number | null; citations: number; first_author?: string; authors_count?: number }[];
}
export interface WordInfo {
  word: string; share: number; lift: number;
  vs: string;                      // what the lift compares with ("the other groups", "all of science")
  unit: string;                    // "this group", "subgroup 3.2", "this field"
  topic: string;                   // the map's topic, to map the word within it
  fetch: () => Promise<WordAnswer | null>;
}

let dlg: HTMLDialogElement | null = null;

export function openWord(info: WordInfo, go: (url: string) => void) {
  dlg?.remove();
  dlg = document.createElement("dialog");
  dlg.className = "wcard";
  dlg.setAttribute("aria-label", `The word “${info.word}”`);
  const more = info.lift >= 1.25 ? `${info.lift >= 10 ? Math.round(info.lift) : info.lift.toFixed(1)} times as often as in ${info.vs}`
    : info.lift <= 0.8 ? `less often than in ${info.vs}` : `about as often as in ${info.vs}`;
  const within = `${info.word} in ${info.topic}`;
  dlg.innerHTML = `<div class="wcard-in"><button type="button" class="wcard-x" aria-label="Close">×</button>
    <p class="wcard-kicker">A word in ${esc(info.unit)}</p>
    <h2 class="wcard-word">“${esc(info.word)}”</h2>
    <p class="wcard-fact">Used by <b>${pct(info.share)}</b> of its studies: ${esc(more)}.</p>
    <div class="wcard-list" aria-live="polite"><p class="loading-line">Finding the studies that use it…</p></div>
    <div class="wcard-go">
      <a class="btn" href="?topics=${encodeURIComponent(within)}" data-go title="The research groups of the studies closest in meaning to “${esc(within)}”">Map “${esc(info.word)}” within ${esc(info.topic)} →</a>
      <a class="btn ghost" href="?q=${encodeURIComponent(`"${info.word}"`)}" data-go>Find it across all of science →</a>
    </div></div>`;
  document.body.appendChild(dlg);
  const d = dlg;
  const close = () => { d.close(); d.remove(); if (dlg === d) dlg = null; };
  d.querySelector<HTMLButtonElement>(".wcard-x")!.onclick = close;
  d.addEventListener("click", (e) => { if (e.target === d) close(); });    // the backdrop
  d.addEventListener("close", () => d.remove());
  for (const a of d.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
    a.onclick = (e) => { if (e.metaKey || e.ctrlKey) return; e.preventDefault(); close(); go(a.getAttribute("href")!); };
  d.showModal();
  info.fetch().then((r) => {
    const list = d.querySelector<HTMLElement>(".wcard-list")!;
    if (!r) { list.innerHTML = `<p class="quiet">The studies didn't come back. Try again in a moment.</p>`; return; }
    if (!r.papers.length) { list.innerHTML = `<p class="quiet">None of its most central studies use it.</p>`; return; }
    const note = r.scanned < r.total
      ? `Found in ${r.matched.toLocaleString()} of its ${r.scanned.toLocaleString()} most central studies. The most central first:`
      : `${r.matched.toLocaleString()} of its ${r.total.toLocaleString()} studies use it. The most central first:`;
    list.innerHTML = `<p class="wcard-note">${esc(note)}</p><ol>${r.papers.slice(0, 8).map((p) => {
      const by = p.first_author ? (p.authors_count && p.authors_count > 1 ? `${p.first_author} et al.` : p.first_author) : "";
      const meta = [by, p.year, p.citations ? (p.citations === 1 ? "cited once" : `cited ${p.citations.toLocaleString()} times`) : ""].filter(Boolean).join(" · ");
      return `<li><a href="?work=${esc(p.id)}" data-go><span class="wcard-t">${esc(plain(p.title) || "(no title)")}</span>` +
        `<span class="wcard-m">${esc(String(meta))}</span></a></li>`;
    }).join("")}</ol>`;
    for (const a of list.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
      a.onclick = (e) => { if (e.metaKey || e.ctrlKey) return; e.preventDefault(); close(); go(a.getAttribute("href")!); };
  });
}

/** The data API's answer for a word of one unit of a topic map (params: the map's own, plus read/group). */
export async function fetchWord(params: URLSearchParams): Promise<WordAnswer | null> {
  try {
    const r = await fetch(`${API}/api/data/topics?${params}`, { signal: AbortSignal.timeout(30000) });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}
