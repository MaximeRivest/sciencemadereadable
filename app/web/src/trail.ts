/**
 * The path walked so far: studies, people, organisations and topics, in the order they were opened, shown at the
 * top of a study's page; and the studies already opened, marked in lists. Readers wandering through linked pages
 * lose track of where they are and how they got there (Otter & Johnson 2000, doi:10.1016/s0953-5438(00)00030-8);
 * a visible trail and "seen" marks answer both.
 *
 * Kept in this tab only (sessionStorage): it is never sent anywhere, and it is gone when the tab closes.
 */
export type StopKind = "work" | "author" | "institution" | "source" | "topic" | "search" | "word";
export interface Stop { kind: StopKind; label: string; url: string }

const KEY = "smr:trail", SEEN = "smr:seen", MAX = 30;
const read = <T>(k: string, d: T): T => { try { return JSON.parse(sessionStorage.getItem(k) ?? "") as T; } catch { return d; } };
const write = (k: string, v: unknown) => { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode, full */ } };

/** Add a stop (the same page twice in a row counts once; coming back to a stop shortens the path to it). */
export function remember(stop: Stop) {
  if (!stop.label) return;
  let t = read<Stop[]>(KEY, []);
  const at = t.findIndex((s) => s.url === stop.url);
  if (at >= 0) t = t.slice(0, at);                  // back to an earlier stop: the path is up to there again
  t.push({ ...stop, label: stop.label.slice(0, 140) });
  write(KEY, t.slice(-MAX));
  if (stop.kind === "work") {
    const m = new URLSearchParams(stop.url.replace(/^.*\?/, "")).get("work");
    if (m) { const s = new Set(read<string[]>(SEEN, [])); s.add(m); write(SEEN, [...s].slice(-500)); }
  }
}

export const seen = (id: string) => read<string[]>(SEEN, []).includes(id);

const NAME: Record<StopKind, string> = { work: "study", author: "person", institution: "organisation", source: "journal", topic: "topic", search: "search", word: "word" };

/** The last few stops before this page, as links (nothing when this is the first page of the walk). */
export function renderTrail(host: HTMLElement, go: (url: string) => void, here = location.search, keep = 4) {
  const t = read<Stop[]>(KEY, []);
  const before = (t.length && t[t.length - 1].url === here ? t.slice(0, -1) : t).slice(-keep);
  if (!before.length) { host.hidden = true; return; }
  host.hidden = false;
  host.replaceChildren();
  const label = document.createElement("span");
  label.className = "trail-label";
  label.textContent = "Your path";
  host.append(label);
  const ol = document.createElement("ol");
  for (const s of before) {
    const li = ol.appendChild(document.createElement("li"));
    const a = li.appendChild(document.createElement("a"));
    a.href = s.url;
    a.title = `Back to this ${NAME[s.kind]}: ${s.label}`;
    a.dataset.kind = s.kind;
    a.append(document.createTextNode(s.label.length > 46 ? `${s.label.slice(0, 44).trimEnd()}…` : s.label));
    a.onclick = (e) => { if (e.metaKey || e.ctrlKey || e.shiftKey) return; e.preventDefault(); go(s.url); };
  }
  const li = ol.appendChild(document.createElement("li"));
  li.className = "trail-here";
  li.setAttribute("aria-current", "page");
  li.textContent = "this page";
  host.append(ol);
}
