/**
 * Usage counts, sent to our own server (stats.py): no cookies, nothing stored in the browser.
 * Browsers asking not to be tracked (Do Not Track, Global Privacy Control) send nothing.
 * Fire and forget: a count never slows the page or shows an error.
 *
 * Each page shown carries the parts of its address that say what is looked at (the search, the topic,
 * the author, the paper), so a visit can be read as a path; leaving a page says how long it was open and
 * whether something was still loading or being written then (where people give up).
 */
import { API } from "./jobs.ts";

const off = navigator.doNotTrack === "1" || (navigator as any).globalPrivacyControl === true;
let firstView = true;
let shownAt = performance.now();

/** The address parameters worth counting. Anything else (sign-in codes, private job keys...) is never sent. */
const KEEP = ["q", "topics", "group", "scope", "definition", "by", "since", "readable", "at", "author", "with",
  "institution", "people", "work", "paper", "doi", "model", "ask", "map", "api", "library", "fields", "reading", "account",
  "support"];

function address(): string {
  const p = new URLSearchParams(location.search), out = new URLSearchParams();
  for (const k of KEEP) if (p.has(k)) out.set(k, p.get(k)!);
  return out.toString();
}

export function track(t: string, data: Record<string, string | number | undefined> = {}) {
  if (off) return;
  const e: Record<string, unknown> = { t, ...data };
  if (t === "view") {
    shownAt = performance.now();
    const u = address();
    if (u) e.u = u;
    if (firstView) {
      firstView = false;
      try {
        const ref = document.referrer ? new URL(document.referrer).hostname : "";
        if (ref && ref !== location.hostname) e.ref = ref;
      } catch { /* no referrer */ }
      e.w = Math.round(window.innerWidth / 100) * 100;
    }
  }
  // text/plain: a simple request, allowed to another origin without a preflight
  const body = new Blob([JSON.stringify(e)], { type: "text/plain" });
  try { if (navigator.sendBeacon?.(`${API}/api/event`, body)) return; } catch { /* fall back */ }
  fetch(`${API}/api/event`, { method: "POST", body, keepalive: true }).catch(() => {});
}

/** What was still under way when the page was left: a rewrite being written, a map being made, a page loading. */
function busy(): string | undefined {
  if (document.getElementById("go")?.textContent === "Stop") return "rewrite";
  if (document.querySelector("#results .topics-title")?.textContent === "Mapping the whole field") return "field-map";
  if (document.querySelector('[aria-busy="true"]')) return "loading";
  return undefined;
}

// "hidden" is the last moment a page can be counted reliably (phones rarely send unload events); a tab put in the
// background and brought back counts twice, which the reports allow for (the last one of a visit is its end).
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "hidden") return;
  track("hide", { view: document.body.dataset.view, s: Math.round((performance.now() - shownAt) / 1000), busy: busy() });
});
