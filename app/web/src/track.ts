/**
 * Usage counts, sent to our own server (stats.py): no cookies, nothing stored in the browser,
 * no search words. Browsers asking not to be tracked (Do Not Track, Global Privacy Control)
 * send nothing. Fire and forget: a count never slows the page or shows an error.
 */
import { API } from "./jobs.ts";

const off = navigator.doNotTrack === "1" || (navigator as any).globalPrivacyControl === true;
let firstView = true;

export function track(t: string, data: Record<string, string | number | undefined> = {}) {
  if (off) return;
  const e: Record<string, unknown> = { t, ...data };
  if (t === "view" && firstView) {
    firstView = false;
    try {
      const ref = document.referrer ? new URL(document.referrer).hostname : "";
      if (ref && ref !== location.hostname) e.ref = ref;
    } catch { /* no referrer */ }
    e.w = Math.round(window.innerWidth / 100) * 100;
  }
  // text/plain: a simple request, allowed to another origin without a preflight
  const body = new Blob([JSON.stringify(e)], { type: "text/plain" });
  try { if (navigator.sendBeacon?.(`${API}/api/event`, body)) return; } catch { /* fall back */ }
  fetch(`${API}/api/event`, { method: "POST", body, keepalive: true }).catch(() => {});
}
