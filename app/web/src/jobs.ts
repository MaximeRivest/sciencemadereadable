/**
 * Our models run on our GPU (a worker anywhere): the page puts a job in the queue and follows it.
 * The queue's address comes from config.js (empty: this site's own server).
 */
declare global { interface Window { SRL_CONFIG?: { api?: string; support?: { github?: string; card?: string } } } }
export const API = (window.SRL_CONFIG?.api ?? "").replace(/\/$/, "");

export interface Status { worker_online: boolean; models: string[]; queued: number; running: number }

export async function status(): Promise<Status | null> {
  try { return await (await fetch(`${API}/api/status`)).json(); } catch { return null; }
}

/** The file name the server and the site use for a paper's saved rewrites. */
export async function key(doi: string): Promise<string> {
  const h = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(doi.toLowerCase()));
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 24);
}

/** Saved rewrites: from the queue server, or (server offline) the copy published with the site. */
export async function savedRewrites(doi: string): Promise<Record<string, any>> {
  try {
    const r = await fetch(`${API}/api/rewrites?doi=${encodeURIComponent(doi)}`);
    if (r.ok) return await r.json();
  } catch { /* offline: the site's copy */ }
  try {
    const r = await fetch(`examples/${await key(doi)}.json`);
    return r.ok ? await r.json() : {};
  } catch { return {}; }
}

export async function examples(): Promise<{ doi: string; title: string; plain_title?: string }[]> {
  for (const url of [`${API}/api/examples`, "examples/index.json"]) {
    try { const r = await fetch(url); if (r.ok) return await r.json(); } catch { /* next */ }
  }
  return [];
}

export interface JobView {
  status: "queued" | "running" | "done" | "failed"; ahead: number; parts: Record<string, string>;
  done: string[]; error?: string; seconds?: number; worker_online: boolean; running_others?: number;
}

/** A rewrite of this paper by this model already in line or running (someone asked before). */
export async function activeJob(doi: string, model: string): Promise<string | null> {
  try {
    const d = await (await fetch(`${API}/api/jobs?${new URLSearchParams({ doi, model })}`)).json();
    return d.id ?? null;
  } catch { return null; }
}

/** Ask for a rewrite and follow it until it ends. `saved: true` means one already exists. */
export async function follow(doi: string, model: string, onView: (v: JobView) => void, signal: AbortSignal,
                             existing?: string): Promise<{ saved: true } | JobView> {
  let d: any = { id: existing };
  if (!existing) {
    const r = await fetch(`${API}/api/jobs`, { method: "POST", headers: { "Content-Type": "application/json" },
                                              body: JSON.stringify({ doi, model }), signal });
    d = await r.json();
    if (!r.ok) throw new Error(d.error ?? `HTTP ${r.status}`);
    if (d.saved) return { saved: true };
  }
  for (;;) {
    await new Promise((res) => setTimeout(res, 900));
    if (signal.aborted) throw new DOMException("stopped", "AbortError");
    const v: JobView = await (await fetch(`${API}/api/jobs/${d.id}`, { signal })).json();
    if ((v as any).error && !v.status) throw new Error((v as any).error);
    onView(v);
    if (v.status === "done") return v;
    if (v.status === "failed") throw new Error(v.error ?? "our GPU couldn't finish this paper");
  }
}

export interface LibraryItem { doi: string; title: string; plain_title?: string; journal?: string; year?: string; models: string[]; example?: boolean; added?: number }
export interface NowItem { doi: string; model: string; status: string; title?: string; plain_title?: string; progress: number; since: number }

/** Every saved rewrite: from the queue server, or (offline) the copy published with the site. */
export async function library(): Promise<LibraryItem[]> {
  for (const url of [`${API}/api/library`, "examples/library.json"]) {
    try { const r = await fetch(url); if (r.ok) return await r.json(); } catch { /* next */ }
  }
  return [];
}

export async function now(): Promise<{ live: NowItem[]; recent: (NowItem & { seconds: number })[] } | null> {
  try { const r = await fetch(`${API}/api/now`); return r.ok ? await r.json() : null; } catch { return null; }
}
