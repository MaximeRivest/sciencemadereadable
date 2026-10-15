/**
 * Whole-field maps: the line they wait in, the maps already made (public, searchable), and paying to make one now.
 * The data API's /fields and /fields/queue (search/srl_search/field.py); payments: app/fieldpay.py.
 *
 * The offer, and why it is shaped this way: a reader who asked for a map sees when the free line gets to it. Only
 * when paying saves at least an hour (and the GPU isn't taken by training, when paying couldn't help) is the offer
 * shown, with the two times side by side, one price, and what it changes for others: the map is public anyway, the
 * payer's name can go on it, and money comes back if it can't be made. No countdowns, no "only N left".
 */
import { API } from "./jobs.ts";

export interface QueueRow {
  key: string; topic: string; definition?: string | null; line: "paid" | "free" | "rebuild"; kind?: string;
  asked?: string; position?: number; start_s?: number; ready_s?: number; credit?: string;
  stage?: string; stage_index?: number; stages?: number;
}
export interface FieldQueue {
  running: QueueRow | null; waiting: QueueRow[]; paid_ready_s: number; gpu_reserved: boolean;
  free_per_day: number; made_today: number;
}
export interface FieldMade {
  key: string; topic: string; definition?: string | null; papers?: number; groups?: number; made?: string;
  credit?: string | null; score?: number;
}
export interface FieldOffer { url: string; amount: number; currency: string; mode?: string }

export async function fieldQueue(): Promise<FieldQueue | null> {
  try { const r = await fetch(`${API}/api/data/fields/queue`); return r.ok ? await r.json() : null; } catch { return null; }
}
export async function fieldsMade(q = "", limit = 200): Promise<FieldMade[]> {
  const p = new URLSearchParams({ limit: String(limit) });
  if (q) p.set("q", q);
  try { const r = await fetch(`${API}/api/data/fields?${p}`); return r.ok ? (await r.json()).maps : []; } catch { return []; }
}
export async function verifyPaid(session: string): Promise<{ ok: boolean; status: number; d: any }> {
  try {
    const r = await fetch(`${API}/api/fields/paid?session=${encodeURIComponent(session)}`);
    return { ok: r.ok, status: r.status, d: await r.json().catch(() => ({})) };
  } catch { return { ok: false, status: 0, d: {} }; }
}

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
export const dollars = (cents: number) => `$${(cents / 100).toFixed(cents % 100 ? 2 : 0)}`;

/** "about 20 minutes", "about 9 hours", "about 3 days" */
export function inTime(s: number): string {
  if (s < 90 * 60) { const m = Math.max(1, Math.round(s / 60)); return `about ${m} minute${m === 1 ? "" : "s"}`; }
  if (s < 36 * 3600) { const h = Math.round(s / 3600); return `about ${h} hour${h === 1 ? "" : "s"}`; }
  const d = Math.round(s / 86400); return `about ${d} day${d === 1 ? "" : "s"}`;
}

export function fieldUrl(topic: string, definition?: string | null) {
  const p = new URLSearchParams({ topics: topic, scope: "field" });
  if (definition) p.set("definition", definition);
  return `?${p}`;
}

/** The offer under a waiting map, or "" when paying would not make it meaningfully sooner. */
export function offerHtml(key: string, etaS: number, paidEtaS: number | undefined, reserved: boolean, offer: FieldOffer | null): string {
  if (!offer || reserved || paidEtaS == null || etaS - paidEtaS < 3600) return "";
  const href = `${offer.url}${offer.url.includes("?") ? "&" : "?"}client_reference_id=${encodeURIComponent(key)}`;
  return `<div class="field-offer">` +
    `<h3>Want it sooner?</h3>` +
    `<p class="fo-times"><span>In the free line: <b>${esc(inTime(etaS))}</b></span>` +
    `<span>Made now: <b>${esc(inTime(paidEtaS))}</b>, for <b>${dollars(offer.amount)}</b></span></p>` +
    `<ul><li>Like every map here, it becomes public for everyone, with your name on it if you like.</li>` +
    `<li>If it can't be made, you get your money back.</li>` +
    `<li>Stripe emails you a receipt, for your lab or grant.</li></ul>` +
    `<a class="primary-btn" id="field-pay" href="${esc(href)}">Make it now · ${dollars(offer.amount)}</a>` +
    `<p class="quiet small">Card, Apple Pay or Google Pay, through Stripe. No account needed.</p></div>`;
}

const LINE = { paid: "Paid", free: "Free", rebuild: "Update" } as const;

function made(m: FieldMade, url: (m: FieldMade) => string): string {
  const when = m.made ? new Date(m.made).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : "";
  return `<li><a href="${esc(url(m))}" data-go class="fm-topic">${esc(m.topic)}</a>` +
    (m.definition ? ` <span class="quiet small">(${esc(m.definition.length > 90 ? m.definition.slice(0, 88) + "…" : m.definition)})</span>` : "") +
    `<div class="quiet small">${m.papers ? `${m.papers.toLocaleString()} studies in ${m.groups} groups · ` : ""}${esc(when)}` +
    (m.credit ? ` · made sooner thanks to ${esc(m.credit)}` : "") + `</div></li>`;
}

/** The ?fields page: the line, the maps already made (searchable), and asking for a new one. */
export async function renderFields(box: HTMLElement, go: (url: string) => void, q = "") {
  const [qu, all] = await Promise.all([fieldQueue(), fieldsMade(q)]);
  const url = (m: { topic: string; definition?: string | null }) => fieldUrl(m.topic, m.definition);
  const shown = q ? all.filter((m) => (m.score ?? 0) >= 0.5).slice(0, 30) : all;   // holds half the closest studies
  let h = `<h1 class="lib-title">Whole-field maps</h1>` +
    `<p class="quiet lib-sub">Every study on a topic, often tens of thousands, sorted into research groups, each with ` +
    `its reading lists. Each map is made once, then it's open to everyone.</p>` +
    `<form id="fm-find" class="fm-find" role="search"><input id="fm-q" type="search" value="${esc(q)}" ` +
    `placeholder="Find a map already made" aria-label="Find a map already made"></form>`;
  if (qu?.running || qu?.waiting.length) {
    h += `<h2 class="fm-h">In the works</h2><ol class="fm-line">`;
    if (qu.running) {
      const r = qu.running;
      h += `<li class="fm-now"><span class="fm-badge now">Now</span><a href="${esc(url(r))}" data-go>${esc(r.topic)}</a>` +
        `<span class="quiet small"> · step ${(r.stage_index ?? 0) + 1} of ${r.stages ?? 8} · ready in ${esc(inTime(r.ready_s ?? 900))}</span></li>`;
    }
    for (const w of qu.waiting)
      h += `<li><span class="fm-badge ${w.line}">${LINE[w.line]}</span><a href="${esc(url(w))}" data-go>${esc(w.topic)}</a>` +
        `<span class="quiet small"> · ready in ${esc(inTime(w.ready_s ?? 0))}${w.credit ? ` · thanks to ${esc(w.credit)}` : ""}</span></li>`;
    h += `</ol>`;
  }
  h += `<h2 class="fm-h">${q ? "Closest maps already made" : "Made"}</h2>` +
    (shown.length ? `<ul class="fm-made">${shown.map((m) => made(m, url)).join("")}</ul>`
      : `<p class="quiet">${q ? "No map made yet is close to that. Ask for one below." : "None yet."}</p>`);
  h += `<h2 class="fm-h">Ask for a map</h2>` +
    `<form id="fm-ask" class="fm-ask"><input id="fm-topic" required minlength="3" maxlength="200" placeholder="A topic, as papers about it would name it" aria-label="Topic">` +
    `<input id="fm-def" maxlength="600" placeholder="What it covers, in a sentence (optional; helps when the words are ambiguous)" aria-label="What it covers">` +
    `<button class="primary-btn" type="submit">Ask for it</button></form>` +
    `<h2 class="fm-h">How the line works</h2>` +
    `<p class="quiet">Maps are made one at a time, in about 15 minutes each. The free line goes in the order asked, about ` +
    `${qu?.free_per_day ?? 10} maps a day (each uses about a dollar of AI to judge its studies), at most two waiting per reader. ` +
    `A map can be paid for to go first. Topics in the line are public, as their maps will be.</p>`;
  box.innerHTML = h;
  for (const a of box.querySelectorAll<HTMLAnchorElement>("a[data-go]"))
    a.onclick = (e) => { e.preventDefault(); go(a.getAttribute("href")!); };
  (box.querySelector("#fm-find") as HTMLFormElement).onsubmit = (e) => {
    e.preventDefault();
    const v = (box.querySelector("#fm-q") as HTMLInputElement).value.trim();
    go(v ? `?fields=${encodeURIComponent(v)}` : "?fields");
  };
  (box.querySelector("#fm-ask") as HTMLFormElement).onsubmit = (e) => {
    e.preventDefault();
    const t = (box.querySelector("#fm-topic") as HTMLInputElement).value.trim();
    const d = (box.querySelector("#fm-def") as HTMLInputElement).value.trim();
    if (t) go(fieldUrl(t, d || null));
  };
}
