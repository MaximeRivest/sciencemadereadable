/**
 * The account page (?account): sign in with GitHub or Google, make and delete API keys, see the plan and what the
 * keys used, upgrade (Stripe payment link) or manage the plan (Stripe's customer portal), delete the account.
 * Back end: app/accounts.py through the door's /api/account/* (a cookie on api.sciencemadereadable.com, sent only
 * by this site's pages). Loaded on demand: most readers never open it.
 */
import { API } from "./jobs.ts";

interface Plan { name: string; per_minute: number; per_month: number | null }
interface Offer { url: string; amount: number; portal?: string | null; tax?: boolean }
interface Key { id: string; name: string; hint: string; created: string; last_used: string | null }
interface Me {
  signed_in: boolean; providers?: { id: string; label: string }[]; plans: Record<"free" | "paid", Plan>; offer: Offer | null;
  id?: string; email?: string | null; name?: string | null; created?: string;
  logins?: { provider: string; label: string; login: string | null; email: string | null }[];
  plan?: { plan: "free" | "paid"; source: "stripe" | "grant" | null; status: string | null; until: number | string | null;
           name: string; per_minute: number; per_month: number | null; note: string | null };
  usage?: { today: number; month: number }; keys?: Key[]; max_keys?: number;
}

const MCP = "https://api.sciencemadereadable.com/mcp";
const BASE = "https://api.sciencemadereadable.com/api/data";
const esc = (s: unknown) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
const dollars = (cents: number) => `$${(cents / 100).toFixed(cents % 100 ? 2 : 0)}`;
const day = (v: string | number | null | undefined) => v == null ? "" :
  new Date(typeof v === "number" ? v * 1000 : v).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
const n = (x: number) => x.toLocaleString();
const dev = /\.ts\.net$|^localhost$|^127\./.test(location.hostname);

const SIGNIN: Record<string, string> = {
  cancelled: "Sign-in was cancelled. Nothing was saved.",
  expired: "That sign-in took too long or was started in another browser. Please try again.",
  provider: "The sign-in service didn't answer properly. Please try again in a minute.",
  unavailable: "That way of signing in isn't available.",
};

async function api(path: string, body?: unknown): Promise<{ ok: boolean; status: number; d: any }> {
  try {
    const r = await fetch(`${API}/api/account${path}`, {
      method: body === undefined ? "GET" : "POST", credentials: "include",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return { ok: r.ok, status: r.status, d: await r.json().catch(() => ({})) };
  } catch { return { ok: false, status: 0, d: { error: "Couldn't reach our server. Check your connection and try again." } }; }
}

function levels(plans: Me["plans"], offer: Offer | null, current?: "free" | "paid") {
  const card = (id: string, title: string, rate: string, sub: string, price: string) =>
    `<div class="acct-level${current === id ? " on" : ""}"><h3>${title}</h3><p class="acct-rate">${rate}</p>` +
    `<p class="quiet small">${sub}</p><p class="acct-price">${price}</p></div>`;
  return `<div class="acct-levels">` +
    card("none", "No key", "30 a minute", "per internet address, for programs and AI agents. Nothing to sign up for.", "Free") +
    card("free", "Free account", `${plans.free.per_minute} a minute`, "for your keys, wherever your programs run.", "Free") +
    card("paid", "Paid", `${plans.paid.per_minute} a minute`,
      `up to ${n(plans.paid.per_month ?? 0)} requests a month, then ${plans.free.per_minute} a minute until the month ends.`,
      offer ? `${dollars(offer.amount)} a month${offer.tax ? " + tax" : ""}` : "Coming soon") +
    `</div><p class="quiet small">A topic map counts as 5 requests. This site's own pages don't need a key.</p>`;
}

function usageHtml(me: Me) {
  const p = me.plan!, u = me.usage!;
  const month = p.per_month ? `${n(u.month)} of ${n(p.per_month)} this month` : `${n(u.month)} this month`;
  return `<p class="acct-usage"><b>${n(p.per_minute)}</b> requests a minute · <b>${n(u.today)}</b> today · ${month}</p>` +
    (p.per_month ? `<div class="acct-bar" role="img" aria-label="${esc(month)}"><span style="width:${Math.min(100, 100 * u.month / p.per_month).toFixed(1)}%"></span></div>` : "") +
    (p.note ? `<p class="acct-note">Right now: ${esc(p.note)}.</p>` : "");
}

function planHtml(me: Me) {
  const p = me.plan!, offer = me.offer;
  let h = `<div class="api-card"><h2 class="acct-h">Your plan: ${esc(p.name)}</h2>${usageHtml(me)}`;
  if (p.plan === "paid" && p.source === "stripe") {
    h += `<p>${p.status === "past_due" ? "Your last payment didn't go through: Stripe will try again. Update your card to keep the paid plan."
      : p.until ? `Renews on ${esc(day(p.until))}.` : ""}</p>` +
      (offer?.portal ? `<a class="primary-btn ghost" href="${esc(offer.portal + (me.email ? `?prefilled_email=${encodeURIComponent(me.email)}` : ""))}">Manage or cancel your plan</a>` +
        `<p class="quiet small">On Stripe's page: your card, receipts, and cancelling. You keep the paid limits until the end of the month paid for.</p>`
        : `<p class="quiet small">To cancel or change your card, use the link in Stripe's emailed receipt.</p>`);
  } else if (p.plan === "paid") {
    h += `<p>Given to you until ${esc(day(String(p.until)))}. Thank you for helping.</p>`;
  } else if (offer && me.id) {
    const href = `${offer.url}${offer.url.includes("?") ? "&" : "?"}client_reference_id=${encodeURIComponent(me.id)}` +
      (me.email ? `&prefilled_email=${encodeURIComponent(me.email)}` : "");
    h += `<p>Need more? The paid plan gives your keys <b>${n(me.plans.paid.per_minute)} requests a minute</b> ` +
      `(${n(me.plans.paid.per_month ?? 0)} a month), and helps pay for the machine this search runs on.</p>` +
      `<a class="primary-btn" id="acct-upgrade" href="${esc(href)}">Upgrade · ${dollars(offer.amount)} a month${offer.tax ? " + tax" : ""}</a>` +
      `<p class="quiet small">Card, Apple Pay or Google Pay, through Stripe. Cancel at any time. Within 14 days of your first ` +
      `payment, ask and you get it back in full. <a href="terms.html">Terms</a></p>`;
  }
  return h + `</div>`;
}

function keysHtml(me: Me) {
  const keys = me.keys ?? [];
  const rows = keys.map((k) => `<li><div><b>${esc(k.name)}</b> <code>${esc(k.hint)}</code><div class="quiet small">made ${esc(day(k.created))}` +
    ` · ${k.last_used ? `last used ${esc(day(k.last_used))}` : "never used yet"}</div></div>` +
    `<button type="button" class="chip" data-revoke="${esc(k.id)}" data-name="${esc(k.name)}">Delete</button></li>`).join("");
  return `<div class="api-card"><h2 class="acct-h">Your API keys</h2>` +
    `<p class="quiet small">A key gives your programs and AI agents your account's limits. Keep it private, like a password. ` +
    `A key that's deleted stops working at once.</p>` +
    (rows ? `<ul class="acct-keys">${rows}</ul>` : `<p>No key yet.</p>`) +
    `<div id="acct-new"></div>` +
    (keys.length < (me.max_keys ?? 10)
      ? `<form id="acct-make" class="acct-make"><input id="acct-key-name" maxlength="60" placeholder="What it's for (e.g. my laptop, lab agent)" aria-label="What the key is for">` +
        `<button class="primary-btn" type="submit">Make a key</button></form>`
      : `<p class="quiet small">That's ${keys.length} keys, the most an account can have. Delete one to make another.</p>`) +
    `</div>`;
}

function newKeyHtml(key: string) {
  const box = (label: string, text: string) => `<p class="api-for">${label}</p><div class="codebox"><pre tabindex="0">${esc(text)}</pre>` +
    `<button type="button" class="copy">Copy</button></div>`;
  return `<div class="acct-fresh" role="status"><h3>Your new key</h3>` +
    `<p><b>Copy it now: it won't be shown again.</b> (Lost it? Delete it and make another.)</p>` +
    box("The key", key) +
    `<h3>How to use it</h3>` +
    box("In code: send it in a header", `curl -H "Authorization: Bearer ${key}" "${BASE}/search?q=coral+reef+bleaching&k=5"`) +
    box("Claude Code", `claude mcp add --transport http --scope user science ${MCP} --header "Authorization: Bearer ${key}"`) +
    box("Any other AI app (Claude app, ChatGPT, Cursor, Codex…): use this MCP address instead of the plain one", `${MCP}?key=${key}`) +
    `<p class="quiet small">An address with the key in it is as private as the key: don't share or publish it.</p></div>`;
}

function wireCopy(root: HTMLElement) {
  for (const b of root.querySelectorAll<HTMLButtonElement>(".codebox .copy"))
    b.onclick = async () => {
      const pre = b.previousElementSibling as HTMLElement;
      try { await navigator.clipboard.writeText(pre.textContent!.trim()); } catch { getSelection()?.selectAllChildren(pre); return; }
      b.textContent = "Copied"; b.classList.add("done");
      setTimeout(() => { b.textContent = "Copy"; b.classList.remove("done"); }, 1600);
    };
}

let message = "";   // shown once at the top: back from signing in or paying

export async function mount(box: HTMLElement, go: (url: string) => void) {
  const q = new URLSearchParams(location.search);
  const signin = q.get("signin"), subscribed = q.get("subscribed");
  if (signin || subscribed) history.replaceState(null, "", "?account");
  if (signin) message = SIGNIN[signin] ?? "Signing in didn't work. Please try again.";
  if (subscribed?.startsWith("cs_")) {
    box.innerHTML = `<h1 class="lib-title">Your account</h1><p role="status">Checking your payment with Stripe…</p>`;
    const r = await api(`/subscribed?session=${encodeURIComponent(subscribed)}`);
    message = r.ok && r.d.accepted ? "Thank you! Your plan is now Paid: your keys have the new limits already. Stripe has emailed you a receipt."
      : r.ok ? `Your payment went through, but ${r.d.why ?? "we couldn't connect it to your account"}. Write to us and we'll sort it out.`
      : `${r.d.error ?? "We couldn't check your payment"}. If you paid, it will show here within a few minutes; reload this page.`;
  }
  await draw(box, go);
}

async function draw(box: HTMLElement, go: (url: string) => void) {
  const r = await api("");
  if (!r.ok) {
    box.innerHTML = `<h1 class="lib-title">Your account</h1><p>${esc(r.d.error ?? "The account service isn't answering right now. Please try again in a minute.")}</p>`;
    return;
  }
  const me = r.d as Me;
  const note = message ? `<p class="acct-message" role="status">${esc(message)}</p>` : "";
  message = "";
  let h = `<h1 class="lib-title">Your account</h1>${note}`;
  if (!me.signed_in) {
    const ps = me.providers ?? [];
    h += `<p class="api-lead">The search is free with no key and no sign-up. An account gives your programs and AI agents ` +
      `their own API key, with more room.</p>` + levels(me.plans, me.offer) +
      `<div class="api-card acct-signin"><h2 class="acct-h">Sign in</h2>` +
      (ps.length ? `<p>No password: use an account you already have. New here? Signing in makes your account.</p><p class="acct-providers">` +
        ps.map((p) => `<a class="primary-btn" href="${esc(`${API}/api/account/login/${p.id}?to=${dev ? "dev" : "site"}`)}">Sign in with ${esc(p.label)}</a>`).join(" ") + `</p>`
        : `<p>Accounts open soon. Until then, everything works without one.</p>`) +
      (dev ? `<p><a class="chip" href="/__dev/login">Dev: sign in as the dev account</a></p>` : "") +
      `<p class="quiet small">We keep your email, your name and how many requests your keys make each day, never what you ` +
      `search. <a href="privacy.html">Privacy</a> · <a href="terms.html">Terms</a></p></div>`;
    box.innerHTML = h;
    return;
  }
  const via = (me.logins ?? []).map((l) => `${esc(l.label)}${l.login ? ` (${esc(l.login)})` : ""}`).join(" and ");
  h += `<p class="acct-who">Signed in as <b>${esc(me.name || me.email || "you")}</b>${me.email && me.name ? ` · ${esc(me.email)}` : ""}` +
    `${via ? ` · with ${via}` : ""} <button type="button" class="chip" id="acct-out">Sign out</button></p>` +
    planHtml(me) + keysHtml(me) +
    `<h2 class="acct-h">Limits</h2>` + levels(me.plans, me.offer, me.plan!.plan) +
    `<p class="quiet small acct-foot">Your account holds your email, name, logins, keys (only a fingerprint of each) and ` +
    `how many requests they made each day; never what you search. <a href="privacy.html">Privacy</a> · <a href="terms.html">Terms</a> · ` +
    `<button type="button" class="linkish" id="acct-delete">Delete my account</button></p>`;
  box.innerHTML = h;
  wire(box, go, me);
}

function wire(box: HTMLElement, go: (url: string) => void, me: Me) {
  const fail = (d: any) => alert(d.error ?? "That didn't work. Please try again.");
  box.querySelector<HTMLButtonElement>("#acct-out")!.onclick = async () => { await api("/logout", {}); await draw(box, go); };
  const make = box.querySelector<HTMLFormElement>("#acct-make");
  if (make) make.onsubmit = async (e) => {
    e.preventDefault();
    const btn = make.querySelector("button")!;
    btn.disabled = true;
    const r = await api("/keys", { name: (box.querySelector("#acct-key-name") as HTMLInputElement).value });
    if (!r.ok) { btn.disabled = false; return fail(r.d); }
    await draw(box, go);
    const fresh = box.querySelector<HTMLElement>("#acct-new")!;
    fresh.innerHTML = newKeyHtml(r.d.key);
    wireCopy(fresh);
    fresh.querySelector<HTMLElement>("h3")?.setAttribute("tabindex", "-1");
    fresh.querySelector<HTMLElement>("h3")?.focus();
  };
  for (const b of box.querySelectorAll<HTMLButtonElement>("[data-revoke]"))
    b.onclick = async () => {
      if (!confirm(`Delete the key "${b.dataset.name}"? Programs using it stop working at once.`)) return;
      const r = await api("/keys/revoke", { id: b.dataset.revoke });
      if (!r.ok) return fail(r.d);
      await draw(box, go);
    };
  box.querySelector<HTMLButtonElement>("#acct-delete")!.onclick = async () => {
    if (me.plan?.source === "stripe" && me.plan.plan === "paid")
      return alert("Cancel your paid plan first (Manage or cancel your plan), then delete the account.");
    if (!confirm("Delete your account? Your keys stop working at once, and everything we keep about you is erased. This can't be undone.")) return;
    const r = await api("/delete", { confirm: "delete" });
    if (!r.ok) return fail(r.d);
    message = "Your account is deleted. Everything we kept about it is erased.";
    await draw(box, go);
  };
}
