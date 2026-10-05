/**
 * The home GPU worker: takes rewrite jobs from the queue, runs our model on this machine's GPU,
 * and sends the text back as it is written.
 *
 *   node --conditions=functai-source worker/worker.ts
 *
 * worker/config.json:
 *   { "queue": "http://127.0.0.1:8795",          where jobs come from (later: the public queue)
 *     "papers": "http://127.0.0.1:8795",         the paper service (sections, glossary): local only
 *     "models": { "our-9b": "http://127.0.0.1:8015/v1" } }   our models' vLLM servers
 * The queue's token is read from app/worker_token.
 *
 * Each paper is rewritten exactly as in the scored runs (web/src/programs.ts: the same requests,
 * checked byte for byte): the opening, then the four other sections at once, greedy decoding,
 * the reference glossary and reply limits from the paper service.
 */
import { readFileSync } from "node:fs";
import { OpenAIChatLM } from "@lm15/lm15";
import * as P from "../web/src/programs.ts";

const here = new URL(".", import.meta.url);
// WORKER_CONFIG: another settings file (e.g. worker/config-h100.json for a rented GPU)
const config = JSON.parse(readFileSync(process.env.WORKER_CONFIG ?? new URL("config.json", here), "utf8"));
const token = readFileSync(new URL("../worker_token", here), "utf8").trim();
const log = (...a: unknown[]) => console.log(new Date().toISOString().slice(11, 19), ...a);

async function queue(path: string, body: unknown = {}) {
  const r = await fetch(config.queue + path, {
    method: "POST", body: JSON.stringify(body),
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
  });
  if (!r.ok) throw new Error(`queue ${path}: HTTP ${r.status}`);
  return r.json();
}

async function streamed(fn: any, inputs: any, settings: any, show: (field: string, text: string) => void) {
  const s = fn.stream(inputs, settings);
  const text: Record<string, string> = {};
  for await (const e of s.events()) {
    if (e.kind === "request" || e.kind === "retry") for (const k of Object.keys(text)) { text[k] = ""; show(k, ""); }
    if (e.kind === "text" && typeof e.text === "string") {
      const f = e.field ?? "result";
      text[f] = (text[f] ?? "") + e.text;
      show(f, text[f]);
    }
  }
  return await s.prediction;
}

async function run(job: { id: string; doi: string; model: string; paper?: any }) {
  const started = Date.now();
  let paper = job.paper;   // prepared by the queue (sections, glossary, reply limits)
  if (!paper) {
    const r = await fetch(`${config.papers}/api/paper?doi=${encodeURIComponent(job.doi)}`);
    paper = await r.json();
    if (!r.ok) throw new Error(paper.error ?? `paper service: HTTP ${r.status}`);
  }
  const client = new OpenAIChatLM({ apiKey: "none", baseUrl: config.models[job.model] });
  const router: any = { resolve: (model: string) => ({ provider: "openai", model }),
    complete: (q: any, o: any) => client.complete(q, o), stream: (q: any, o: any) => client.stream(q, o) };
  const base = { lm: job.model, router, temperature: 0, apiRetries: 0 } as any;
  const s = paper.sections as Record<string, string>;

  const parts: Record<string, string> = {};
  const done: string[] = [];
  let dirty = true;
  let quiet = 0;
  const send = setInterval(() => {   // progress, about once a second (and a sign of life every 5 s)
    if (!dirty && ++quiet < 5) return;
    dirty = false;
    quiet = 0;
    queue(`/api/worker/jobs/${job.id}`, { parts, done }).catch((e) => log("progress not sent:", e.message));
  }, 1000);
  const show = (k: string, t: string) => { if (s[k] !== undefined) { parts[k] = t; dirty = true; } };
  try {
    const whole = P.ALL_SECTIONS.filter((k) => s[k]).map((k) => `## ${k}\n\n${s[k]}`).join("\n\n");
    const op: any = await streamed(P.studentOpening,
      { paper: whole, reference_glossary: paper.glossary.opening ?? "", writer: "opus" },
      { ...base, maxTokens: paper.max_tokens.opening ?? 6000 }, show);
    const opening = op.outputs;
    for (const k of P.OPENING_SECTIONS) if (s[k]) { parts[k] = opening[k] ?? ""; done.push(k); }
    dirty = true;
    const rewrittenOpening = P.OPENING_SECTIONS.filter((k) => opening[k]).map((k) => `## ${k}\n\n${opening[k]}`).join("\n\n");
    await Promise.all(P.OTHER_SECTIONS.filter((k) => s[k]).map(async (k) => {
      const pr: any = await streamed(P.studentSection, {
        section_name: k, original_section: s[k], rewritten_opening: rewrittenOpening,
        reference_glossary: paper.glossary[k] ?? "", writer: "opus",
      }, { ...base, maxTokens: paper.max_tokens[k] }, (_f, t) => show(k, t));
      parts[k] = pr.answer;
      done.push(k);
      dirty = true;
    }));
  } finally {
    clearInterval(send);
  }
  await queue(`/api/worker/jobs/${job.id}`, { parts, done, status: "done" });
  log(`${job.id} ${job.model} ${job.doi}: done in ${((Date.now() - started) / 1000).toFixed(0)} s`);
}

const PARALLEL = config.parallel ?? 3;   // papers at once: vLLM batches their requests on the GPU
const NAME = config.name ?? (process.env.WORKER_CONFIG ?? "config.json").split("/").pop()!.replace(/\.json$/, "");
let busy = 0;
log(`worker ${NAME}: models ${Object.keys(config.models).join(", ")}; ${PARALLEL} papers at once; queue ${config.queue}`);

// Only models that answer are offered: one restarting, evicted by the model router, or behind a dropped
// tunnel is left out, so its papers wait in line instead of failing here. Checked every 15 s.
let live: string[] = [];
let checked = 0;
async function liveModels(): Promise<string[]> {
  if (Date.now() - checked < 15000) return live;
  checked = Date.now();
  const now = await Promise.all(Object.entries(config.models as Record<string, string>).map(async ([m, url]) => {
    try { const r = await fetch(`${url}/models`, { signal: AbortSignal.timeout(4000) }); return r.ok ? m : null; }
    catch { return null; }
  }));
  const next = now.filter((m): m is string => !!m);
  if (next.join() !== live.join()) log(`models answering: ${next.join(", ") || "none"}`);
  live = next;
  return live;
}
for (;;) {
  try {
    if (busy >= PARALLEL) {   // all slots taken: just say we're alive
      await queue("/api/worker/next", { name: NAME, models: await liveModels(), parallel: PARALLEL, busy: true });
      await new Promise((r) => setTimeout(r, 5000));
      continue;
    }
    const models = await liveModels();
    const job = await queue("/api/worker/next", { name: NAME, models, parallel: PARALLEL });
    if (!job.id) { await new Promise((r) => setTimeout(r, 1500)); continue; }
    log(`${job.id} ${job.model} ${job.doi}: started (${busy + 1} running)`);
    busy++;
    run(job).catch(async (e: any) => {
      log(`${job.id}: failed: ${e.message}`);
      await queue(`/api/worker/jobs/${job.id}`, { status: "failed", error: e.message }).catch(() => {});
    }).finally(() => { busy--; });
  } catch (e: any) {
    log("queue unreachable:", e.message);
    await new Promise((r) => setTimeout(r, 10000));
  }
}
