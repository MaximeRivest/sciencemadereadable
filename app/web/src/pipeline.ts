/**
 * Rewriting one paper, step by step as in the scored runs:
 * - big models: rewrite_benchmark/translator.py translate_paper (the opening from the
 *   whole paper, then the four other sections at the same time, each up to 3 attempts);
 * Our models run on our GPU (worker/worker.ts), through the queue (jobs.ts).
 */
import { LMRouter, createTransport } from "@lm15/lm15/browser";

/**
 * The requests go out from the reader's browser. Anthropic answers a page's request only when it
 * carries "anthropic-dangerous-direct-browser-access: true" (its way of making sure a site means
 * to expose the user's own key to the browser); lm15 sends that header only for Claude Code
 * logins, so the transport adds it for api.anthropic.com. Long read budget: a slow model can take
 * minutes before the first byte of a long section.
 */
const BASE = createTransport({ timeouts: { read: 1800 } });
const transport = {
  send: (req: any, o?: any) => BASE.send(req.url.startsWith("https://api.anthropic.com/")
    ? { ...req, headers: [...req.headers, ["anthropic-dangerous-direct-browser-access", "true"]] } : req, o),
};
import * as P from "./programs.ts";
import { partialStrings } from "./partial.ts";
import type { ModelInfo } from "./models.ts";

export interface Paper { sections: Record<string, string> }

export interface Keys { anthropic?: string; openai?: string }

export interface Progress {
  part(name: string, text: string, done: boolean): void;   // a rewritten part, so far (done: final, checked)
  status(text: string): void;
  usage(input: number, output: number): void;
}

const wholePaper = (s: Record<string, string>) =>
  P.ALL_SECTIONS.filter((k) => s[k]).map((k) => `## ${k}\n\n${s[k]}`).join("\n\n");

const openingMarkdown = (o: Record<string, string>) =>
  P.OPENING_SECTIONS.filter((k) => o[k]).map((k) => `## ${k}\n\n${o[k]}`).join("\n\n");

function addUsage(pred: any, progress: Progress) {
  for (const r of pred?.responses ?? []) {
    const u = r?.usage ?? {};
    progress.usage(u.inputTokens ?? u.input_tokens ?? 0, u.outputTokens ?? u.output_tokens ?? 0);
  }
}

/**
 * One call, streamed: `show(field, textSoFar)` for every piece, the finished prediction at the end.
 * A new request (a re-ask after an unreadable reply) starts every field afresh.
 */
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

/** Up to 3 attempts, as translator.call_with_one_retry; a cancel or a refused key stops at once. */
async function attempts<T>(f: () => Promise<T>, signal: AbortSignal): Promise<T> {
  let last: unknown;
  for (let i = 0; i < 3; i++) {
    try { return await f(); } catch (e: any) {
      last = e;
      const name = e?.name ?? e?.constructor?.name ?? "";
      if (signal.aborted || /Cancel|Auth|Billing|Abort/.test(name) || /401|403|api key|credit/i.test(String(e?.message))) throw e;
      if (/RateLimit/.test(name)) await new Promise((r) => setTimeout(r, 20000));
      else await new Promise((r) => setTimeout(r, 3000 * (i + 1)));
    }
  }
  throw last;
}

/** `router` replaces the key-based connection (tests: a subscription login instead of an API key). */
/**
 * OpenAI's answer to a wrong key on /v1/responses lacks the header that lets a page read it, so
 * the browser only reports "fetch failed". Its model list does carry the header: ask it first, for
 * a clear message, and to check that the account can use this model.
 */
async function checkOpenAIKey(key: string, m: ModelInfo, signal: AbortSignal) {
  let r: Response;
  try {
    r = await fetch("https://api.openai.com/v1/models", { headers: { Authorization: `Bearer ${key}` }, signal });
  } catch {
    throw new Error("Couldn't reach OpenAI from this browser (network, or an extension blocking it).");
  }
  if (r.status === 401) throw new Error("OpenAI says this API key is invalid. Check the key in “API keys”.");
  if (!r.ok) throw new Error(`OpenAI refused the key check (HTTP ${r.status}).`);
  const ids: string[] = ((await r.json()).data ?? []).map((d: any) => d.id);
  if (ids.length && !ids.includes(m.lm)) throw new Error(`Your OpenAI account can't use ${m.lm} (it isn't in your model list).`);
}

export async function rewrite(model: ModelInfo, paper: Paper, keys: Keys, progress: Progress, signal: AbortSignal,
                              router?: any) {
  return big(model, paper, keys, progress, signal, router);   // our models: the GPU queue (jobs.ts)
}

async function big(m: ModelInfo, paper: Paper, keys: Keys, progress: Progress, signal: AbortSignal, given?: any) {
  const key = m.provider === "anthropic" ? keys.anthropic : keys.openai;
  if (!key && !given) throw new Error(`Add your ${m.provider === "anthropic" ? "Anthropic" : "OpenAI"} API key first.`);
  if (m.provider === "openai" && !given) await checkOpenAIKey(key!, m, signal);
  const router = given ?? new LMRouter({ apiKeys: { [m.provider]: key! }, transport });
  const settings: any = { lm: m.lm, router, signal, config: { reasoning: { effort: m.effort ?? "off" } },
                          ...(m.maxTokens ? { maxTokens: m.maxTokens } : {}) };
  const s = paper.sections;
  progress.status("Writing the opening (title, abstract, first paragraph, conclusion)…");
  const op: any = await attempts(() => streamed(P.rewriteOpening, {
    title: s.title ?? "", abstract: s.abstract ?? "", introduction_first: s.introduction_first ?? "",
    conclusion: s.conclusion ?? "", original_paper: wholePaper(s), writing_brief: P.WRITING_BRIEF,
  }, settings, (_f, json) => {
    const so = partialStrings(json, P.OPENING_SECTIONS);
    for (const k of P.OPENING_SECTIONS) if (s[k]) progress.part(k, so[k] ?? "", false);
  }), signal);
  addUsage(op, progress);
  const opening = op.answer;
  for (const k of P.OPENING_SECTIONS) if (s[k]) progress.part(k, opening[k] ?? "", true);
  const todo = P.OTHER_SECTIONS.filter((k) => s[k]);
  progress.status(`Writing ${todo.length} sections at the same time…`);
  await Promise.all(todo.map(async (k) => {
    const pr: any = await attempts(() => streamed(P.rewriteSection, {
      section_name: k, original_section: s[k], section_guidance: P.SECTION_GUIDANCE[k],
      rewritten_opening: openingMarkdown(opening), glossary: opening.glossary ?? [],
      reader_habits: P.READER_HABITS, writing_brief: P.WRITING_BRIEF,
    }, settings, (_f, json) => progress.part(k, partialStrings(json, ["text"]).text ?? "", false)), signal);
    addUsage(pr, progress);
    progress.part(k, pr.answer.text, true);
  }));
}
