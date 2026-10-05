/**
 * The page's big-model path (pipeline.ts), run in Node through the saved subscription logins
 * instead of API keys:  node --conditions=functai-source tools/smoke_big.ts luna|sonnet DOI
 */
import { LMRouter, ClaudeCodeLM } from "@lm15/lm15";
import { MODELS } from "../web/src/models.ts";
import { rewrite } from "../web/src/pipeline.ts";

const [which, doi] = process.argv.slice(2);
const base = MODELS.find((m) => m.id === which)!;
const paper = await (await fetch(`http://127.0.0.1:8795/api/paper?doi=${encodeURIComponent(doi)}`)).json();
let router: any, lm: string;
if (base.provider === "anthropic") {
  const client = new ClaudeCodeLM({ claudeCodeVersion: "2.1.284" });
  router = { resolve: (model: string) => ({ provider: "anthropic", model }), complete: (r: any, o: any) => client.complete(r, o),
             stream: (r: any, o: any) => client.stream(r, o) };
  lm = base.lm;
} else {
  router = new LMRouter();
  lm = `openai-codex:${base.lm}`;
}
const started = Date.now();
const parts: Record<string, string> = {};
let tin = 0, tout = 0, updates = 0, first = 0;
await rewrite({ ...base, lm }, paper, {}, {
  part: (k, t, done) => {
    if (!done) { updates++; if (!first && t) first = Date.now() - started; return; }
    parts[k] = t; console.log(`  ${k}: ${t.length} chars · ${t.slice(0, 70).replace(/\s+/g, " ")}`);
  },
  status: (s) => console.log(s), usage: (i, o) => { tin += i; tout += o; },
}, new AbortController().signal, router);
console.log(`${which}: first text after ${(first / 1000).toFixed(1)} s, ${updates} streamed updates`);
console.log(`${which}: ${Object.keys(parts).length} parts in ${((Date.now() - started) / 1000).toFixed(0)} s; ${tin} in, ${tout} out`);
