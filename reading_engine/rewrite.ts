/** Off-site rewrite: uses the live program definitions, never the live queue.
 * node --conditions=functai-source rewrite.ts runs/toxics
 * Sequential section calls deliberately limit contention with the site's GPU.
 */
import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  existsSync,
  renameSync,
} from "node:fs";
import { resolve, join } from "node:path";
import { createHash } from "node:crypto";
import { OpenAIChatLM } from "../app/node_modules/@lm15/lm15/dist/index.js";
import * as P from "../app/web/src/programs.ts";

const dir = resolve(process.argv[2] ?? "runs/toxics");
const paper = JSON.parse(readFileSync(join(dir, "paper.json"), "utf8"));
const labels = JSON.parse(readFileSync(join(dir, "labelled.json"), "utf8"));
if (
  !labels.ready ||
  JSON.stringify(labels.sections) !== JSON.stringify(paper.sections)
)
  throw new Error(
    "Labels need review or paper.json is stale; prepare the input again.",
  );
const baseUrl = process.env.REWRITER_URL ?? "http://127.0.0.1:8015/v1";
const model = process.env.REWRITER_MODEL ?? "our-9b";
const inventoryResponse = await fetch(`${baseUrl}/models`);
if (!inventoryResponse.ok)
  throw new Error(`Model service: HTTP ${inventoryResponse.status}`);
const inventory = await inventoryResponse.json();
const modelInfo = inventory.data.find((m: any) => m.id === model);
if (!modelInfo) throw new Error(`Model ${model} is not served`);
const client = new OpenAIChatLM({ apiKey: "none", baseUrl });
const router: any = {
  resolve: (name: string) => ({ provider: "openai", model: name }),
};
const cacheDir = join(dir, "rewrite-cache");
mkdirSync(cacheDir, { recursive: true });
const calls: any[] = [];
const started = performance.now();
const settings = { lm: model, router, temperature: 0, apiRetries: 0 } as any;
function save(path: string, value: any) {
  writeFileSync(path + ".tmp", JSON.stringify(value, null, 2) + "\n");
  renameSync(path + ".tmp", path);
}
function tag(text: string, name: string): string {
  const matches = [
    ...text.matchAll(
      new RegExp(`<${name}>\\n?([\\s\\S]*?)\\n?</${name}>`, "g"),
    ),
  ];
  if (matches.length !== 1)
    throw new Error(`Expected exactly one complete <${name}> field`);
  return matches[0][1].trim();
}
async function call(
  name: string,
  fn: any,
  inputs: any,
  maxTokens: number,
): Promise<string> {
  // Render the application's actual function: no copied prompt templates.
  const request: any = await fn.render(inputs, { ...settings, maxTokens });
  const requestJSON = JSON.parse(JSON.stringify(request));
  const key = createHash("sha256")
    .update(
      JSON.stringify({
        request: requestJSON,
        baseUrl,
        modelRoot: modelInfo.root,
      }),
    )
    .digest("hex");
  const path = join(cacheDir, `${key}.json`);
  let record: any;
  const cached = existsSync(path);
  if (cached) record = JSON.parse(readFileSync(path, "utf8"));
  else {
    console.log(`Rewriting ${name}…`);
    const begin = performance.now();
    const reply = await client.complete(request);
    record = {
      name,
      request: requestJSON,
      modelInfo,
      seconds: (performance.now() - begin) / 1000,
      text: reply.text,
      finishReason: reply.finishReason,
      response: reply.toJSON(),
    };
    // Keep failures for diagnosis, but never treat them as a completed cached call.
    if (reply.finishReason !== "stop" || !reply.text?.trim()) {
      save(path + ".failed", record);
      throw new Error(`${name} did not finish normally: ${reply.finishReason}`);
    }
    save(path, record);
  }
  calls.push({
    name,
    seconds: record.seconds,
    cacheHit: cached,
    finishReason: record.finishReason,
    usage: record.response.usage,
    file: `rewrite-cache/${key}.json`,
  });
  console.log(
    `${name}: ${cached ? "cached" : record.seconds.toFixed(1) + "s"}`,
  );
  return record.text;
}
const sections = paper.sections as Record<string, string>;
const whole = P.ALL_SECTIONS.filter((k) => sections[k])
  .map((k) => `## ${k}\n\n${sections[k]}`)
  .join("\n\n");
const raw = await call(
  "opening",
  P.studentOpening,
  {
    paper: whole,
    reference_glossary: paper.glossary.opening ?? "",
    writer: "opus",
  },
  paper.max_tokens.opening,
);
const parts: Record<string, string> = {};
for (const k of P.OPENING_SECTIONS) {
  const value = tag(raw, k);
  if (sections[k] && !value) throw new Error(`Opening omitted ${k}`);
  if (!sections[k] && value) throw new Error(`Opening invented absent ${k}`);
  parts[k] = value;
}
const opening = P.OPENING_SECTIONS.filter((k) => parts[k])
  .map((k) => `## ${k}\n\n${parts[k]}`)
  .join("\n\n");
for (const k of P.OTHER_SECTIONS.filter((k) => sections[k])) {
  parts[k] = tag(
    await call(
      k,
      P.studentSection,
      {
        section_name: k,
        original_section: sections[k],
        rewritten_opening: opening,
        reference_glossary: paper.glossary[k] ?? "",
        writer: "opus",
      },
      paper.max_tokens[k],
    ),
    "result",
  );
  if (!parts[k]) throw new Error(`Rewriter omitted ${k}`);
}
const out = {
  parts,
  calls,
  modelInfo,
  baseUrl,
  sectionsSha256: paper.sections_sha256,
  secondsThisRun: (performance.now() - started) / 1000,
  noLiveQueueUsed: true,
  note: "Smoke test, not an evaluation of factual fidelity. Same programs, style, glossary and reply limits as the site; sequential section scheduling.",
};
save(join(dir, "rewrite.json"), out);
const order = [
  "title",
  "abstract",
  "introduction_first",
  "introduction_rest",
  "methods",
  "results",
  "discussion",
  "conclusion",
];
writeFileSync(
  join(dir, "rewrite.md"),
  "> Prototype output, not fully fact-checked. Tables below are model output, not verified original cells. Consult original.pdf for authoritative numbers.\n\n" +
    order
      .filter((k) => parts[k])
      .map((k) => `## ${k}\n\n${parts[k]}`)
      .join("\n\n") +
    "\n",
);
console.log(`Done; results only in ${dir}`);
