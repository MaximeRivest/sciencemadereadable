/**
 * Do the TypeScript programs send exactly what the Python ones sent?
 *   node --conditions=functai-source tools/check_prompts.ts
 * Reads tools/fixtures.json (python_requests.py): the same inputs rendered by Python functai,
 * and, for the students, the prompt they were trained on.
 */
import { readFileSync } from "node:fs";
import * as P from "../web/src/programs.ts";

const fx = JSON.parse(readFileSync(new URL("./fixtures.json", import.meta.url), "utf8"));
const programs: Record<string, any> = {
  rewriteOpening: P.rewriteOpening, rewriteSection: P.rewriteSection,
  studentOpening: P.studentOpening, studentSection: P.studentSection,
};

const textOf = (content: any): string =>
  typeof content === "string" ? content
    : (content ?? []).map((p: any) => p.text ?? p.value ?? "").join("");

function firstDiff(a: string, b: string): string {
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) i++;
  return `at ${i}: ts=${JSON.stringify(a.slice(Math.max(0, i - 40), i + 60))}\n           py=${JSON.stringify(b.slice(Math.max(0, i - 40), i + 60))}`;
}

let ok = true;
for (const [name, f] of Object.entries<any>(fx)) {
  const lm = name.startsWith("student") ? "openai:local" : f.lm;
  const req: any = await programs[name].render(f.inputs, { lm });
  const tsSystem = textOf(req.system);
  const tsUser = (req.messages ?? []).map((m: any) => `${m.role}: ${textOf(m.parts ?? m.content)}`).join("\n---\n");
  const pyReq = f.request;
  const pySystem = textOf(pyReq.system);
  const pyUser = (pyReq.messages ?? []).map((m: any) => `${m.role}: ${textOf(m.parts ?? m.content)}`).join("\n---\n");
  const checks: [string, string, string][] = [["system", tsSystem, pySystem], ["messages", tsUser, pyUser],
    ["stop", JSON.stringify(req.config?.stop ?? []), JSON.stringify(pyReq.config?.stop ?? [])]];
  if (f.training_prompt) {
    const sys = f.training_prompt.find((m: any) => m.role === "system")?.content ?? "";
    const user = f.training_prompt.filter((m: any) => m.role !== "system").map((m: any) => `${m.role}: ${m.content}`).join("\n---\n");
    checks.push(["system vs training", tsSystem, sys], ["messages vs training", tsUser, user]);
  }
  for (const [what, a, b] of checks) {
    const same = a === b;
    ok &&= same;
    console.log(`${same ? "same     " : "DIFFERENT"} ${name} ${what} (${a.length} chars)` + (same ? "" : "\n   " + firstDiff(a, b)));
  }
}
process.exit(ok ? 0 : 1);
