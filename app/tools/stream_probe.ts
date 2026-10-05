import { OpenAIChatLM } from "@lm15/lm15";
import * as P from "/home/maxime/Projects/scholarsreadinglist/sciencemadereadable/app/web/src/programs.ts";
const client = new OpenAIChatLM({ apiKey: "none", baseUrl: "http://127.0.0.1:8014/v1" });
const router: any = { resolve: (model: string) => ({ provider: "openai", model }),
  complete: (r: any, o: any) => client.complete(r, o), stream: (r: any, o: any) => client.stream(r, o) };
const paper = await (await fetch("http://127.0.0.1:8795/api/paper?doi=10.3390/toxics12010075")).json();
const s: any = P.studentOpening.stream({ paper: "## title\n\n" + paper.sections.title + "\n\n## abstract\n\n" + paper.sections.abstract,
  reference_glossary: "", writer: "opus" } as any, { lm: "our-0.8b", router, temperature: 0, maxTokens: 800 } as any);
const kinds: Record<string, number> = {}; let shown = 0;
for await (const e of s.events()) {
  kinds[e.kind] = (kinds[e.kind] ?? 0) + 1;
  if (e.kind !== "started" && shown < 6 && (e.kind === "text" || e.kind === "field" || e.kind.includes("delta"))) { console.log(JSON.stringify(e).slice(0, 300)); shown++; }
}
console.log(kinds); const v = await s; console.log("value keys:", Object.keys(v ?? {}), "| s.text tail:", JSON.stringify(s.text.slice(-80)));
