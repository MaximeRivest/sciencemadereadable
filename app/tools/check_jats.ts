/** web/src/jats.ts against the Python corpus code: node --conditions=functai-source tools/check_jats.ts TRUTH.json */
import { readFileSync } from "node:fs";
import { DOMParser } from "@xmldom/xmldom";
import { splitSections, licenseOk } from "../web/src/jats.ts";
const truth = JSON.parse(readFileSync(process.argv[2], "utf8"));
let same = 0, lic = 0; const bad: string[] = [];
for (const t of truth) {
  const doc = new DOMParser().parseFromString(readFileSync(t.file, "utf8"), "text/xml");
  const s = splitSections(doc.documentElement);
  const a = JSON.stringify(s), b = JSON.stringify(t.sections);
  if (a === b) same++;
  else if (bad.length < 4) {
    const k = Object.keys(t.sections ?? s ?? {}).find((k) => (s?.[k] ?? null) !== (t.sections?.[k] ?? null));
    const x = s?.[k!] ?? "", y = t.sections?.[k!] ?? "";
    let i = 0; while (i < x.length && x[i] === y[i]) i++;
    bad.push(`${t.file.split("/").pop()} ${k}: js=${JSON.stringify(x.slice(i - 30, i + 40))} py=${JSON.stringify(y.slice(i - 30, i + 40))}`);
  }
  if (licenseOk(doc.documentElement) === t.license) lic++;
}
console.log(`sections identical: ${same}/${truth.length}; licence verdict identical: ${lic}/${truth.length}`);
bad.forEach((b) => console.log("  ", b));
