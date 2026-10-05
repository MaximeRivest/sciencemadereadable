import { readFileSync } from "node:fs";
import { DOMParser } from "@xmldom/xmldom";
import { splitSections, sectionNodes, layout } from "../web/src/jats.ts";
for (const id of ["PMC12995867", "PMC13376281", "PMC10175599"]) {
  const doc = new DOMParser().parseFromString(readFileSync(new URL(`../../paper_corpus/xml/${id}.xml`, import.meta.url), "utf8"), "text/xml");
  const s = splitSections(doc.documentElement)!; const nodes = sectionNodes(doc.documentElement);
  const figs = doc.getElementsByTagName("fig").length, tabs = doc.getElementsByTagName("table-wrap").length;
  const found: string[] = [];
  for (const k of Object.keys(s)) {
    if (!s[k]) continue;
    const marks = layout(nodes[k] as any, s[k].split("\n\n"));
    for (const m of marks) if (m.kind !== "heading") found.push(`${k}:${m.kind}@${m.at}+${m.count}${m.kind === "figure" || m.kind === "table" ? "[" + m.images.length + " img" + (m.kind === "table" ? ", html " + m.html.length : "") + "]" : ""}`);
    const heads = marks.filter((m) => m.kind === "heading").length; if (heads) found.push(`${k}:${heads} headings`);
  }
  console.log(id, `figs ${figs}, tables ${tabs} in XML ->`, found.join("  "));
}
