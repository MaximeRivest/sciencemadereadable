/**
 * Reading a JATS article (PubMed Central's XML) in the browser, exactly as the Python corpus code
 * does, so every model gets the text it was scored with:
 *   render / text      rewrite_benchmark/prepare.py
 *   splitSections      paper_corpus/harvest.py split_sections
 *   licenseOk          paper_corpus/harvest.py license_ok
 * tools/check_jats.ts checks these against the Python output on the corpus papers.
 * On top: layout(), where each figure, table and heading sits among the paragraphs, for display.
 */

type N = any;   // a DOM node (the browser's DOMParser, or xmldom in the tests)

const BLOCKS = new Set(["article-title", "abstract", "sec", "title", "p", "list", "list-item", "fig", "caption",
  "table-wrap", "table", "thead", "tbody", "tfoot", "tr", "disp-formula", "ref", "ref-list", "ack", "app",
  "app-group", "fn", "fn-group", "supplementary-material", "statement"]);
const KINDS: [string, string[]][] = [["introduction", ["introduction", "background"]], ["methods", ["method", "material"]],
  ["results", ["result"]], ["discussion", ["discussion"]], ["conclusion", ["conclusion"]]];

export const tag = (n: N): string => n.localName ?? String(n.nodeName).replace(/^.*:/, "");
const elements = (n: N): N[] => Array.from(n.childNodes ?? []).filter((c: N) => c.nodeType === 1);
const child = (n: N, name: string): N | null => elements(n).find((c) => tag(c) === name) ?? null;
const children = (n: N, name: string): N[] => elements(n).filter((c) => tag(c) === name);

// Python's str.isspace / \s (Unicode) and JavaScript's differ on a few characters: use Python's
const PY_SPACE = "\\t\\n\\x0b\\x0c\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const SPACES = new RegExp(`[${PY_SPACE}]+`, "g");
const BLANK_LINE = new RegExp(`\\n[${PY_SPACE}]*\\n`);
const strip = (s: string) => s.replace(new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "g"), "");

/** ElementTree's element.text: the text before the first child element. */
function leadText(n: N): string {
  let s = "";
  for (const c of Array.from(n.childNodes ?? []) as N[]) {
    if (c.nodeType === 1) break;
    if (c.nodeType === 3 || c.nodeType === 4) s += c.nodeValue;
  }
  return s;
}

export function render(n: N): string {
  const name = tag(n);
  if (name === "graphic" || name === "inline-graphic" || name === "object-id") return "";
  if (name === "alternatives") {
    for (const preferred of ["table", "tex-math", "math"]) {
      const m = child(n, preferred);
      if (m) return render(m);
    }
    return "";
  }
  let value = "";
  for (const c of Array.from(n.childNodes ?? []) as N[]) {
    if (c.nodeType === 3 || c.nodeType === 4) value += c.nodeValue;
    else if (c.nodeType === 1) value += render(c);
  }
  if (BLOCKS.has(name)) return "\n\n" + value + "\n\n";
  if (name === "td" || name === "th") return " | " + value + " ";
  return value;
}

/** The node as the models read it: paragraphs separated by a blank line. */
export function text(n: N | null): string {
  if (!n) return "";
  return render(n).split(BLANK_LINE).map((p) => strip(p.replace(SPACES, " "))).filter(Boolean).join("\n\n");
}

export type Sections = Record<string, string>;

/** The translator's sections; null when it is not a normal research paper. */
export function splitSections(root: N): Sections | null {
  const front = child(root, "front");
  const meta = front ? child(front, "article-meta") : null;
  const found: Record<string, N> = {};
  const body = child(root, "body");
  for (const sec of body ? children(body, "sec") : []) {
    const t = child(sec, "title");
    const heading = (t ? leadText(t) : "").toLowerCase();
    for (const [kind, words] of KINDS) {
      if (!(kind in found) && words.some((w) => heading.includes(w))) { found[kind] = sec; break; }
    }
  }
  if (!["introduction", "methods", "results"].every((k) => k in found) || !meta) return null;
  const intro = found.introduction;
  const paragraphs = children(intro, "p");
  const titleGroup = child(meta, "title-group");
  const s: Sections = {
    title: text(titleGroup ? child(titleGroup, "article-title") : null),
    abstract: text(child(meta, "abstract")),
    introduction_first: paragraphs.length ? text(paragraphs[0]) : "",
    introduction_rest: [...paragraphs.slice(1).map(text), ...children(intro, "sec").map(text)].join("\n\n"),
  };
  for (const kind of ["methods", "results", "discussion", "conclusion"]) s[kind] = found[kind] ? text(found[kind]) : "";
  return s;
}

function iterAll(n: N, out: N[] = []): N[] {
  out.push(n);
  for (const c of elements(n)) iterAll(c, out);
  return out;
}
function itertext(n: N): string {
  let s = "";
  for (const c of Array.from(n.childNodes ?? []) as N[]) {
    if (c.nodeType === 3 || c.nodeType === 4) s += c.nodeValue;
    else if (c.nodeType === 1) s += itertext(c);
  }
  return s;
}

/** CC BY (any version); non-commercial, no-derivatives and share-alike licences refused. */
export function licenseOk(root: N): boolean {
  for (const lic of iterAll(root)) {
    if (!String(lic.nodeName).endsWith("license")) continue;
    const links = iterAll(lic).flatMap((e: N) => Array.from(e.attributes ?? []).map((a: N) => a.value));
    const blob = (links.join(" ") + " " + itertext(lic)).toLowerCase();
    if (["licenses/by-nc", "licenses/by-nd", "licenses/by-sa", "noncommercial", "no derivatives"].some((x) => blob.includes(x))) return false;
    if (blob.includes("creativecommons.org/licenses/by/") || blob.includes("creative commons attribution")
        || blob.includes("cc by license") || blob.includes("(cc by)")) return true;
    if (blob.includes("broadest form of re-use") && blob.includes("commercial")) return true;
  }
  return false;
}

// ---------------------------------------------------------------- display

export interface Figure { kind: "figure"; at: number; count: number; label: string; images: string[] }
export interface Table { kind: "table"; at: number; count: number; label: string; captionCount: number; html: string; images: string[] }
export interface Heading { kind: "heading"; at: number; count: number }
export type Mark = Figure | Table | Heading;

/**
 * Where figures, tables and subheadings sit among a section's paragraphs (indices into
 * text(section).split("\n\n")), found by matching each element's own paragraphs in order.
 */
export function layout(sectionNode: N | N[] | null, paragraphs: string[]): Mark[] {
  const nodes = Array.isArray(sectionNode) ? sectionNode : sectionNode ? [sectionNode] : [];
  const marks: Mark[] = [];
  let cursor = 0;
  const locate = (own: string[]) => {
    for (let i = cursor; i + own.length <= paragraphs.length; i++) {
      if (own.every((p, k) => paragraphs[i + k] === p)) return i;
    }
    return -1;
  };
  const visit = (n: N) => {
    const name = tag(n);
    if (name === "fig" || name === "table-wrap") {
      const own = text(n).split("\n\n").filter(Boolean);
      const at = own.length ? locate(own) : -1;
      const images = iterAll(n).filter((e: N) => tag(e) === "graphic" && (e.getAttribute("content-type") ?? "image") !== "thumb")
        .map((e: N) => e.getAttribute("xlink:href") ?? e.getAttributeNS?.("http://www.w3.org/1999/xlink", "href") ?? "")
        .filter(Boolean);
      const label = leadText(child(n, "label") ?? { childNodes: [] }).trim() || itertext(child(n, "label") ?? { childNodes: [] }).trim();
      if (at >= 0) {
        if (name === "fig") marks.push({ kind: "figure", at, count: own.length, label, images });
        else {
          const captionCount = [child(n, "label"), child(n, "caption")].filter(Boolean)
            .reduce((k: number, e: N) => k + text(e).split("\n\n").filter(Boolean).length, 0);
          const table = iterAll(n).find((e: N) => tag(e) === "table");
          marks.push({ kind: "table", at, count: own.length, label, captionCount, html: table ? tableHtml(table) : "", images });
        }
        cursor = at + own.length;
      }
      return;
    }
    if (name === "title") {
      const own = text(n).split("\n\n").filter(Boolean);
      const at = own.length ? locate(own) : -1;
      if (at >= 0) { marks.push({ kind: "heading", at, count: own.length }); cursor = at + own.length; }
      return;
    }
    for (const c of elements(n)) visit(c);
  };
  for (const n of nodes) visit(n);
  return marks;
}

/** A table as safe HTML: only table structure and plain text (no attributes but spans). */
function tableHtml(t: N): string {
  const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));
  const walk = (n: N): string => {
    let s = "";
    for (const c of Array.from(n.childNodes ?? []) as N[]) {
      if (c.nodeType === 3 || c.nodeType === 4) { s += esc(c.nodeValue); continue; }
      if (c.nodeType !== 1) continue;
      const name = tag(c);
      if (["thead", "tbody", "tfoot", "tr"].includes(name)) s += `<${name}>${walk(c)}</${name}>`;
      else if (name === "td" || name === "th") {
        const span = ["colspan", "rowspan"].map((a) => /^\d+$/.test(c.getAttribute(a) ?? "") ? ` ${a}="${c.getAttribute(a)}"` : "").join("");
        s += `<${name}${span}>${walk(c)}</${name}>`;
      } else if (name === "sup" || name === "sub" || name === "italic" || name === "bold") {
        const h = { sup: "sup", sub: "sub", italic: "i", bold: "b" }[name as "sup"];
        s += `<${h}>${walk(c)}</${h}>`;
      } else if (name === "break") s += "<br>";
      else s += walk(c);
    }
    return s;
  };
  return `<table>${walk(t)}</table>`;
}

/** The section elements as splitSections chose them (for layout). */
export function sectionNodes(root: N): Record<string, N | N[] | null> {
  const front = child(root, "front");
  const meta = front ? child(front, "article-meta") : null;
  const body = child(root, "body");
  const found: Record<string, N> = {};
  for (const sec of body ? children(body, "sec") : []) {
    const t = child(sec, "title");
    const heading = (t ? leadText(t) : "").toLowerCase();
    for (const [kind, words] of KINDS) {
      if (!(kind in found) && words.some((w) => heading.includes(w))) { found[kind] = sec; break; }
    }
  }
  const intro = found.introduction;
  const ps = intro ? children(intro, "p") : [];
  return {
    title: null, abstract: meta ? child(meta, "abstract") : null,
    introduction_first: ps[0] ?? null,
    introduction_rest: intro ? [...ps.slice(1), ...children(intro, "sec")] : null,
    methods: found.methods ?? null, results: found.results ?? null,
    discussion: found.discussion ?? null, conclusion: found.conclusion ?? null,
  };
}
