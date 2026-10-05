/**
 * The reading view: the rewrite (main column) next to the original (side column), paragraph by
 * paragraph, with the paper's own figures and tables in their places. Tables are always the
 * publisher's (a model's rewritten table is never shown); figure captions are shown rewritten.
 * A section still being written shows as one flowing block; once finished, its paragraphs are
 * matched to the original's (align.ts).
 */
import { marked } from "marked";
import DOMPurify from "dompurify";
import { align, paragraphs } from "./align.ts";
import { image, type Paper } from "./paper.ts";
import type { Mark } from "./jats.ts";

const ORDER = ["abstract", "introduction_first", "introduction_rest", "methods", "results", "discussion", "conclusion"];
const LABEL: Record<string, string> = { abstract: "In short", introduction_first: "Introduction" };

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
const renderer = new marked.Renderer();
renderer.html = ({ text }: any) => esc(text);              // model text: no raw HTML, ever
const md = (t: string) => DOMPurify.sanitize(marked.parse(t, { renderer, async: false }) as string);
const plainHeading = (t: string) => esc(t.replace(/^#+\s*/, "").replace(/\*\*/g, "").trim());
const SKELETON = `<div class="skeleton"><i></i><i></i><i></i><i class="short"></i></div>`;
const isTableText = (t: string) => /^\s*\|/.test(t) || /^\s*[-|: ]+$/.test(t);

function el(tag: string, cls: string, html = ""): HTMLElement {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html) e.innerHTML = html;
  return e;
}

export class Reader {
  readonly root: HTMLElement;
  private paper: Paper;
  private parts: Record<string, string> = {};
  private done = new Set<string>();

  constructor(root: HTMLElement, paper: Paper) {
    this.root = root;
    this.paper = paper;
  }

  private pending = false;

  /** While a rewrite is on its way: shimmering placeholders where text will come. */
  setPending(on: boolean) {
    this.pending = on;
    this.root.classList.toggle("pending", on);
    for (const [k, s] of this.secs) {
      if (s.mode === "flowing" && s.right && (s.text === undefined || !s.text.trim())) s.right.innerHTML = on ? SKELETON : "";
      void k;
    }
    this.drawHead();
  }

  private secs = new Map<string, { el: HTMLElement; mode: "flowing" | "aligned"; right?: HTMLElement; text?: string }>();
  private head?: HTMLElement;

  /** Show these parts (finished ones in `done`). Only what changed is redrawn. */
  set(parts: Record<string, string>, done: Iterable<string>) {
    this.parts = { ...parts };
    this.done = new Set(done);
    if (!this.head) this.build();
    this.drawHead();
    for (const k of ORDER) {
      const s = this.secs.get(k);
      if (!s) continue;
      const text = this.parts[k];
      if (text && this.done.has(k)) {
        if (s.mode === "aligned" && s.text === text) continue;
        s.el.innerHTML = "";
        this.label(s.el, k);
        this.aligned(s.el, k, this.paper.sections[k], text);
        Object.assign(s, { mode: "aligned", text, right: undefined });
      } else {
        if (s.mode !== "flowing") {
          s.el.innerHTML = "";
          this.label(s.el, k);
          s.right = this.flowing(s.el, k, this.paper.sections[k]);
          s.mode = "flowing";
        }
        if (s.text !== text) {
          s.text = text;
          s.right!.parentElement!.classList.toggle("writing", text !== undefined);
          s.right!.innerHTML = text?.trim() ? md(text) : this.pending ? SKELETON : "";
        }
      }
    }
  }

  private build() {
    this.root.innerHTML = "";
    this.head = el("header", "paper-head");
    this.root.appendChild(this.head);
    for (const k of ORDER) {
      if (!this.paper.sections[k]) continue;
      const sec = el("section", "sec");
      this.label(sec, k);
      const right = this.flowing(sec, k, this.paper.sections[k]);
      this.secs.set(k, { el: sec, mode: "flowing", right });
      this.root.appendChild(sec);
    }
  }

  private label(sec: HTMLElement, k: string) {
    if (LABEL[k]) sec.appendChild(el("div", "row label", `<div class="o"></div><div class="r"><span>${LABEL[k]}</span></div>`));
  }

  private headKey = "";

  private drawHead() {
    const p = this.paper;
    const plain = this.parts.title?.trim();
    const key = `${plain ?? ""}|${this.pending}`;
    if (key === this.headKey) return;   // redrawing restarts the fade-in: only when it changed
    const arrives = !!plain && !this.headKey.split("|")[0];
    this.headKey = key;
    this.head!.innerHTML = "";
    if (!plain && this.pending) this.head!.appendChild(el("p", "rewriting", "✦ Making it readable…"));
    this.head!.appendChild(el("h1", "plain-title" + (plain ? (arrives ? " arrived" : "") : " pending"), esc(plain || p.title)));
    if (plain) this.head!.appendChild(el("p", "orig-title", `Original title: ${esc(p.title)}`));
    this.head!.appendChild(el("p", "byline", esc([p.authors, p.journal, p.year].filter(Boolean).join(" · "))));
  }

  /** A finished section: paragraph rows, figures and tables in place. */
  private aligned(sec: HTMLElement, k: string, original: string, text: string) {
    const O = original.split("\n\n");
    const groups = align(O, paragraphs(text));
    const marks = this.paper.marks[k] ?? [];
    const at = new Map<number, Mark>(marks.map((m) => [m.at, m]));
    // rewrite paragraphs matched to a heading that aren't headings themselves move to the next row
    let carry: string[] = [];
    const looksLikeHeading = (t: string) => /^#/.test(t) || (t.length < 120 && !/[.:;!?]["”)]?$/.test(t.trim()));
    for (let i = 0; i < O.length;) {
      const m = at.get(i);
      if (m?.kind === "heading") {
        const r = [...carry, ...groups.slice(i, i + m.count).flat()];
        const cut = r.findIndex((x) => !looksLikeHeading(x));
        const heads = cut < 0 ? r : r.slice(0, cut);
        carry = cut < 0 ? [] : r.slice(cut);
        sec.appendChild(el("div", "row heading",
          `<div class="o">${esc(O.slice(i, i + m.count).join(" "))}</div><div class="r">${plainHeading(heads.join(" ")) || esc(O[i])}</div>`));
        i += m.count;
        continue;
      }
      if (carry.length && m) { groups[i] = [...carry, ...groups[i]]; carry = []; }
      if (m?.kind === "figure") {
        sec.appendChild(this.figure(m.images, m.label));
        const r = groups.slice(i, i + m.count).flat().filter((x) => !isTableText(x));
        sec.appendChild(el("div", "row caption",
          `<div class="o">${O.slice(i, i + m.count).map(esc).join("<br>")}</div><div class="r">${md(r.join("\n\n"))}</div>`));
        i += m.count;
        continue;
      }
      if (m?.kind === "table") {
        const cap = groups.slice(i, i + m.captionCount).flat().filter((x) => !isTableText(x));
        sec.appendChild(el("div", "row caption",
          `<div class="o">${O.slice(i, i + m.captionCount).map(esc).join("<br>")}</div><div class="r">${md(cap.join("\n\n"))}</div>`));
        if (m.html) sec.appendChild(el("div", "wide table", m.html));
        else if (m.images.length) sec.appendChild(this.figure(m.images, m.label));
        i += m.count;   // the model's version of the table rows is not shown
        continue;
      }
      const r = [...carry, ...groups[i]].filter((x) => !isTableText(x));
      carry = [];
      sec.appendChild(el("div", "row", `<div class="o">${esc(O[i])}</div><div class="r">${r.length ? md(r.join("\n\n")) : ""}</div>`));
      i++;
    }
  }

  /** A section not yet rewritten (or being written): the original with its figures and tables, the rewrite flowing beside it. */
  private flowing(sec: HTMLElement, k: string, original: string): HTMLElement {
    const O = original.split("\n\n");
    const marks = this.paper.marks[k] ?? [];
    const at = new Map<number, Mark>(marks.map((m) => [m.at, m]));
    const wrap = el("div", "row flowing");
    const left = el("div", "o");
    for (let i = 0; i < O.length;) {
      const m = at.get(i);
      if (m?.kind === "heading") { left.appendChild(el("h4", "", esc(O.slice(i, i + m.count).join(" ")))); i += m.count; continue; }
      if (m && (m.kind === "figure" || m.kind === "table")) {
        if (m.kind === "table" && m.html) left.appendChild(el("div", "table", m.html));
        else left.appendChild(this.figure(m.images, m.label));
        left.appendChild(el("p", "cap", esc(O.slice(i, i + (m.kind === "table" ? m.captionCount : m.count)).join(" "))));
        i += m.count;
        continue;
      }
      left.appendChild(el("p", "", esc(O[i])));
      i++;
    }
    const right = el("div", "r");
    wrap.append(left, right);
    sec.appendChild(wrap);
    return right;
  }

  private figure(images: string[], label: string): HTMLElement {
    const f = el("figure", "wide");
    for (const href of images) f.appendChild(image(this.paper.pmcid, href, label || "Figure"));
    return f;
  }
}
