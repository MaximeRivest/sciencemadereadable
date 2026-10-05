"""A two-column HTML page: the original paper (left) and a model's rewrite (right),
aligned section by section and paragraph by paragraph.

    .venv/bin/python training/side_by_side.py CANDIDATE PAPER_ID OUT.html "model description"

Paragraph alignment: each original paragraph is matched to a run of consecutive
rewrite paragraphs (the rewrite may split a paragraph, never reorders), chosen by
dynamic programming to maximise the overlap of numbers and content words.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import dpyr

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = [("title", "Title"), ("abstract", "Abstract"), ("introduction_first", "Introduction (first paragraph)"),
            ("introduction_rest", "Introduction (rest)"), ("methods", "Methods"), ("results", "Results"),
            ("discussion", "Discussion"), ("conclusion", "Conclusion")]
STOP = set("the a an of and or in on at to for from by with is are was were be been this that these those we our it its as "
           "than which who not no but if so such into over under between among also can may".split())


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def bag(text: str) -> set[str]:
    nums = set(re.findall(r"\d+(?:[.,]\d+)?", text))
    words = {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}
    return nums | words


def sim(a: str, bs: list[str]) -> float:
    if not bs:
        return 0.0
    A, B = bag(a), bag(" ".join(bs))
    if not A or not B:
        return 0.05
    return len(A & B) / len(A | B)


def align(orig: list[str], new: list[str]) -> list[list[str]]:
    """groups[i] = the rewrite paragraphs for original paragraph i (contiguous, in order)."""
    n, m = len(orig), len(new)
    NEG = -1e9
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(m + 1):
            for k in range(max(0, j - 6), j + 1):          # at most 6 rewrite paragraphs per original one
                if best[i - 1][k] == NEG:
                    continue
                v = best[i - 1][k] + sim(orig[i - 1], new[k:j]) - (0.02 if j == k else 0)
                if v > best[i][j]:
                    best[i][j], back[i][j] = v, k
    groups, j = [], m
    for i in range(n, 0, -1):
        k = back[i][j]
        groups.append(new[k:j])
        j = k
    groups.reverse()
    if j > 0:                                               # rewrite paragraphs before the first match
        groups[0] = new[:j] + groups[0]
    return groups


def render(text: str) -> str:
    t = html.escape(text)
    if t.lstrip().startswith("|"):                          # a Markdown table: keep it monospaced
        return f'<pre class="table">{t}</pre>'
    t = re.sub(r"^#+\s*(.+)$", r"<b>\1</b>", t, flags=re.M)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    return f"<p>{t.replace(chr(10), '<br>')}</p>"


def main():
    cand, pid, out, model = sys.argv[1:5]
    paper = [p for p in dpyr.read_parquet(ROOT / "paper_corpus/papers.parquet").to_dicts()
             if p["paper_id"] == pid][0]
    rw = {r["section"]: r["rewrite"] or "" for r in dpyr.read_parquet(
        ROOT / f"model_baselines/rewrites/{cand}/rewrites/{pid}.parquet").to_dicts()}
    rows = []
    words_o = words_n = 0
    for key, name in SECTIONS:
        if not paper.get(key):
            continue
        o, n = paragraphs(paper[key]), paragraphs(rw.get(key, ""))
        words_o += len((paper[key] or "").split())
        words_n += len(rw.get(key, "").split())
        rows.append(f'<tr class="sec"><td colspan="2"><h2>{name}</h2></td></tr>')
        for op, group in zip(o, align(o, n)):
            right = "".join(render(g) for g in group) or '<p class="missing">(nothing matched here)</p>'
            rows.append(f"<tr><td>{render(op)}</td><td>{right}</td></tr>")
    doi = paper.get("doi") or ""
    doi_url = doi if doi.startswith("http") else f"https://doi.org/{doi}"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(paper['title'])}: side by side</title>
<style>
 :root {{ --bg:#fbfaf6; --fg:#1d1d1b; --muted:#6b6b66; --line:#e3e0d8; --left:#f3f1ea; --right:#fff; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#141413; --fg:#ecebe6; --muted:#9a9890; --line:#33322f; --left:#1b1b19; --right:#202020; }} }}
 body {{ margin:0; background:var(--bg); color:var(--fg); font:16px/1.6 Georgia, "Iowan Old Style", serif; }}
 header {{ max-width:1400px; margin:0 auto; padding:24px 20px 8px; font-family:system-ui, sans-serif; }}
 header h1 {{ font:600 22px/1.3 system-ui, sans-serif; margin:0 0 6px; }}
 .meta {{ color:var(--muted); font-size:14px; }}
 table {{ width:100%; max-width:1400px; margin:0 auto 60px; border-collapse:collapse; table-layout:fixed; }}
 thead th {{ position:sticky; top:0; background:var(--bg); font:600 14px system-ui, sans-serif; text-align:left; padding:10px 16px;
            border-bottom:2px solid var(--line); z-index:1; }}
 td {{ vertical-align:top; padding:4px 18px; border-bottom:1px solid var(--line); }}
 td:first-child {{ background:var(--left); color:var(--muted); font-size:15px; }}
 td:last-child {{ background:var(--right); }}
 tr.sec td {{ background:var(--bg) !important; border-bottom:none; }}
 h2 {{ font:600 18px system-ui, sans-serif; margin:28px 0 4px; }}
 p {{ margin:10px 0; }}
 pre.table {{ font-size:12px; white-space:pre-wrap; }}
 .missing {{ color:#b04030; font-style:italic; }}
</style></head><body>
<header><h1>{html.escape(paper['title'])}</h1>
<div class="meta">{html.escape(paper.get('journal') or '')} {paper.get('year') or ''} · <a href="{doi_url}">{html.escape(doi)}</a> ·
licensed <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>; original text © the authors.<br>
Right column: the same paper rewritten for a curious 12–14-year-old by <b>{html.escape(model)}</b> (our student model, not reviewed by a person;
this is an adaptation, with changes). Original {words_o:,} words · rewrite {words_n:,} words.</div></header>
<table><thead><tr><th>Original</th><th>Rewrite</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table></body></html>"""
    Path(out).write_text(page)
    print(f"wrote {out}: {len(rows)} rows")


if __name__ == "__main__":
    main()
