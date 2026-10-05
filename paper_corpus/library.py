"""Build a small browsable library of the rewritten papers (static HTML).

    .venv/bin/python paper_corpus/library.py      # rebuild (a few seconds)

Then open paper_corpus/library/index.html, or serve it:
    .venv/bin/python -m http.server 8765 --directory paper_corpus/library
"""
import html
import json
from pathlib import Path

import dpyr

HERE = Path(__file__).resolve().parent
OUT = HERE / "library"
RUNS = {"Opus": HERE / "rewrites" / "rewrites", "Astra": HERE / "rewrites_astra" / "rewrites"}
ORDER = ["title", "abstract", "introduction_first", "introduction_rest", "methods", "results", "discussion", "conclusion"]
NAMES = {"title": "Title", "abstract": "Abstract", "introduction_first": "Introduction (first paragraph)",
         "introduction_rest": "Introduction (rest)", "methods": "Methods", "results": "Results",
         "discussion": "Discussion", "conclusion": "Conclusion"}

CSS = """
body{font:16px/1.6 system-ui,sans-serif;margin:0;color:#1d2521;background:#fbfbf8}
header{padding:18px 28px;border-bottom:1px solid #ddd;background:#fff;position:sticky;top:0}
h1{font-size:22px;margin:0} a{color:#1f6f64} .dim{color:#6b756f;font-size:14px}
main{max-width:1500px;margin:auto;padding:20px 28px}
input{font:inherit;padding:8px 12px;width:100%;max-width:520px;border:1px solid #ccc;border-radius:8px;margin-top:10px}
table{border-collapse:collapse;width:100%} td,th{padding:8px 10px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
th{font-size:13px;color:#6b756f} tr:hover{background:#f1f5f2}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:26px 0;border-top:2px solid #e3e6e1;padding-top:14px}
.pair h3{grid-column:1/-1;margin:0 0 4px;font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:#1f6f64}
.col{white-space:pre-wrap;font-size:15px} .orig{color:#4b554f;font-size:14px}
.label{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#8a948e;margin-bottom:4px}
.toggle{float:right} .only-new .orig{display:none} .only-new .pair{grid-template-columns:1fr} .only-new .pair{max-width:780px}
@media(max-width:900px){.pair{grid-template-columns:1fr}}
@media(prefers-color-scheme:dark){body{background:#151a18;color:#e6ece8}header{background:#1c2320;border-color:#333}
tr:hover{background:#222b27}td,th{border-color:#2a332f}.orig{color:#aab4ae}.pair{border-color:#2f3833}}
"""


def load():
    papers = {p["paper_id"]: p for p in dpyr.read_parquet(HERE / "papers.parquet").to_dicts()}
    rewrites = {}
    for writer, folder in RUNS.items():
        for f in folder.glob("*.parquet"):   # one file at a time: column types can differ
            for r in dpyr.read_parquet(f).to_dicts():
                rewrites.setdefault(r["paper_id"], {"writer": writer, "sections": {}})["sections"][r["section"]] = r
    return papers, rewrites


def paper_page(pid, meta, rw):
    sec = rw["sections"]
    title_new = html.escape(sec.get("title", {}).get("rewrite", meta["title"]))
    parts = []
    for s in ORDER:
        if s not in sec:
            continue
        r = sec[s]
        parts.append(f"<section class='pair'><h3>{NAMES[s]}</h3>"
                     f"<div><div class='label'>Rewritten ({rw['writer']})</div><div class='col'>{html.escape(r['rewrite'])}</div></div>"
                     f"<div class='orig'><div class='label'>Original</div><div class='col'>{html.escape(r['original'])}</div></div></section>")
    doi = meta.get("doi") or ""
    return f"""<!doctype html><meta charset=utf-8><title>{title_new}</title><style>{CSS}</style>
<header><a href='index.html'>← Library</a>
<button class='toggle' onclick="document.body.classList.toggle('only-new')">Show / hide original</button>
<h1>{title_new}</h1>
<div class='dim'>Original title: {html.escape(meta['title'])}<br>{html.escape(meta['journal'] or '')} · {meta['year']} ·
{html.escape(meta['subfield'])} · rewritten by {rw['writer']} ·
<a href='https://doi.org/{html.escape(doi)}' target=_blank>doi:{html.escape(doi)}</a> · CC BY</div></header>
<main>{''.join(parts)}</main>"""


def main():
    papers, rewrites = load()
    (OUT / "papers").mkdir(parents=True, exist_ok=True)
    rows = []
    for pid, rw in sorted(rewrites.items(), key=lambda kv: papers.get(kv[0], {}).get("title", "")):
        meta = papers.get(pid)
        if not meta:
            continue
        (OUT / "papers" / f"{pid}.html").write_text(paper_page(pid, meta, rw))
        new_title = rw["sections"].get("title", {}).get("rewrite", meta["title"])
        rows.append(f"<tr data-s=\"{html.escape((new_title + ' ' + meta['title'] + ' ' + (meta['journal'] or '') + ' ' + meta['subfield']).lower())}\">"
                    f"<td><a href='papers/{pid}.html'>{html.escape(new_title)}</a><div class='dim'>{html.escape(meta['title'])}</div></td>"
                    f"<td>{html.escape(meta['subfield'])}</td><td>{html.escape(meta['journal'] or '')}</td>"
                    f"<td>{meta['year']}</td><td>{rw['writer']}</td></tr>")
    (OUT / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Rewritten papers</title><style>{CSS}</style>
<header><h1>Rewritten papers</h1><div class='dim'>{len(rows):,} papers, rewritten for curious 12–14-year-olds · original texts CC BY</div>
<input placeholder='Search title, journal or topic…' oninput="for(const r of document.querySelectorAll('tbody tr'))r.style.display=r.dataset.s.includes(this.value.toLowerCase())?'':'none'"></header>
<main><table><thead><tr><th>Title (rewritten, then original)</th><th>Topic</th><th>Journal</th><th>Year</th><th>Writer</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></main>""")
    print(f"{len(rows)} papers → {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
