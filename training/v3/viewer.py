"""One HTML page to inspect the v3 pilot data: for each conversation, what the student
gets (glossary given, entries withheld), the old Opus answer and the grounded answer with
the edits highlighted, how each term was handled, the outside facts removed, the checks.

    python3 training/v3/viewer.py      ->  training/v3/pilot/index.html
"""
import difflib
import sys
import glob
import html
import json
import re
from pathlib import Path

import sys
HERE = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "pilot4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "rewrite_benchmark"))
import readability as RD                                          # noqa: E402

COLORS = {"original paper": "#8a8a8a", "old Opus answer": "#2f6fde", "v3 answer": "#27ae60"}


def ridges(texts: dict[str, str], kind: str, xmin, xmax, ticks, unit, mark=None) -> str:
    W, rowH, top, left, right = 520, 40, 10, 130, 10
    H = top + rowH * len(texts) + 46
    X = lambda v: left + (v - xmin) / (xmax - xmin) * (W - left - right)
    hs = {k: RD.histogram(t, kind) for k, t in texts.items()}
    mx = max(max(y) for _, y in hs.values()) or 1
    out = [f'<svg width="{W}" height="{H}" style="color:inherit">']
    for t in ticks:
        out.append(f'<line x1="{X(t)}" x2="{X(t)}" y1="{top}" y2="{H-36}" stroke="currentColor" stroke-opacity=".12"/>'
                   f'<text x="{X(t)}" y="{H-22}" font-size="11" text-anchor="middle" fill="currentColor" opacity=".7">{t}</text>')
    out.append(f'<text x="{(left+W)/2}" y="{H-6}" font-size="12" text-anchor="middle" fill="currentColor" opacity=".8">{unit}</text>')
    if mark is not None:
        out.append(f'<line x1="{X(mark)}" x2="{X(mark)}" y1="{top}" y2="{H-36}" stroke="currentColor" stroke-dasharray="4,3" stroke-opacity=".6"/>')
    for i, (name, (xs, ys)) in enumerate(hs.items()):
        base = top + rowH * (i + 1) + 4
        pts = [(X(x), base - y / mx * rowH * 2.1) for x, y in zip(xs, ys)]
        d = "M" + f"{pts[0][0]:.1f},{base}" + "".join(f"L{x:.1f},{y:.1f}" for x, y in pts) + f"L{pts[-1][0]:.1f},{base}Z"
        c = COLORS.get(name, "#888")
        out.append(f'<path d="{d}" fill="{c}" fill-opacity=".3" stroke="{c}" stroke-width="1.6"/>'
                   f'<text x="{left-8}" y="{base-3}" font-size="12" text-anchor="end" fill="currentColor">{name}</text>')
    return "".join(out) + "</svg>"


def readability_block(convs: list[dict]) -> str:
    texts = {"original paper": "\n\n".join(c["original"] for c in convs),
             "old Opus answer": "\n\n".join(c["old_answer"] for c in convs),
             "v3 answer": "\n\n".join(c["new_answer"] for c in convs)}
    ms = {k: RD.measure(t) for k, t in texts.items()}
    head = "".join(f"<th>{k}</th>" for k in texts)
    rows = "".join(f"<tr><td>{label}</td>" + "".join(f"<td>{ms[k][f]:.1f}</td>" for k in texts) + "</tr>"
                   for f, label in RD.FIELDS.items() if f != "words")
    return (f'<div class="conv"><h2>Readability, all {len(convs)} conversations</h2>'
            f'<table class="t"><tr><th></th>{head}</tr>{rows}</table>'
            f'<div style="display:flex;gap:24px;flex-wrap:wrap;margin-top:10px">'
            f'<div><b>Sentence length</b>{ridges(texts, "sentence", 0, 72, [0,10,20,30,40,50,60,70], "words per sentence")}</div>'
            f'<div><b>Age words are usually learned</b>{ridges(texts, "word_age", 2, 18, [2,4,6,8,10,12,14,16,18], "years (dashed: reader, 14)", 14)}</div>'
            f'</div><div class="meta">Words of 4+ letters in the Kuperman et al. (2012) list; sentences of 3+ words; '
            f'headings, tables and citation brackets ignored. Markers count as their term.</div></div>')
NAMES = {"opening": "Opening (title, abstract, first intro paragraph, conclusion)", "introduction_rest": "Introduction (rest)",
         "methods": "Methods", "results": "Results", "discussion": "Discussion"}


def esc(t):
    return html.escape(t or "")


def mark(t):
    return re.sub(r"⟦([^⟧]+)⟧", r'<span class="mk">⟦\1⟧</span>', t)


def diff(old, new):
    a, b = re.findall(r"\S+|\n", old), re.findall(r"\S+|\n", new)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(esc(" ".join(a[i1:i2])))
        if op in ("delete", "replace"):
            out.append(f'<del>{esc(" ".join(a[i1:i2]))}</del>')
        if op in ("insert", "replace"):
            out.append(f'<ins>{esc(" ".join(b[j1:j2]))}</ins>')
    return mark(" ".join(out)).replace(" \n ", "<br>").replace("\n", "<br>")


def rd_line(c):
    a, b = RD.measure(c["old_answer"]), RD.measure(c["new_answer"])
    return ("readability old → new: " + f"{a['sent_mean']:.1f} → {b['sent_mean']:.1f} words/sentence · "
            f"{a['sent_over_30']:.0f}% → {b['sent_over_30']:.0f}% sentences over 30 words · "
            f"{a['age_over_reader']:.1f}% → {b['age_over_reader']:.1f}% words learned at 14+ · grade {a['fk_grade']:.1f} → {b['fk_grade']:.1f}")


def page():
    parts = []
    toc = []
    allconv = [c for f in sorted(glob.glob(str(HERE / "PMC*.json"))) for c in json.load(open(f))["conversations"]]
    for f in sorted(glob.glob(str(HERE / "PMC*.json"))):
        r = json.load(open(f))
        pid = r["paper_id"]
        toc.append(f'<a href="#{pid}">{esc(r["title"][:80])}</a>')
        doi = r.get("doi") or ""
        parts.append(f'<section id="{pid}"><h1>{esc(r["title"])}</h1><div class="meta">{esc(r.get("journal"))} {r.get("year") or ""} · '
                     f'<a href="{doi if doi.startswith("http") else "https://doi.org/" + doi}">{esc(doi)}</a> · CC BY 4.0, © the authors; '
                     f'shown here with the training-data edits made by our pipeline.<br>{len(r["terms"])} candidate terms · '
                     f'{r["entries"]} glossary entries · {len(r["no_reference"])} content terms without a reference (always markers)</div>')
        if r["renamed"]:
            rows = "".join(f'<tr><td>{esc(x["real"])}</td><td>{esc(", ".join(f"{k} → {v}" for k, v in x["forms"].items()))}</td>'
                           f"<td>{esc(x.get('entry', '') or x['kind'])}</td></tr>" for x in r["renamed"])
            parts.append(f'<details open><summary><b>Invented names</b> ({len(r["renamed"])})</summary><table class="t">'
                         f'<tr><th>real</th><th>replaced by</th><th>glossary entry given</th></tr>{rows}</table></details>')
        for c in r["conversations"]:
            k = c["checks"]
            given = "".join(f'<li><b>{esc(e["term"])}</b>{" (" + esc(e.get("stands_for")) + ")" if e.get("stands_for") else ""}'
                            f'{" <i>[" + esc(e.get("category")) + "]</i>" if e.get("category") else ""}: {esc(e["reference"][:220])}</li>'
                            for e in c["glossary_given"])
            terms = "".join(f'<tr class="h-{t["handling"]}"><td>{esc(t["term"])}</td><td>{t["handling"]}</td><td>{", ".join(t["sources"])}</td>'
                            f'<td>{mark(esc(t["explanation"]))}</td></tr>' for t in c["terms"])
            removed = "".join(f"<li>{esc(x)}</li>" for x in c["removed_outside_facts"])
            warn = f' · <span class="bad">broken markers: {esc("; ".join(k.get("marker_problems", [])))}</span>' if k.get("marker_problems") else ""
            parts.append(f'''<div class="conv"><h2>{NAMES.get(c["section"], c["section"])}</h2>
<div class="meta">input: <b>{c["dropout"]}</b> · {len(c["glossary_given"])} entries given · {len(c["no_entry_terms"])} terms without entry ·
sentences kept as they were {k["sentences_unchanged"]:.0%} · length ×{k["length_ratio"]:.2f} · numbers kept {k["numbers_kept_old"]:.0%} → {k["numbers_kept_new"]:.0%} ·
{len(k["markers"])} markers{warn}</div>
<div class="meta">{rd_line(c)}</div>
<details><summary>Glossary given to the student ({len(c["glossary_given"])})</summary><ul class="g">{given or "<li>none</li>"}</ul></details>
<details><summary>Terms with no entry: must become markers ({len(c["no_entry_terms"])})</summary><p>{esc(", ".join(c["no_entry_terms"])) or "none"}</p></details>
<details><summary>Original text (what the student rewrites)</summary><div class="orig">{esc(c["original"]).replace(chr(10), "<br>")}</div></details>
<h3>Training answer: old Opus answer → v3 answer <span class="legend"><del>removed</del> <ins>added</ins> <span class="mk">⟦marker⟧</span></span></h3>
<div class="ans">{diff(c["old_answer"], c["new_answer"])}</div>
<details><summary>How each term was handled ({len(c["terms"])})</summary><table class="t"><tr><th>term</th><th>handling</th><th>sources</th><th>explanation as written</th></tr>{terms}</table></details>
<details><summary>Outside facts removed ({len(c["removed_outside_facts"])})</summary><ul>{removed or "<li>none</li>"}</ul></details></div>''')
        parts.append("</section>")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Training set v3 pilot</title><style>
:root{{--bg:#fbfaf6;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e0d8;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141413;--fg:#ecebe6;--muted:#9a9890;--line:#33322f;--card:#1e1e1c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}} main{{max-width:1150px;margin:0 auto;padding:20px}}
h1{{font-size:20px;margin:36px 0 4px}} h2{{font-size:16px;margin:0 0 4px}} h3{{font-size:13px;margin:12px 0 4px;color:var(--muted)}}
.meta{{color:var(--muted);font-size:13px;margin-bottom:6px}} .conv{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}}
.ans{{font-family:Georgia,serif;font-size:15.5px;line-height:1.65}} del{{background:#f6d5d1;color:#7a2a20;text-decoration:line-through}}
ins{{background:#d3efd8;color:#1e5a2b;text-decoration:none}} @media (prefers-color-scheme:dark){{del{{background:#4a2420;color:#f0b8b0}} ins{{background:#1f3d26;color:#b8e8c2}}}}
.mk{{background:#ffe9a8;color:#5a4300;border-radius:3px;padding:0 2px;font-weight:600}} @media (prefers-color-scheme:dark){{.mk{{background:#5a4a10;color:#ffe9a8}}}}
.legend{{font-weight:400;margin-left:10px}} .legend *{{margin-right:6px}} summary{{cursor:pointer;color:var(--muted);font-size:13px;margin-top:6px}}
.t{{border-collapse:collapse;font-size:13px;width:100%}} .t td,.t th{{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top}}
.g{{font-size:13px}} .orig{{font-size:13px;color:var(--muted);max-height:300px;overflow:auto}} .bad{{color:#c0392b;font-weight:600}}
body.final del{{display:none}} body.final ins{{background:none;color:inherit}}
.h-marker td:nth-child(2){{color:#a07800;font-weight:600}} .toc a{{display:block}}</style></head><body><main>
<h1 style="margin-top:0">Training set v3, pilot: 3 papers, 15 conversations</h1>
<div class="meta">Each answer is the old Opus training answer edited so that every fact comes from the paper, the glossary given, or what a 12-year-old knows.
Terms without a glossary entry become markers ⟦term⟧. Per conversation, a random share of entries is withheld from the input ("dropout"), or all of them ("no glossary").
In the lizard paper, species get invented names. Not yet reviewed by a person.</div>
<div class="toc">{"".join(toc)}</div>
<p><label><input type="checkbox" id="final" onchange="document.body.classList.toggle('final', this.checked)"> show the final training answers only (hide removed words and highlights)</label></p>
{readability_block(allconv)}{"".join(parts)}</main></body></html>'''


(HERE / "index.html").write_text(page())
print("wrote", HERE / "index.html")
