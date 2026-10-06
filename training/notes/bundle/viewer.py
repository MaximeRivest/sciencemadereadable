"""
One page to read round 4 (the dense bundle)
===========================================

    .venv/bin/python training/notes/bundle/viewer.py [4a]   ->  training/notes/bundle/out/<try>/index.html

Top: the summary. Then for each paper: the whole bundle as the writer got it (folded), and for
each part: the judge's verdicts with reasons, what the bundle check found, the trace, and the
three texts side by side (from the bundle, round 3 level 1, old answer), the original last.
"""
from __future__ import annotations

import html
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_tries = sorted(p for p in (HERE / "out").iterdir() if p.is_dir())
OUT = HERE / "out" / sys.argv[1] if len(sys.argv) > 1 else _tries[-1]   # a try ("4a"), default the latest
_spec = importlib.util.spec_from_file_location("notes_round3_viewer", HERE.parent / "viewer.py")
V3 = importlib.util.module_from_spec(_spec)          # round 3's viewer: markdown and page style
_spec.loader.exec_module(V3)
E, md = html.escape, V3.md
NAMES = {"bundle": "This try · from the dense bundle", "level1": "Round 3 · from notes (level 1)",
         "4b": "Try 4b · from the dense bundle", "4c": "Try 4c · from the dense bundle (text only)",
         "old": "Old Opus answer (from the paper)"}


def shown(text: str) -> str:
    """Placeholders as the reader will see them: a box where the original figure or table goes."""
    return re.sub(r"^\[([FT])([A-Z]?\d+)\][ \t]*$", lambda m: f"⟪{'Table' if m.group(1) == 'T' else 'Figure'} {m.group(2)}: "
                  "the original, shown by the app⟫", text, flags=re.M)


def listing(problems: list[dict]) -> str:
    return "".join(f"<li><b>{E(x['kind'])}</b> ({x['severity']}): {E(x['detail'])} <q>{E(x['quote'][:200])}</q></li>"
                   for x in problems) or "<li>none</li>"


def summary_block(s: dict) -> str:
    fmt = lambda c: ", ".join(f"{k}: {v}" for k, v in sorted(c.items())) or "none"
    judge = "".join(f"<tr><td>{E(pair)}</td><td>{c.get(pair.split(' vs ')[0], 0)}</td><td>{c.get('no clear winner', 0)}</td>"
                    f"<td>{c.get(pair.split(' vs ')[1], 0)}</td></tr>" for pair, c in s["judge"].items())
    d = s["density"]
    nums = s.get("numbers_kept", {})
    read = "".join(f"<tr><td>{NAMES.get(k, k)}</td><td>{r['words']:.0f}</td><td>{r['sent_mean']}</td><td>{r['sent_over_30']}</td>"
                   f"<td>{r['para_mean']:.0f}</td><td>{r['fk_grade']}</td><td>{nums[k]:.0%}</td></tr>" if k in nums else
                   f"<tr><td>{NAMES.get(k, k)}</td><td>{r['words']:.0f}</td><td>{r['sent_mean']}</td><td>{r['sent_over_30']}</td>"
                   f"<td>{r['para_mean']:.0f}</td><td>{r['fk_grade']}</td><td></td></tr>"
                   for k, r in s["readability"].items())
    extra = ""
    if "form" in s:
        extra = (f"<h3>Form, checked by code (this try's texts)</h3><p>{E(fmt(s['form']))}</p>"
                 f"<p>Numbers of the notes found in the text: {s['numbers_of_notes_kept']:.0%}. Readability is measured "
                 f"without placeholders and citation brackets.</p>")
    return f"""<div class=card><h2>Summary · {s['papers']} papers, {s['parts']} parts</h2>
<h3>Fair judge (sees the whole paper; both orders; a win must hold both ways)</h3>
<table class=t><tr><th>pair</th><th>first wins</th><th>no clear winner</th><th>second wins</th></tr>{judge}</table>
<h3>How dense</h3><p>Papers: {d['paper_words']:,} words. Bundles: {d['bundle_words']:,} words, of which
{d['bundle_words'] - d['bundle_words_without_sheets']:,} are the reference sheets (statistics, methods, units) and the rest
is title, abstract, things, glossary, captions, tables and {d['facts']} dense facts.</p>
<h3>Does the bundle hold the paper?</h3><p>Before the fix: {E(fmt(s['check_before_fix']))}<br>After the fix: {E(fmt(s['check']))}</p>
<h3>Trace: items of the bundle in the text</h3><p>{E(fmt(s['trace']))}</p>
{extra}<h3>Readability</h3><table class=t><tr><th></th><th>words</th><th>words/sentence</th><th>% sentences &gt;30 words</th>
<th>words/paragraph</th><th>grade</th><th>numbers of the original kept</th></tr>{read}</table></div>"""


def part_block(pid: str, p: dict) -> str:
    verdicts = "".join(f"<details><summary><b>{E(pair)}</b>: {E(r['winner'])}</summary><p><i>In this order:</i> "
                       f"{E(r['reasons_in_order'])}</p><p><i>Swapped:</i> {E(r['reasons_swapped'])}</p></details>"
                       for pair, r in p["judge"].items())
    t = p["trace"]
    bad = [f"<li>{E(x['id'])} <b>{E(x['status'])}</b>: {E(x['note'])}</li>" for x in t["facts"] if x["status"] != "kept"]
    bad += [f"<li><b>{E(x['kind'])}</b>: <q>{E(x['quote'][:200])}</q> {E(x['note'])}</li>" for x in t["additions"]]
    kept = sum(x["status"] == "kept" for x in t["facts"])
    trace = f"<div class=trace>Trace: {kept} of {len(t['facts'])} items kept<ul>{''.join(bad)}</ul></div>"
    form = ""
    if "form" in p:
        bad = {k: v for k, v in p["form"].items() if v}
        form = f"<div class=trace>Form check: {E(json.dumps(bad, ensure_ascii=False)) if bad else 'all fine'}</div>"
    cols = "".join(f"<div class=col><h4>{NAMES.get(k, k)}</h4>{(form + trace) if k == 'bundle' else ''}"
                   f"<div class=text>{md(shown(p['texts'][k]))}</div></div>" for k in p["texts"])
    d = p["density"]
    return f"""<div class=card id="{pid}-{p['section']}"><h2>{E(pid)} · {E(p['section'])}</h2>
<p><small>This part: {d['original_words']} words in the paper → {d['part_words']} words of dense outline.</small></p>
<div class=verdicts>{verdicts}</div>
<details><summary>Bundle check: {len(p['check_before_fix'])} problems found, {len(p['check'])} left</summary>
<ul><li><i>Before the fix:</i><ul>{listing(p['check_before_fix'])}</ul></li><li><i>After the fix:</i><ul>{listing(p['check'])}</ul></li></ul></details>
<div class=cols>{cols}</div>
<details><summary>The original</summary><pre>{E(p['original'])}</pre></details></div>"""


def main():
    papers = [json.loads(f.read_text()) for f in sorted(OUT.glob("PMC*.json"))]
    summary = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else None
    toc = "".join(f"<li>{E(d['paper_id'])}: {E(d['title'])} <small>({E(d['subfield'])}"
                  f"{', invented species names' if d['renamed'] else ''})</small> <a href='#{d['paper_id']}-bundle'>bundle</a> "
                  + " ".join(f"<a href='#{d['paper_id']}-{p['section']}'>{E(p['section'])}</a>" for p in d["parts"]) + "</li>"
                  for d in papers)
    body = (summary_block(summary) if summary else "") + f"<div class=card><h2>Papers</h2><ul>{toc}</ul></div>"
    for d in papers:
        body += (f"<div class=card id='{d['paper_id']}-bundle'><h2>{E(d['paper_id'])} · the bundle the writer got</h2>"
                 f"<p>{d['density']['paper_words']:,} words of paper → {d['density']['bundle_words']:,} words of bundle "
                 f"({d['density']['bundle_words_without_sheets']:,} without the reference sheets)</p>"
                 f"<details><summary>Show the whole bundle</summary><pre>{E(d['bundle'])}</pre></details></div>")
        body += "".join(part_block(d["paper_id"], p) for p in d["parts"])
    page = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Writing from a dense bundle · round 4</title>
<style>{STYLE}</style></head><body><div class=card><h1>Writing from a dense bundle · round 4</h1>
<p>The writer never sees the paper. It gets one dense, structured bundle per paper: the title, abstract, captions and
tables word for word, plus the paper's things, a glossary, and every other piece of information as dense facts in a
paragraph plan. See ../README.md.</p></div>{body}</body></html>"""
    (OUT / "index.html").write_text(page)
    print(OUT / "index.html")


STYLE = """
body{font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;background:#f6f6f3;color:#222}
.card{background:#fff;border-radius:10px;padding:14px 18px;margin:0 auto 16px;max-width:1800px;box-shadow:0 1px 3px #0001}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;margin-top:10px}
.col{border:1px solid #e3e3e3;border-radius:8px;padding:8px 12px;max-height:80vh;overflow:auto}
.text{font:16px/1.6 Georgia,serif}.text h3,.text h4,.text h5{font-family:system-ui;margin:.8em 0 .3em}
.trace{font-size:13px;background:#faf6ea;border-radius:6px;padding:4px 8px;margin-bottom:8px}
pre{white-space:pre-wrap;font-size:13px;background:#f3f3f3;padding:8px;border-radius:6px}
table.t{border-collapse:collapse;font-size:14px}.t td,.t th{border:1px solid #ddd;padding:3px 8px;text-align:left}
summary{cursor:pointer;margin:4px 0}q{color:#555}
@media (prefers-color-scheme:dark){body{background:#1b1b1b;color:#ddd}.card{background:#252525}.col{border-color:#3a3a3a}
pre{background:#1e1e1e}.trace{background:#33301f}.t td,.t th{border-color:#444}q{color:#aaa}}
"""

if __name__ == "__main__":
    main()
