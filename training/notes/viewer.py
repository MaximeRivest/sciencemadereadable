"""
One page to read the notes pilot
================================

    python3 training/notes/viewer.py          ->  the latest round: training/notes/out/round<N>/index.html
    python3 training/notes/viewer.py 1        ->  round 1

At the top: the summary. Then, for each paper and part: the judge's verdicts (with reasons), the
notes as the level 1 writer got them, what the notes check found, the four texts side by side
(level 1, level 3, v3, old) and what the trace found in each, and the original at the bottom.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

_rounds = sorted((Path(__file__).resolve().parent / "out").glob("round*"), key=lambda p: int(p.name[5:]))
OUT = _rounds[-1].parent / f"round{sys.argv[1]}" if len(sys.argv) > 1 else _rounds[-1]
LEVELS = ("level1", "level3")
NAMES = {"level1": "From notes · level 1 (with plan)", "level3": "From notes · level 3 (shuffled facts)",
         "v3": "v3 (from the paper, rules)", "old": "Old Opus answer (from the paper)"}
E = html.escape


def md(text: str) -> str:
    """Just enough markdown: headings, lists, tables, paragraphs, bold."""
    out = []
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = block.strip().split("\n")
        if all(l.lstrip().startswith("|") for l in lines):
            rows = [l for l in lines if not re.match(r"^\s*\|[\s:|-]+\|\s*$", l)]
            cells = [[E(c.strip()) for c in r.strip().strip("|").split("|")] for r in rows]
            out.append("<table class=t>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells) + "</table>")
            continue
        for line in lines:
            m = re.match(r"^(#{1,4})\s+(.*)", line)
            if m:
                out.append(f"<h{len(m.group(1)) + 2}>{E(m.group(2))}</h{len(m.group(1)) + 2}>")
        rest = [l for l in lines if not re.match(r"^#{1,4}\s", l)]
        if not rest:
            continue
        if all(re.match(r"^\s*([-*•]|\d+\.)\s", l) for l in rest):
            out.append("<ul>" + "".join(f"<li>{E(re.sub(r'^\s*([-*•]|\d+\.)\s', '', l))}</li>" for l in rest) + "</ul>")
        else:
            out.append(f"<p>{E(' '.join(rest))}</p>")
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", "\n".join(out))


def summary_block(s: dict) -> str:
    rows = "".join(f"<tr><td>{E(pair)}</td><td>{c.get(pair.split(' vs ')[0], 0)}</td><td>{c.get('old', 0)}</td>"
                   f"<td>{c.get('no clear winner', 0)}</td></tr>" for pair, c in s["judge"].items())
    trace = "".join(f"<tr><td>{NAMES[lv]}</td>" + "".join(f"<td>{s['trace'][lv].get(k, 0)}</td>" for k in
                    ("kept", "certainty changed", "changed", "missing", "addition: outside fact", "addition: wrong explanation"))
                    + "</tr>" for lv in LEVELS)
    read = "".join(f"<tr><td>{NAMES[v]}</td><td>{r['words']:.0f}</td><td>{r['sent_mean']}</td><td>{r['sent_over_30']}</td>"
                   f"<td>{r['para_mean']:.0f}</td><td>{r['fk_grade']}</td><td>{s['numbers_kept'][v]:.0%}</td></tr>"
                   for v, r in s["readability"].items())
    notes = ", ".join(f"{k}: {v}" for k, v in sorted(s["notes_check"].items())) or "none"
    if "notes_check_before_fix" in s:
        before = ", ".join(f"{k}: {v}" for k, v in sorted(s["notes_check_before_fix"].items())) or "none"
        notes = f"before the fix: {before}<br>after the fix: {notes}"
    return f"""<div class=card><h2>Summary · {s['parts']} parts</h2>
<h3>Fair blind judge (both orders; a win must hold both ways)</h3>
<table class=t><tr><th>pair</th><th>first wins</th><th>old wins</th><th>no clear winner</th></tr>{rows}</table>
<h3>Did the notes keep the paper? ({s['facts_in_notes']} facts in all notes)</h3><p>{notes}</p>
<h3>Trace: each text back to its notes</h3>
<table class=t><tr><th></th><th>kept</th><th>certainty changed</th><th>changed</th><th>missing</th><th>outside facts added</th><th>wrong explanations</th></tr>{trace}</table>
<h3>Readability</h3>
<table class=t><tr><th></th><th>words</th><th>words/sentence</th><th>% sentences &gt;30 words</th><th>words/paragraph</th><th>grade</th><th>numbers of the original kept</th></tr>{read}</table></div>"""


def part_block(pid: str, p: dict) -> str:
    verdicts = "".join(
        f"<details><summary><b>{E(pair)}</b>: {E(r['winner'])}</summary>"
        f"<p><i>In this order:</i> {E(r['reasons_in_order'])}</p><p><i>Swapped:</i> {E(r['reasons_swapped'])}</p></details>"
        for pair, r in p["judge"].items())
    def listing(problems):
        return "".join(f"<li><b>{E(x['kind'])}</b> ({x['severity']}): {E(x['detail'])} <q>{E(x['quote'][:200])}</q></li>"
                       for x in problems) or "<li>none</li>"
    problems = listing(p["notes_check"])
    if "notes_check_before_fix" in p:
        problems = (f"<li><i>Before the fix ({len(p['notes_check_before_fix'])}):</i><ul>{listing(p['notes_check_before_fix'])}</ul></li>"
                    f"<li><i>After the fix ({len(p['notes_check'])}):</i><ul>{problems}</ul></li>")

    def trace_of(lv):
        t = p["trace"][lv]
        bad = [f"<li>{E(x['id'])} <b>{E(x['status'])}</b>: {E(x['note'])}</li>" for x in t["facts"] if x["status"] != "kept"]
        bad += [f"<li><b>{E(x['kind'])}</b>: <q>{E(x['quote'][:200])}</q> {E(x['note'])}</li>" for x in t["additions"]]
        kept = sum(x["status"] == "kept" for x in t["facts"])
        return f"<div class=trace>Trace: {kept} of {len(t['facts'])} facts kept<ul>{''.join(bad)}</ul></div>"

    cols = "".join(
        f"<div class=col><h4>{NAMES[v]}</h4>"
        + (trace_of(v) if v in LEVELS else "")
        + f"<div class=text>{md(p['texts'][v] if v in LEVELS else p[v])}</div></div>"
        for v in ("level1", "level3", "v3", "old"))
    return f"""<div class=card id="{pid}-{p['section']}"><h2>{E(pid)} · {E(p['section'])}</h2>
<div class=verdicts>{verdicts}</div>
<details><summary>The notes, as the level 1 writer got them ({len(p['notes']['facts'])} facts, {len(p['notes']['definitions'])} definitions)</summary><pre>{E(p['inputs']['level1'])}</pre></details>
<details><summary>The notes, as the level 3 writer got them</summary><pre>{E(p['inputs']['level3'])}</pre></details>
<details><summary>Notes check: {len(p.get('notes_check_before_fix', p['notes_check']))} problems found, {len(p['notes_check'])} left</summary><ul>{problems}</ul></details>
<div class=cols>{cols}</div>
<details><summary>The original</summary><pre>{E(p['original'])}</pre></details></div>"""


def main():
    papers = [json.loads(f.read_text()) for f in sorted(OUT.glob("PMC*.json"))]
    summary = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else None
    toc = "".join(f"<li>{E(d['paper_id'])}: {E(d['title'])} <small>({E(d['subfield'])}{', invented species names' if d['renamed'] else ''})</small> "
                  + " ".join(f"<a href='#{d['paper_id']}-{p['section']}'>{E(p['section'])}</a>" for p in d["parts"]) + "</li>"
                  for d in papers)
    body = (summary_block(summary) if summary else "") + f"<div class=card><h2>Papers</h2><ul>{toc}</ul></div>" + \
        "".join(part_block(d["paper_id"], p) for d in papers for p in d["parts"])
    page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Writing from notes · pilot</title><style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;background:#f6f6f3;color:#222}}
.card{{background:#fff;border-radius:10px;padding:14px 18px;margin:0 auto 16px;max-width:1800px;box-shadow:0 1px 3px #0001}}
.cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px;margin-top:10px}}
.col{{border:1px solid #e3e3e3;border-radius:8px;padding:8px 12px;max-height:80vh;overflow:auto}}
.text{{font:16px/1.6 Georgia,serif}} .text h3,.text h4,.text h5{{font-family:system-ui;margin:.8em 0 .3em}}
.trace{{font-size:13px;background:#faf6ea;border-radius:6px;padding:4px 8px;margin-bottom:8px}}
pre{{white-space:pre-wrap;font-size:13px;background:#f3f3f3;padding:8px;border-radius:6px}}
table.t{{border-collapse:collapse;font-size:14px}} .t td,.t th{{border:1px solid #ddd;padding:3px 8px;text-align:left}}
summary{{cursor:pointer;margin:4px 0}} q{{color:#555}}
@media (prefers-color-scheme:dark){{body{{background:#1b1b1b;color:#ddd}}.card{{background:#252525}}.col{{border-color:#3a3a3a}}
pre{{background:#1e1e1e}}.trace{{background:#33301f}}.t td,.t th{{border-color:#444}}q{{color:#aaa}}}}
</style></head><body><div class=card><h1>Writing from notes · pilot</h1>
<p>Can a writer that never sees the paper write it well for a curious 14-year-old, from notes alone? Here the writer is Opus (the ceiling).
See README.md for the idea and how to read this page.</p></div>{body}</body></html>"""
    (OUT / "index.html").write_text(page)
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
