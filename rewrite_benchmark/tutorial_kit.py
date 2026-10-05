"""Loading and display helpers for notebooks/rewriting-tutorial.md.

Kept out of the notebook so each tutorial cell stays short and readable.
Everything here reads saved files; nothing calls a model.
"""
from __future__ import annotations

import builtins
import html
import json
import numbers
import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
LAB = PROJECT / "rewrite_benchmark"
RUN = LAB / "runs/pine-pilot-v1/replicate-0"

ORDER = ["title", "abstract", "introduction_first", "introduction_rest",
         "methods", "results", "discussion", "conclusion"]
STEPS = ["opening", "introduction", "methods", "results", "discussion"]
NAMES = {"conversation_reference": "Conversation", "astra": "Astra", "luna": "Luna"}
COLORS = {"conversation_reference": "#8a8f98", "astra": "#2f6fb0", "luna": "#c07a2c"}
NAME_COLORS = {NAMES[k]: c for k, c in COLORS.items()}           # keyed "Astra", "Luna", ...
MODEL_COLORS = {"gpt-6-astra": COLORS["astra"], "gpt-6-luna": COLORS["luna"]}


# ---------------------------------------------------------------- loading

def load_paper():
    return json.loads((LAB / "data/paper.json").read_text())


def load_candidates():
    """The three rewrites, as {id: {"segments": {...}, "notes": [...], "model": ...}}."""
    pairs = [json.loads(l) for l in (LAB / "data/reference_pairs.jsonl").read_text().splitlines()]
    out = {"conversation_reference": {
        "model": "gpt-6-astra (our conversation, with tools and feedback)",
        "segments": {p["section_id"]: p["rewrite_text"] for p in pairs},
        "notes": list(dict.fromkeys(p["editorial_notes"] for p in pairs if p["editorial_notes"])),
    }}
    for label in ("astra", "luna"):
        path = RUN / "candidates" / f"{label}.json"
        if path.exists():
            c = json.loads(path.read_text())
            notes = [n for section in c["notes"].values() for n in section]
            out[label] = {"model": c["writer_model"].split(":", 1)[-1],
                          "segments": c["segments"], "notes": list(dict.fromkeys(notes)),
                          "glossary": c.get("glossary", [])}
    return out


def load_calls():
    """One row per saved model call: what program, which model, time, tokens."""
    rows = []
    for path in sorted((RUN / "calls").glob("*.json")):
        r = json.loads(path.read_text())
        usage = r.get("usage") or {}
        rows.append({
            "program": r["program"], "model": r["model"].split(":", 1)[-1], "status": r["status"],
            "kind": "write" if r["program"].startswith("rewrite_") else "judge",
            "step": r["program"].replace("rewrite_", "").replace("evaluate_", ""),
            "minutes": (r.get("elapsed_seconds") or 0) / 60,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": usage.get("reasoning_tokens"),
        })
    return pd.DataFrame(rows)


def load_judgments():
    """One row per judgment, plus the issues each judge listed."""
    raw = json.loads((RUN / "assessments.json").read_text())
    rows, issues = [], []
    for r in raw:
        rows.append({k: r.get(k) for k in ("candidate", "judge", "section", "aspect", "status", "score")}
                    | {"problems": "; ".join(r.get("validation_problems") or [])})
        if r["status"] == "ok":
            for i in r["assessment"]["issues"]:
                issues.append({"candidate": r["candidate"], "judge": r["judge"], "section": r["section"],
                               "aspect": r["aspect"], "severity": i["severity"], "explanation": i["explanation"],
                               "source_quote": i["source_quote"], "rewrite_quote": i["rewrite_quote"]})
    return pd.DataFrame(rows), pd.DataFrame(issues), raw


# ---------------------------------------------------------------- display

class HTML:
    def __init__(self, body):
        self.body = body

    def _repr_html_(self):
        return CSS + self.body


CSS = """<style>
.tk{font-family:system-ui,sans-serif;font-size:14px;line-height:1.5;color:inherit}
.tk h4{margin:14px 0 6px;font-size:15px}
.tk table{border-collapse:collapse;margin:6px 0 12px}
.tk th,.tk td{border:1px solid rgba(127,127,127,.35);padding:4px 9px;text-align:left;vertical-align:top}
.tk th{background:rgba(127,127,127,.12);font-weight:600}
.tk td.n{text-align:right;font-variant-numeric:tabular-nums}
.tk .note{font-size:12.5px;opacity:.75;margin:2px 0 10px}
.tk .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.tk .card{border:1px solid rgba(127,127,127,.35);border-radius:8px;padding:10px 14px}
.tk .card h4{margin-top:2px}
.tk .text{white-space:pre-wrap}
.tk .bar{height:14px;border-radius:3px;display:inline-block;vertical-align:middle}
.tk .issue{border-left:4px solid #c0504d;padding:4px 10px;margin:8px 0;background:rgba(192,80,77,.06)}
.tk .issue.minor{border-color:#d9a441;background:rgba(217,164,65,.07)}
.tk .tag{display:inline-block;font-size:11.5px;padding:0 6px;border-radius:9px;background:rgba(127,127,127,.18);margin-right:4px}
.tk .big{font-size:26px;font-weight:700;line-height:1.1}
</style>"""


def show(*parts):
    """Show HTML fragments as one output block (uses rat's display)."""
    getattr(builtins, "display", print)(HTML("<div class='tk'>" + "".join(parts) + "</div>"))


def esc(value):
    return html.escape(str(value))


def title(text, note=None):
    return f"<h4>{esc(text)}</h4>" + (f"<p class='note'>{esc(note)}</p>" if note else "")


def table(df, heading=None, note=None, digits=2, max_chars=220):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" · ".join(map(str, c)) for c in df.columns]
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = ""
    for _, row in df.iterrows():
        cells = ""
        for v in row:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                cells += "<td class='n'>—</td>"
            elif isinstance(v, numbers.Number) and not isinstance(v, bool):
                txt = f"{int(v):,}" if float(v).is_integer() else f"{v:,.{digits}f}"
                cells += f"<td class='n'>{txt}</td>"
            else:
                s = str(v)
                cells += f"<td>{esc(s if len(s) <= max_chars else s[:max_chars] + '…')}</td>"
        body += f"<tr>{cells}</tr>"
    return (title(heading) if heading else "") + f"<table><tr>{head}</tr>{body}</table>" + \
        (f"<p class='note'>{esc(note)}</p>" if note else "")


def bars(values, heading=None, note=None, unit="", colors=None, fmt="{:,.0f}"):
    """Horizontal bars: values is {label: number}. Starts at zero."""
    top = max([v for v in values.values() if v == v] + [1e-9])
    rows = ""
    for label, v in values.items():
        color = (colors or {}).get(label, "#2f6fb0")
        width = 0 if v != v else 320 * v / top
        rows += (f"<tr><td style='border:0;padding:3px 8px 3px 0'>{esc(label)}</td>"
                 f"<td style='border:0;padding:3px 0'><span class='bar' style='width:{width:.0f}px;background:{color}'></span>"
                 f" <span style='font-variant-numeric:tabular-nums'>{fmt.format(v)}{esc(unit)}</span></td></tr>")
    return (title(heading) if heading else "") + f"<table>{rows}</table>" + \
        (f"<p class='note'>{esc(note)}</p>" if note else "")


def heat(df, heading=None, note=None, lo=0, hi=3, fmt="{:.1f}"):
    """A table whose cells are shaded red (low) to green (high)."""
    head = "<th></th>" + "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = ""
    for idx, row in df.iterrows():
        body += f"<tr><th>{esc(idx)}</th>"
        for v in row:
            if v is None or pd.isna(v):
                body += "<td class='n' style='background:rgba(127,127,127,.12)'>—</td>"
            else:
                t = max(0, min(1, (v - lo) / (hi - lo)))
                body += f"<td class='n' style='background:hsl({120 * t:.0f},60%,78%);color:#111'>{fmt.format(v)}</td>"
        body += "</tr>"
    return (title(heading) if heading else "") + f"<table>{head and '<tr>' + head + '</tr>'}{body}</table>" + \
        (f"<p class='note'>{esc(note)}</p>" if note else "")


def text_box(heading, text):
    """A heading and a block of plain text, line breaks kept."""
    return title(heading) + f"<div class='text'>{esc(text)}</div>"


def bullet_list(heading, items, numbered=False):
    """A heading and a list of short texts."""
    tag = "ol" if numbered else "ul"
    lines = "".join(f"<li>{esc(item)}</li>" for item in items) or "<li>(none)</li>"
    return title(heading) + f"<{tag}>{lines}</{tag}>"


def judgment_card(judgment):
    """One judge's verdict: the score, its summary, and every problem it listed."""
    a = judgment["assessment"]
    head = (f"Judge {judgment['judge']} on {NAMES.get(judgment['candidate'], judgment['candidate'])}"
            f" · {judgment['section']} · {judgment['aspect']}")
    boxes = "".join(issue_box({**i, "candidate": judgment["candidate"], "judge": judgment["judge"],
                                "section": judgment["section"], "aspect": judgment["aspect"]})
                    for i in a["issues"])
    return title(head) + f"<div class='big'>{a['score']} / 3</div><p>{esc(a['summary'])}</p>" + boxes


def side_by_side(columns):
    """columns: list of (heading, text). Each becomes a card; cards wrap on narrow screens."""
    return "<div class='grid'>" + "".join(
        f"<div class='card'><h4>{esc(h)}</h4><div class='text'>{esc(t)}</div></div>" for h, t in columns) + "</div>"


def issue_box(row):
    cls = "issue" if row["severity"] in ("major", "critical") else "issue minor"
    quotes = ""
    if row.get("source_quote"):
        quotes += f"<br><i>Paper says:</i> “{esc(row['source_quote'])}”"
    if row.get("rewrite_quote"):
        quotes += f"<br><i>Rewrite says:</i> “{esc(row['rewrite_quote'])}”"
    return (f"<div class='{cls}'><span class='tag'>{esc(row['severity'])}</span>"
            f"<span class='tag'>{esc(NAMES.get(row['candidate'], row['candidate']))}</span>"
            f"<span class='tag'>{esc(row['section'])}</span><span class='tag'>{esc(row['aspect'])}</span>"
            f"<span class='tag'>judge: {esc(row['judge'])}</span><br>{esc(row['explanation'])}{quotes}</div>")


# ---------------------------------------------------------------- the paper's known traps

# name -> ("any" or "all", keyword patterns)
TRAPS = {
    "Abstract calls 125 vs 89 'pine nuts'; Table 4 says seeds in shells": ("all", [r"abstract", r"\b125\b"]),
    "The printed damage formula is broken": ("all", [r"formula", r"damage|\bds\b"]),
    "Results text swaps 2.60% and 4.04% (Figure 1 is right)": ("all", [r"2\.60", r"4\.04"]),
    "Figure 1 uses 328 cones, not all 560": ("any", [r"\b328\b"]),
}


def trap_check(notes):
    """For each known error in the paper: do the writer's notes mention it?
    A keyword check, not proof the error was handled correctly."""
    text = " ".join(notes).lower()
    found = {}
    for trap, (mode, patterns) in TRAPS.items():
        hits = [bool(re.search(p, text)) for p in patterns]
        found[trap] = all(hits) if mode == "all" else any(hits)
    return found
