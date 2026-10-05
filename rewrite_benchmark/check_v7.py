import sys, json, re
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import dpyr, translator
from prepare import prepare_any
from pathlib import Path
X = Path("article-tokens/xml"); R = Path("rewrite_benchmark/runs/pine-pilot-v1/replicate-0")
new = translator.load_results("rewrite_benchmark/runs/v7_authors_voice").collect().to_dicts()
v7 = {}
for r in new: v7.setdefault(r["paper_id"], {})[r["section"]] = r["rewrite"]
# earlier Opus opening-only rewrites (v6 brief)
old = {}
ctx = json.load(open(R/"candidates/ctx_opus_opening_only.json"))["segments"]; old["pine"] = ctx
ctx2 = json.load(open(R/"candidates/ctx2.json"))
for pid in ("psychiatry","muscle"): old[pid] = ctx2[f"{pid}/opus/opening_only"]["segments"]
orig = {pid: prepare_any(X/f"{f}.xml")["sections"] for pid,f in [("pine","0300008"),("psychiatry","0300004"),("muscle","0300006")]}

def subheadings(text):
    # original: short lines without final punctuation; rewrite: markdown headings or bold-only lines
    return [l.strip() for l in text.split("\n\n") if l.strip() and len(l.split()) <= 8 and not l.strip().endswith((".",":")) and not l.strip().startswith("|")]
def md_headings(text):
    return [l for l in text.splitlines() if re.match(r"\s*#{1,6}\s", l)]
def third(text): return len(re.findall(r"\b(the authors|the researchers|the study (found|shows|says)|this study (found|shows|says)|the paper (says|reports|shows))\b", text, re.I))
def we(text): return len(re.findall(r"\b(we|our|us)\b", text, re.I))
def bullets(text): return sum(bool(re.match(r"\s*([-*]|\d+\.)\s", l)) for l in text.splitlines())
def paras(text): return len([p for p in text.split("\n\n") if p.strip() and not p.strip().startswith(("#","|"))])

for pid in ("pine","psychiatry","muscle"):
    print("=====", pid)
    for s in translator.OTHER_SECTIONS:
        if s not in v7[pid]: continue
        o = orig[pid][s]
        print(f"{s:<18} orig subheads {len(subheadings(o)):2d} | v6 md-heads {len(md_headings(old[pid].get(s,''))):2d} v7 md-heads {len(md_headings(v7[pid][s])):2d}"
              f" | paras orig {paras(o):3d} v6 {paras(old[pid].get(s,'')):3d} v7 {paras(v7[pid][s]):3d}"
              f" | 3rd-person v6 {third(old[pid].get(s,'')):2d} v7 {third(v7[pid][s]):2d} | we v6 {we(old[pid].get(s,'')):3d} v7 {we(v7[pid][s]):3d}"
              f" | bullets orig {bullets(o):2d} v6 {bullets(old[pid].get(s,'')):3d} v7 {bullets(v7[pid][s]):3d}")
    print("original subheadings, methods:", subheadings(orig[pid]["methods"])[:12])
    print("v7 headings, methods:", md_headings(v7[pid]["methods"])[:12])
