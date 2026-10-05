import sys, json
sys.path.insert(0, "/home/maxime/Projects/scholarsreadinglist/sciencemadereadable/rewrite_benchmark")
import dpyr, translator
from prepare import prepare_any, prepare_source
from pathlib import Path
X = Path("article-tokens/xml")
papers = []
for pid, f in [("pine", "0300008"), ("psychiatry", "0300004"), ("muscle", "0300006")]:
    p = prepare_any(X / f"{f}.xml")
    row = {"paper_id": pid}
    for s in translator.ALL_SECTIONS:
        row[s] = p["sections"].get(s, "")
    papers.append(row)
res = translator.translate_papers(dpyr.read(papers), "rewrite_benchmark/runs/v7_authors_voice")
print(res.shape)
