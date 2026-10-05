"""The training data of the paper-rewriter models, ready for Hugging Face, with the writing
models anonymised (writer_a, writer_b) and every source paper credited.

    .venv/bin/python training/hf/make_dataset.py      (repository root, main venv)

Writes training/hf/dataset/: data/train.parquet, data/validation.parquet,
papers.parquet. Each example is one chat: system, user (the paper or section, its
reference glossary, the writer label), assistant (the rewrite), exactly as trained.
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "dataset"
X = ROOT / "paper_corpus/xml"
LABEL = {"opus": "writer_a", "astra": "writer_b"}

t = pd.read_parquet(ROOT / "training/data/examples-glossary.parquet")
papers = pd.read_parquet(ROOT / "paper_corpus/papers.parquet",
                         columns=["paper_id", "doi", "journal", "year", "title"])
papers = papers[papers.paper_id.isin(set(t.paper_id))].copy()


def authors_and_license(pid):
    s = (X / f"{pid}.xml").read_text(errors="ignore")
    front = s[:s.find("</front>")] if "</front>" in s else s
    names = []
    groups = re.findall(r'<contrib-group\b[^>]*content-type="author"[^>]*>(.*?)</contrib-group>', front, re.S)
    contribs = [c for g in groups for c in re.findall(r"<contrib\b[^>]*>(.*?)</contrib>", g, re.S)] or \
        re.findall(r'<contrib\b[^>]*contrib-type="author"[^>]*>(.*?)</contrib>', front, re.S)
    for c in contribs:
        sur, giv = re.search(r"<surname>(.*?)</surname>", c, re.S), re.search(r"<given-names[^>]*>(.*?)</given-names>", c, re.S)
        coll = re.search(r"<collab[^>]*>(.*?)</collab>", c, re.S)
        if sur:
            names.append(" ".join(x for x in [giv.group(1) if giv else "", sur.group(1)] if x).strip())
        elif coll:
            names.append(re.sub(r"<[^>]+>", "", coll.group(1)).strip())
    urls = sorted(set(m.rstrip("/") for m in re.findall(r"creativecommons\.org/licenses/by/[0-9.]+", s)))
    lic = f"CC BY {urls[-1].rsplit('/', 1)[1]}" if urls else "CC BY"
    return "; ".join(re.sub(r"\s+", " ", n) for n in names), lic


papers[["authors", "license"]] = papers.paper_id.map(authors_and_license).apply(pd.Series)
papers["title"] = papers.title.str.replace("\ufeff", "", regex=False).str.strip()
papers["source_url"] = "https://pmc.ncbi.nlm.nih.gov/articles/" + papers.paper_id + "/"
papers["year"] = papers.year.astype("Int64")


def anonymise(row):
    msgs = json.loads(row.prompt)
    tag = f"<writer>\n{row.writer}\n</writer>"
    assert msgs[-1]["content"].count(tag) == 1, row.paper_id
    msgs[-1]["content"] = msgs[-1]["content"].replace(tag, f"<writer>\n{LABEL[row.writer]}\n</writer>")
    return msgs + [{"role": "assistant", "content": row.answer}]


t["messages"] = t.apply(anonymise, axis=1)
t["writer"] = t.writer.map(LABEL)
t = t.rename(columns={"step": "call"}).merge(
    papers[["paper_id", "title", "doi", "license"]].rename(columns={"title": "paper_title", "doi": "paper_doi",
                                                                    "license": "paper_license"}), on="paper_id")
cols = ["paper_id", "paper_title", "paper_doi", "paper_license", "split", "call", "section", "writer", "messages"]
(OUT / "data").mkdir(parents=True, exist_ok=True)
for split in ["train", "validation"]:
    t[t.split == split][cols].drop(columns="split").to_parquet(OUT / "data" / f"{split}.parquet", index=False)
papers[["paper_id", "title", "authors", "journal", "year", "doi", "license", "source_url"]].to_parquet(
    OUT / "papers.parquet", index=False)

# checks: no trace of the original labels, nothing left out
blob = " ".join(json.dumps(m) for m in t.messages)
assert "<writer>\\nopus" not in blob and "<writer>\\nastra" not in blob
print(len(t), "examples:", t.split.value_counts().to_dict(), "| writers", t.writer.value_counts().to_dict(),
      "| papers", len(papers), "| licences", papers.license.value_counts().to_dict(),
      "| papers without authors", int((papers.authors == "").sum()))
