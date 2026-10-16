"""Build a self-contained, offline, inspectable report from a completed run."""
import argparse
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent


def build(run, output):
    def load(name):
        return json.loads((run / name).read_text())
    labels, rewrite, extraction = load("labelled.json"), load("rewrite.json"), load("extraction.json")
    comparison, paper = load("comparison.json"), load("paper.json")
    if not labels["ready"] or rewrite["sectionsSha256"] != paper["sections_sha256"] or labels["sections"] != paper["sections"]:
        raise ValueError("Input/rewrite mismatch or labels not ready; regenerate before reporting.")
    output.mkdir(parents=True, exist_ok=True)
    data = {"title": re.sub(r"^#+\s*", "", labels["sections"]["title"]),
            "doi": comparison["doi"], "pages": extraction["pages"],
            "labels": labels, "rewrite": rewrite, "comparison": comparison,
            "reviewCount": sum(bool(b["manual_review"]) for b in labels["ledger"]),
            "extractionSeconds": extraction["seconds"], "glossarySeconds": paper["seconds"]}
    (output / "data.js").write_text("window.REPORT = " + json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + ";\n")
    (output / "summary.json").write_text(json.dumps({k: v for k, v in data.items() if k not in {"labels", "rewrite", "comparison"}} | {
        "accounting": labels["accounting"], "jev_usage": labels["usage"], "jev_estimated_usd": labels["estimated_usd"],
        "reference_paragraphs_located": comparison["located_in_expected"], "reference_paragraphs_checked": comparison["paragraph_count"],
        "rewrite_calls": rewrite["calls"], "limitations": "One existing corpus paper, four inspected labels, no full factual/table audit; not a held-out benchmark."}, indent=2))
    shutil.copyfile(HERE / "report.html", output / "index.html")
    for name in ("original.pdf", "rewrite.md"):
        shutil.copyfile(run / name, output / name)
    shutil.copytree(run / "pages", output / "pages", dirs_exist_ok=True)
    vendor = output / "vendor"
    vendor.mkdir(exist_ok=True)
    packages = HERE.parent / "app" / "node_modules"
    shutil.copyfile(packages / "marked/lib/marked.umd.js", vendor / "marked.umd.js")
    shutil.copyfile(packages / "dompurify/dist/purify.min.js", vendor / "purify.min.js")
    for name in ("marked", "dompurify"):
        for source in (packages / name).glob("LICENSE*"):
            shutil.copyfile(source, vendor / f"{name}-{source.name}")
    print(f"Report: {output / 'index.html'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=HERE / "demo")
    a = p.parse_args()
    build(a.run, a.out)
