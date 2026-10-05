"""Term check (rewrite_benchmark/term_check.py) on the v3 pilot: the old Opus
answers vs the v3 answers, same 15 conversations. A ⟦marker⟧ counts as "flagged for later
explanation", not as unexplained; explanations are checked against the glossary that
conversation actually received.

    .venv/bin/python training/v3/check_pilot.py pilot2
"""
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import term_check as T                                            # noqa: E402
from build import glossary_lines                                  # noqa: E402

DIR = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "pilot2")


def judge(text, context, reference):
    marked = sorted({m.strip() for m in re.findall(r"⟦([^⟧]+)⟧", text)})
    plain = re.sub(r"⟦([^⟧]+)⟧", r"\1", text)
    terms = T.candidates(plain)
    key = hashlib.sha256(json.dumps(["v3", T.JUDGE.version, plain, context, terms, reference]).encode()).hexdigest()[:24]
    path = T.CACHE / f"v3-{key}.json"
    if path.exists():
        rec = json.loads(path.read_text())
    else:
        r = T.JUDGE.predict(rewrite=plain, reading_context=context, candidate_terms=terms,
                            reference_glossary=reference, reader=T.READER).result
        rec = {"terms": [t.model_dump() for t in r.terms], "missed_terms": [t.model_dump() for t in r.missed_terms]}
        T.CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec))
    rec["marked"] = marked
    rec["words"] = len(plain.split())
    return rec


def main():
    tasks = []
    for f in sorted(glob.glob(str(DIR / "PMC*.json"))):
        r = json.load(open(f))
        new_opening = next(c["new_answer"] for c in r["conversations"] if c["section"] == "opening")
        old_opening = next(c["old_answer"] for c in r["conversations"] if c["section"] == "opening")
        for c in r["conversations"]:
            ref = glossary_lines(c["glossary_given"])
            for version, text, ctx in (("old Opus answer", c["old_answer"], "" if c["section"] == "opening" else old_opening),
                                       ("v3 answer", c["new_answer"], "" if c["section"] == "opening" else new_opening)):
                tasks.append((r["paper_id"], c["section"], c["dropout"], version, text, ctx, ref))
    with ThreadPoolExecutor(12) as pool:
        results = list(pool.map(lambda t: (t, judge(t[4], t[5], t[6])), tasks))
    rows = []
    for (pid, sec, drop, version, *_), rec in results:
        marked = {m.lower() for m in rec["marked"]}
        for t in rec["terms"] + rec["missed_terms"]:
            status = "marked" if t["term"].lower() in marked or any(m in t["term"].lower() for m in marked) else t["status"]
            rows.append({"paper": pid, "section": sec, "dropout": drop, "version": version, "term": t["term"],
                         "status": status, "correct": t["explanation_correct"], "note": t["note"]})
    (DIR / "term_check.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    words = Counter()
    for (pid, sec, drop, version, *_), rec in results:
        words[version] += rec["words"]
    print(f"{'':<18}{'needed':>7}{'by 1st use':>11}{'later':>7}{'marked':>8}{'bare':>7}{'bare/1000w':>11}{'wrong':>7}")
    for v in ("old Opus answer", "v3 answer"):
        rs = [x for x in rows if x["version"] == v and x["status"] != "no_need"]
        c = Counter(x["status"] for x in rs)
        expl = [x for x in rs if x["status"].startswith("explained")]
        wrong = sum(x["correct"] == "no" for x in expl)
        n = len(rs)
        print(f"{v:<18}{n:>7}{100*(c['explained_at_first_use']+c['explained_earlier'])/n:>10.0f}%{100*c['explained_later']/n:>6.0f}%"
              f"{100*c['marked']/n:>7.0f}%{100*c['not_explained']/n:>6.0f}%{1000*c['not_explained']/words[v]:>11.1f}"
              f"{100*wrong/max(1,len(expl)):>6.0f}%")
    print("\nby input (v3 answers):")
    for mode in ("no glossary", "dropout"):
        rs = [x for x in rows if x["version"] == "v3 answer" and x["status"] != "no_need" and x["dropout"].startswith(mode)]
        c = Counter(x["status"] for x in rs)
        print(f"  {mode:<12} {len(rs):>4} needed: explained {100*sum(v for k, v in c.items() if k.startswith('explained'))/max(1,len(rs)):.0f}%"
              f", marked {100*c['marked']/max(1,len(rs)):.0f}%, bare {100*c['not_explained']/max(1,len(rs)):.0f}%")
    bare = [x for x in rows if x["version"] == "v3 answer" and x["status"] == "not_explained"]
    print("\nbare in v3 (sample):", ", ".join(sorted({x["term"] for x in bare}))[:900])
    wrong = [x for x in rows if x["version"] == "v3 answer" and x["correct"] == "no"]
    print("\nwrong explanations in v3:")
    for x in wrong[:10]:
        print(f"  {x['paper']} {x['section']}: {x['term']}: {x['note'][:170]}")


if __name__ == "__main__":
    main()
