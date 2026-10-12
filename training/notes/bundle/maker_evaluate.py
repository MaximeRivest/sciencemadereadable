"""
Test the bundle-maker on the held-back sections
===============================================

For each checkpoint of training/runs/bundlemaker-0.8b: serve it with vLLM, make the facts of
the 97 held-back sections (the writer's 20 held-back papers), and score them against the
teacher's final facts for the same section:

  free, every section   the answer reads back (parse); every paragraph has notes; the code
                        checks (numbers, citations, hedge words of each paragraph kept);
                        number of facts and words, vs the teacher
  Opus, some papers     the meaning check (check_part) the teachers' facts passed: problems
                        missing / changed / not in the paper, major or minor

    .venv/bin/python training/notes/bundle/maker_evaluate.py watch     # test checkpoints as they come,
                                                                       # then fully test the best and last
Results in training/notes/bundle/out/maker-eval/<checkpoint>/ and summary.json; progress.json
and _progress.jsonl for the dashboard. GPU from MAKER_GPU (default 0).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import importlib.util                               # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"bundle_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = _load("pilot")
PR = _load("prompts")
render = _load("render")
checks = _load("checks")
BM = _load("bundlemaker_examples")

RUN = ROOT / "training/runs/bundlemaker-0.8b"
OUT = HERE / "out" / "maker-eval"
DATA = HERE / "out" / "data300"
VLLM = "/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/vllm"
PORT = 8032
GPU = os.environ.get("MAKER_GPU", "0")
OPUS_PAPERS = 4                                     # papers per checkpoint with the Opus meaning check


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def progress(**fields):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "progress.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **fields, "updated": now()}, indent=1))


_lock = threading.Lock()


def note(ck: str, row: dict):
    (OUT / ck).mkdir(parents=True, exist_ok=True)
    with _lock, (OUT / ck / "_progress.jsonl").open("a") as f:
        f.write(json.dumps({**row, "at": time.time()}) + "\n")


def held_back() -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "training/data/examples-bundlemaker.parquet")
    v = frame[frame.split == "validation"].copy()
    order = {pid: i for i, pid in enumerate(sorted(set(v.paper_id)))}
    v["order"] = v.paper_id.map(order)
    by_teacher = {t: sorted(set(v[v.writer == t].paper_id)) for t in ("opus", "astra")}
    opus_set = set(by_teacher["opus"][:OPUS_PAPERS // 2] + by_teacher["astra"][:OPUS_PAPERS // 2])
    v["opus_check"] = v.paper_id.isin(opus_set)
    return v.sort_values(["order", "section"])


def generate(ck: str) -> None:
    folder = RUN / "adapters" / ck
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPU}
    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([VLLM, "serve", str(folder), "--tokenizer", "Qwen/Qwen3.5-0.8B", "--served-model-name", "maker",
                               "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", "24576",
                               "--gpu-memory-utilization", "0.85", "--dtype", "bfloat16",
                               "--default-chat-template-kwargs", '{"enable_thinking": false}'],
                              env=env, stdout=open(OUT / f"vllm-{ck}.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
                break
            except Exception:                                            # noqa: BLE001
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM stopped; see {OUT}/vllm-{ck}.log")
                time.sleep(10)
        target = OUT / ck / "answers.json"
        answers = json.loads(target.read_text()) if target.exists() else {}

        def one(row):
            key = f"{row.paper_id}|{row.section}"
            if key in answers:
                return
            t = time.time()
            body = json.dumps({"model": "maker", "messages": json.loads(row.prompt), "temperature": 0, "max_tokens": 8000,
                               "chat_template_kwargs": {"enable_thinking": False}}).encode()
            r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=1800) as resp:
                data = json.load(resp)
            with _lock:
                answers[key] = data["choices"][0]["message"]["content"]
                target.write_text(json.dumps(answers, ensure_ascii=False, indent=1))
            note(ck, {"phase": "written", "key": key, "tokens_in": data["usage"]["prompt_tokens"],
                      "tokens_out": data["usage"]["completion_tokens"], "seconds": round(time.time() - t, 1)})

        with ThreadPoolExecutor(12) as pool:
            list(pool.map(one, held_back().itertuples()))
    finally:
        server.terminate()
        server.wait(60)


def score(ck: str, opus_everywhere: bool = False) -> dict:
    answers = json.loads((OUT / ck / "answers.json").read_text())
    path = OUT / ck / "scores.json"
    scores = json.loads(path.read_text()) if path.exists() else {}

    def one(row):
        key = f"{row.paper_id}|{row.section}"
        if key in scores and (not opus_everywhere or "opus" in scores[key]):
            return
        folder = DATA / row.writer / row.paper_id
        paper = json.loads((folder / "paper.json").read_text())
        bundle = json.loads((folder / "bundle.json").read_text())
        structure = paper["structure"]["parts"][row.section]
        notes = BM.parse(answers[key])
        teacher = bundle["notes"][row.section]["final"]
        words = lambda n: sum(len(f["note"].split()) for p in n["paragraphs"] for f in p["facts"])
        facts = lambda n: sum(len(p["facts"]) for p in n["paragraphs"])
        problems = checks.part_problems(structure, notes)
        s = {"parsed": bool(notes["paragraphs"]), "paragraphs": len(structure["paragraphs"]),
             "paragraphs_with_notes": len({p["paragraph"] for p in notes["paragraphs"]} & {p["n"] for p in structure["paragraphs"]}),
             "code_problems": problems, "facts": facts(notes), "teacher_facts": facts(teacher),
             "words": words(notes), "teacher_words": words(teacher),
             "section_words": sum(len(p["text"].split()) for p in structure["paragraphs"])}
        if row.opus_check or opus_everywhere:
            v = paper["verbatim"]
            from_parts = {"headings": structure["headings"], "facts": [], "plan": []}
            heading = {p["n"]: p["heading"] for p in structure["paragraphs"]}
            for p in notes["paragraphs"]:
                ids = [f["id"] for f in p["facts"]]
                from_parts["facts"] += [{**f, "heading": heading.get(p["paragraph"], structure["headings"][-1])} for f in p["facts"]]
                from_parts["plan"].append({"heading": heading.get(p["paragraph"], structure["headings"][-1]),
                                           "purpose": p["purpose"], "items": ids})
            rest = "\n\n".join(t for n, t in paper["part_texts"].items() if n != row.section)
            try:
                found = P.ask(PR.CHECK_PART, "maker check", original=paper["part_texts"][row.section],
                              bundle_part=render.part_view(v, bundle["front"], row.section, from_parts),
                              rest_of_paper=rest).problems
                s["opus"] = [x.model_dump() for x in found]
            except Exception as error:                                   # noqa: BLE001
                s["opus_error"] = f"{type(error).__name__}: {str(error)[:200]}"
        with _lock:
            scores[key] = s
            path.write_text(json.dumps(scores, ensure_ascii=False, indent=1))
        note(ck, {"phase": "checked", "key": key, "with_opus": "opus" in s})

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(one, held_back().itertuples()))
    return summarize(ck, scores)


def summarize(ck: str, scores: dict) -> dict:
    rows = list(scores.values())
    opus = [r for r in rows if "opus" in r]
    kinds = Counter(f"{x['kind']} ({x['severity']})" for r in opus for x in r["opus"])
    teacher_opus = Counter()                                           # the teachers' first facts, same sections, same check
    s = {"sections": len(rows),
         "answer reads back": round(sum(r["parsed"] for r in rows) / len(rows), 3),
         "paragraphs covered": round(sum(r["paragraphs_with_notes"] for r in rows) / max(1, sum(r["paragraphs"] for r in rows)), 3),
         "sections clean on code checks": round(sum(not r["code_problems"] for r in rows) / len(rows), 3),
         "code problems": dict(Counter(x["kind"] for r in rows for x in r["code_problems"])),
         "facts vs teacher": round(sum(r["facts"] for r in rows) / max(1, sum(r["teacher_facts"] for r in rows)), 3),
         "words vs section": round(sum(r["words"] for r in rows) / max(1, sum(r["section_words"] for r in rows)), 3),
         "teacher words vs section": round(sum(r["teacher_words"] for r in rows) / max(1, sum(r["section_words"] for r in rows)), 3),
         "sections checked by Opus": len(opus),
         "Opus: sections with a major problem": sum(any(x["severity"] == "major" for x in r["opus"]) for r in opus),
         "Opus: problems per section": round(sum(len(r["opus"]) for r in opus) / len(opus), 2) if opus else None,
         "Opus: problems by kind": dict(kinds)}
    path = OUT / "summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old[ck] = s
    path.write_text(json.dumps(dict(sorted(old.items(), key=lambda kv: (kv[0] == "final", kv[0]))), indent=1))
    return s


def teacher_reference() -> None:
    """The teachers' FIRST facts (before checks and fixes) on the same held-back sections, scored
    the same way: what the bundle-maker is compared with (its target is the fixed facts)."""
    if (OUT / "teacher-first" / "scores.json").exists():
        return
    answers = {}
    for row in held_back().itertuples():
        b = json.loads((DATA / row.writer / row.paper_id / "bundle.json").read_text())
        answers[f"{row.paper_id}|{row.section}"] = BM.answer_text(b["notes"][row.section]["first"])
    (OUT / "teacher-first").mkdir(parents=True, exist_ok=True)
    (OUT / "teacher-first" / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=1))
    score("teacher-first")


def ready(folder: Path) -> bool:
    w = folder / "model.safetensors"
    return w.exists() and (folder / "config.json").exists() and time.time() - w.stat().st_mtime > 120


def finished() -> bool:
    try:
        return str(json.loads((RUN / "status.json").read_text()).get("state", "")).startswith(("finished", "crashed", "stopped"))
    except FileNotFoundError:
        return False


def watch():
    progress(state="waiting for checkpoints")
    teacher_reference()
    done = list(json.loads((OUT / "progress.json").read_text()).get("done", []))
    while True:
        end = finished()
        names = sorted(p.name for p in (RUN / "adapters").iterdir() if p.name.startswith("step-") and ready(p)) \
            if (RUN / "adapters").exists() else []
        if end and (RUN / "adapters/final").exists() and ready(RUN / "adapters/final"):
            names.append("final")
        todo = [n for n in names if n not in done]
        if not todo:
            if end:
                break
            progress(state="waiting for the GPU (training uses it); checkpoints are tested when it ends", current=None, done=done)
            time.sleep(60)
            continue
        if not end:                                                  # GPU 0 is busy with training
            time.sleep(60)
            continue
        ck = todo[0]
        progress(state="testing checkpoints", current=ck, done=done)
        generate(ck)
        score(ck)
        done.append(ck)
        progress(done=done)
    summary = json.loads((OUT / "summary.json").read_text())
    tested = [c for c in done if c in summary and summary[c]["sections checked by Opus"]]
    best = min(tested, key=lambda c: (summary[c]["Opus: problems per section"], -summary[c]["sections clean on code checks"]))
    progress(state=f"full test of the best ({best}) and the last checkpoints", best=best, current=None)
    for ck in dict.fromkeys([best, done[-1]]):
        progress(current=ck)
        score(ck, opus_everywhere=True)
    progress(state="finished", current=None)


if __name__ == "__main__":
    {"watch": watch, "generate": lambda: generate(sys.argv[2]), "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
