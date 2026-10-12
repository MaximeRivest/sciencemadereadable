"""
Test the trained writer on the 20 held-back papers
==================================================

1. generate  serve a checkpoint with vLLM (GPU 0), and write every part of the 20 validation
             papers from their bundle, exactly as in training: the opening first, then the four
             sections with the model's own opening as already read. Greedy decoding.
2. score     the same checks as for the teacher texts: code form checks, the trace (Opus: facts
             kept / changed / missing, additions), readability, and the fair blind judge (Opus,
             prose only, both orders) against the teacher's final text for that part.

    .venv/bin/python training/notes/bundle/evaluate.py generate step-00072
    .venv/bin/python training/notes/bundle/evaluate.py generate final
    .venv/bin/python training/notes/bundle/evaluate.py score step-00072 final
Results: training/notes/bundle/out/eval/<checkpoint>/<paper_id>.json and summary.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import importlib.util                               # noqa: E402


def _load(name: str):
    """This folder's module by path: other folders on sys.path have files with the same names
    (paper_corpus/pilot.py runs a rewrite job when imported)."""
    spec = importlib.util.spec_from_file_location(f"bundle_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = _load("pilot")                                  # form checks, judge, readability helpers
PR = _load("prompts")
render = _load("render")
EX = _load("examples")                              # the exact training prompt

RUN = ROOT / "training/runs" / os.environ.get("EVAL_RUN", "bundle-0.8b")
DATA = HERE / "out" / "data300"
OUT = HERE / "out" / ("eval" if RUN.name == "bundle-0.8b" else f"eval-{RUN.name}")
GPU = os.environ.get("EVAL_GPU", "0")
TRACE_PAPERS = int(os.environ.get("EVAL_TRACE_PAPERS", "999"))   # papers per checkpoint that get the Opus checks
JUDGE = os.environ.get("EVAL_JUDGE", "1") == "1"
VLLM = "/nix/store/skvyvs8jj0k7q4vw2137blrnfp3y5cb2-vllm-runtime-0.29.0/bin/vllm"
PORT = 8031


def validation_papers() -> list[dict]:
    frame = pd.read_parquet(ROOT / "training/data/examples-bundle.parquet")
    ids = sorted(set(frame[frame.split == "validation"].paper_id))
    papers = {p["paper_id"]: p for p in json.loads((DATA / "papers.json").read_text())["papers"]}
    by = {t: [papers[i] for i in ids if papers[i]["teacher"] == t] for t in ("opus", "astra")}
    return [p for pair in zip(by["opus"], by["astra"]) for p in pair]   # alternate, so any first N is balanced


def chat(messages: list[dict]) -> str:
    body = json.dumps({"model": "student", "messages": messages, "temperature": 0, "max_tokens": 6000,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1800) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def generate(checkpoint: str) -> None:
    folder = RUN / "adapters" / checkpoint
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": GPU}
    server = subprocess.Popen([VLLM, "serve", str(folder), "--tokenizer", "Qwen/Qwen3.5-0.8B", "--served-model-name", "student",
                               "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", "32768",
                               "--gpu-memory-utilization", "0.85", "--dtype", "bfloat16", "--enable-prefix-caching",
                               "--default-chat-template-kwargs", '{"enable_thinking": false}'],
                              env=env, stdout=open(OUT / f"vllm-{checkpoint}.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5)
                break
            except Exception:                                           # noqa: BLE001
                if server.poll() is not None:
                    raise RuntimeError(f"vLLM stopped; see {OUT}/vllm-{checkpoint}.log")
                time.sleep(10)
        out = OUT / checkpoint
        out.mkdir(parents=True, exist_ok=True)

        def one(p):
            target = out / f"{p['paper_id']}.json"
            if target.exists():
                return
            folder = DATA / p["teacher"] / p["paper_id"]
            bundle = (folder / "bundle.txt").read_text()
            names = list(json.loads((folder / "texts.json").read_text()))
            msg = lambda n, already: [{"role": "system", "content": EX.SYSTEM},
                                      {"role": "user", "content": EX.user(bundle, n, already)}]
            t = time.time()
            texts = {names[0]: chat(msg(names[0], ""))}
            with ThreadPoolExecutor(4) as pool:
                texts.update(dict(zip(names[1:], pool.map(lambda n: chat(msg(n, texts[names[0]])), names[1:]))))
            target.write_text(json.dumps({"paper_id": p["paper_id"], "teacher": p["teacher"], "texts": texts,
                                          "seconds": round(time.time() - t)}, indent=1, ensure_ascii=False))
            print(f"{checkpoint} {p['paper_id']} {round(time.time() - t)} s", flush=True)

        with ThreadPoolExecutor(6) as pool:
            list(pool.map(one, validation_papers()))
    finally:
        server.terminate()
        server.wait(60)


def score(checkpoints: list[str]) -> None:
    papers = validation_papers()

    def one(job):
        ck, p = job
        f = OUT / ck / f"{p['paper_id']}.json"
        d = json.loads(f.read_text())
        if "scores" in d:
            return d
        folder = DATA / p["teacher"] / p["paper_id"]
        bundle = (folder / "bundle.txt").read_text()
        teacher = json.loads((folder / "texts.json").read_text())
        b = json.loads((folder / "bundle.json").read_text())
        paper = json.loads((folder / "paper.json").read_text())
        v, parts_text = paper["verbatim"], paper["part_texts"]
        names = list(teacher)
        figures = "\n\n".join(render._figure(x) for x in v["figures"]) or "(none)"
        scores = {}

        with_model = papers.index(p) < TRACE_PAPERS

        def score_part(n):
            text = d["texts"].get(n, "")
            part = b["parts"][n]
            form = P.form_checks(text, part, [x["id"] for x in v["figures"] if x["part"] == n], len(v["references"]))
            free = {"numbers_of_facts_kept": P.numbers_kept(" ".join(f["note"] for f in part["facts"]), text),
                    "teacher_numbers_of_facts_kept": P.numbers_kept(" ".join(f["note"] for f in part["facts"]), teacher[n]["final"])}
            if not with_model:
                return {"form": form, "free": free, "readability": P.RD.measure(P.prose(text)),
                        "readability_teacher": P.RD.measure(P.prose(teacher[n]["final"]))}
            tr = P.ask(PR.TRACE, "trace", items=render.trace_view(v, n, part), text=text, allowed_sources=bundle,
                       reader=PR.READER).model_dump()
            if not JUDGE:
                return {"form": form, "free": free, "trace": tr, "readability": P.RD.measure(P.prose(text)),
                        "readability_teacher": P.RD.measure(P.prose(teacher[n]["final"]))}
            nothing = "(nothing: this is the first part)"
            before = lambda x: nothing if n == names[0] else x
            j = P.judge(parts_text[n], "\n\n".join(parts_text[m] for m in names if m != n), figures,
                        P.for_judge(before(d["texts"][names[0]])), P.for_judge(text),
                        P.for_judge(before(teacher[names[0]]["final"])), P.for_judge(teacher[n]["final"]))
            j["winner"] = {"first": "student", "second": "teacher"}.get(j["winner"], j["winner"])
            return {"form": form, "free": free, "trace": tr, "judge": j, "readability": P.RD.measure(P.prose(text)),
                    "readability_teacher": P.RD.measure(P.prose(teacher[n]["final"]))}

        for n in names:
            try:
                scores[n] = score_part(n)
            except Exception as error:                                   # noqa: BLE001  one part must not stop the rest
                scores[n] = {"error": f"{type(error).__name__}: {str(error)[:300]}"}
                print(f"{ck} {p['paper_id']} {n}: {scores[n]['error']}", flush=True)
        d["scores"] = scores
        f.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        print(f"scored {ck} {p['paper_id']}", flush=True)
        return d

    jobs = [(ck, p) for ck in checkpoints for p in papers]
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(one, jobs))
    summary = {}
    for ck in checkpoints:
        rs = [r for (c, _), r in zip(jobs, results) if c == ck]
        parts = [(r["teacher"], s) for r in rs for s in r["scores"].values() if "error" not in s]
        traced = [(t, s) for t, s in parts if "trace" in s]
        judged = [(t, s) for t, s in parts if "judge" in s]
        status = Counter(x["status"] for _, s in traced for x in s["trace"]["facts"])
        adds = Counter(x["kind"] for _, s in traced for x in s["trace"]["additions"])
        form = Counter()
        for _, s in parts:
            form.update({k: (len(v) if isinstance(v, list) else v) for k, v in s["form"].items()})
        mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
        rd = lambda key: {k: round(sum(s[key][k] for _, s in parts) / len(parts), 1) for k in ("sent_mean", "para_mean", "fk_grade", "words")}
        summary[ck] = {"parts": len(parts), "parts traced": len(traced), "parts judged": len(judged),
                       "numbers of the facts kept": mean([s["free"]["numbers_of_facts_kept"] for _, s in parts if "free" in s]),
                       "teacher: numbers of the facts kept": mean([s["free"]["teacher_numbers_of_facts_kept"] for _, s in parts if "free" in s]),
                       "form errors per part": mean([sum((len(v) if isinstance(v, list) else v) for k, v in s["form"].items()
                                                         if k != "citations_not_in_facts") for _, s in parts]),
                       "facts": dict(status),
                       "facts kept share": round(status["kept"] / sum(status.values()), 3) if status else None,
                       "additions": dict(adds),
                       "judge vs teacher": dict(Counter(s["judge"]["winner"] for _, s in judged)),
                       "form": dict(form),
                       "readability (mean per part)": rd("readability"), "teacher readability": rd("readability_teacher")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"generate": lambda: generate(sys.argv[2]), "score": lambda: score(sys.argv[2:])}[sys.argv[1]]()
