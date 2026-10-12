"""
A dashboard for the bundle-maker: one page to follow and understand its training run
=====================================================================================

    .venv/bin/python training/notes/bundle/dashboard_maker.py    # http://0.0.0.0:8794
    From the tailnet: https://lambda.tail69222b.ts.net:18794
Built from dashboard.py (the writer's); same layout, the bundle-maker's own tests.

Read-only. Shows, in plain words: where the run is and when it ends; how well it is learning
(loss on the training papers and on the 20 held-back papers, compared with the first run);
what each saved checkpoint writes and how faithful it is (once tested); a side-by-side view of a
held-back section (model vs teacher); and the data and the GPUs behind it. Standard library.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "training/runs"
DATA = HERE / "out" / "data300"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8794
RUN = "bundlemaker-0.8b"
BASELINE = None
EVAL = HERE / "out" / "maker-eval"


def read(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:                                       # noqa: BLE001
        return default


def log(run: str) -> list[dict]:
    out = []
    try:
        for line in (RUNS / run / "log.jsonl").read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except FileNotFoundError:
        pass
    return out


def gpus() -> list[dict]:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        rows = []
        for line in r.stdout.strip().splitlines():
            i, u, m, t, p, temp = [x.strip() for x in line.split(",")]
            rows.append({"gpu": int(i), "busy": float(u), "mem": float(m) / 1024, "total": float(t) / 1024,
                         "watts": float(p), "temp": float(temp)})
        return rows
    except Exception:                                       # noqa: BLE001
        return []


def data_progress() -> dict:
    papers = read(DATA / "papers.json", {}).get("papers", [])
    out = {}
    for t in ("opus", "astra"):
        ps = [p for p in papers if p["teacher"] == t]
        out[t] = {"done": sum((DATA / t / p["paper_id"] / "done.json").exists() for p in ps), "total": len(ps),
                  "status": ((DATA / f"STATUS-{t}.txt").read_text().strip() if (DATA / f"STATUS-{t}.txt").exists() else "")}
    return out


def evaluation() -> dict:
    progress = read(EVAL / "progress.json", {})
    cfg = read(RUNS / RUN / "config.json", {})
    every, total = cfg.get("eval_every") or 0, cfg.get("total_steps") or 0
    planned = [f"step-{s:05d}" for s in range(every, total, every)] + ["final"] if every else []
    saved = {p.name for p in (RUNS / RUN / "adapters").iterdir()} if (RUNS / RUN / "adapters").exists() else set()
    rows = []
    for ck in planned:
        events = []
        try:
            events = [json.loads(l) for l in (EVAL / ck / "_progress.jsonl").read_text().splitlines() if l.strip()]
        except FileNotFoundError:
            pass
        written = [e for e in events if e["phase"] == "written"]
        checked = {e["key"]: e for e in events if e["phase"] == "checked"}
        recent = [e for e in written if e["at"] > time.time() - 60]
        rows.append({"checkpoint": ck, "saved": ck in saved, "current": progress.get("current") == ck,
                     "written": len({e["key"] for e in written}), "checked": len(checked),
                     "checked_opus": sum(e.get("with_opus", False) for e in checked.values()),
                     "tokens_out": sum(e["tokens_out"] for e in written),
                     "tokens_per_second": round(sum(e["tokens_out"] for e in recent) / 60) if recent else 0})
    return {"progress": progress, "summary": read(EVAL / "summary.json", {}), "checkpoints": rows, "sections": 97}


def sample(checkpoint: str, key: str) -> dict:
    answers = read(EVAL / checkpoint / "answers.json", {})
    scores = read(EVAL / checkpoint / "scores.json", {})
    teacher = read(EVAL / "teacher-first" / "answers.json", {})
    pid, part = key.split("|") if "|" in key else ("", "")
    meta = next((p for p in read(DATA / "papers.json", {}).get("papers", []) if p["paper_id"] == pid), None)
    final, section = "", ""
    if meta:
        b = read(DATA / meta["teacher"] / pid / "bundle.json", {})
        paper = read(DATA / meta["teacher"] / pid / "paper.json", {})
        section = (paper.get("part_texts") or {}).get(part, "")
        import importlib.util
        spec = importlib.util.spec_from_file_location("bm", HERE / "bundlemaker_examples.py")
        bm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bm)
        final = bm.answer_text(b["notes"][part]["final"]) if b else ""
    return {"model": answers.get(key, ""), "teacher": final, "section": section,
            "teacher_name": meta["teacher"] if meta else "", "score": scores.get(key)}


def samples_index() -> dict:
    cks = sorted(p.name for p in EVAL.iterdir() if p.is_dir() and p.name != "teacher-first") if EVAL.exists() else []
    keys = sorted(read(EVAL / cks[0] / "answers.json", {})) if cks else []
    return {"checkpoints": cks, "keys": keys}


def api() -> dict:
    cfg = read(RUNS / RUN / "config.json", {})
    status = read(RUNS / RUN / "status.json", {})
    rows = log(RUN)
    train = [{"step": r["step"], "loss": r["loss"], "seconds": r.get("seconds"), "at": r["at"]} for r in rows if r["kind"] == "train"]
    val = [{k: r.get(k) for k in ("step", "all", "opus", "astra", "at")} for r in rows if r["kind"] == "validation"]
    base_val, base_cfg = [], {}
    recent = [t["seconds"] for t in train[-10:] if t.get("seconds")]
    sec = sum(recent) / len(recent) if recent else None
    total = cfg.get("total_steps") or 0
    step = train[-1]["step"] if train else 0
    left = (total - step) * sec * 1.06 if sec and total else None   # + validations
    return {"now": time.time(), "run": RUN, "config": cfg, "status": status, "train": train, "validation": val,
            "baseline": {"validation": base_val, "steps": base_cfg.get("total_steps"), "conversations": base_cfg.get("train_conversations")},
            "step": step, "total": total, "seconds_per_step": sec, "seconds_left": left,
            "gpus": gpus(), "data": data_progress(), "evaluation": evaluation(), "samples": samples_index(),
            "teacher_first": read(EVAL / "summary.json", {}).get("teacher-first")}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import parse_qs, urlparse
        url = urlparse(self.path)
        if url.path == "/api":
            body, kind = json.dumps(api()).encode(), "application/json"
        elif url.path == "/sample":
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            body, kind = json.dumps(sample(q.get("checkpoint", ""), q.get("key", ""))).encode(), "application/json"
        elif url.path in ("/", "/index.html"):
            body, kind = (HERE / "dashboard_maker.html").read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
