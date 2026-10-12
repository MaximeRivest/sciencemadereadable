"""
After a training run: test every checkpoint, then give GPU 1 back to the services
==================================================================================

Waits until training/runs/<run>/status.json says finished, then for each saved checkpoint
(oldest first): writes the 20 held-back papers with vLLM on GPU 1, and scores them
(code checks on all 20 papers; the Opus fact check on the first 4 papers, 2 per teacher).
The best checkpoint (most facts kept) and the last one also get the Opus fact check and the
blind judge on all 20 papers. Finally the services stopped for training are started again.

    .venv/bin/python training/notes/bundle/after_training.py bundle-0.8b-1500
Progress: training/notes/bundle/out/eval-<run>/progress.json (read by the dashboard)
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN_NAME = sys.argv[1]
RUN = ROOT / "training/runs" / RUN_NAME
OUT = HERE / "out" / f"eval-{RUN_NAME}"
PY = str(ROOT / ".venv/bin/python")
SYSTEM_SERVICES = ["parakeet-server", "chattering-semantic", "kokoro-tts"]
USER_SERVICES = ["inktype-speech", "qwen3-embed-4b"]


def progress(**fields):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "progress.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **fields, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))


def evaluate(*args, **env):
    e = {**os.environ, "EVAL_RUN": RUN_NAME, "EVAL_GPU": "1", **{k: str(v) for k, v in env.items()}}
    subprocess.run([PY, str(HERE / "evaluate.py"), *args], env=e, check=True,
                   stdout=open(OUT / "evaluate.log", "a"), stderr=subprocess.STDOUT)


def main():
    progress(state="waiting for training to finish")
    while True:
        status = json.loads((RUN / "status.json").read_text())
        if str(status.get("state", "")).startswith(("finished", "crashed", "stopped")):
            break
        time.sleep(60)
    try:
        checkpoints = sorted(p.name for p in (RUN / "adapters").iterdir() if p.name.startswith("step-"))
        if (RUN / "adapters/final").exists():
            checkpoints.append("final")
        progress(state="testing checkpoints", checkpoints=checkpoints, done=[])
        done = []
        for ck in checkpoints:
            progress(current=ck)
            evaluate("generate", ck)
            evaluate("score", ck, EVAL_TRACE_PAPERS=4, EVAL_JUDGE=0)
            done.append(ck)
            progress(done=done)
        summary = json.loads((OUT / "summary.json").read_text())
        best = max((c for c in done if summary.get(c, {}).get("facts kept share") is not None),
                   key=lambda c: summary[c]["facts kept share"])
        progress(state=f"full test of the best ({best}) and last checkpoints", best=best, current=None)
        for ck in dict.fromkeys([best, done[-1]]):
            for f in (OUT / ck).glob("*.json"):                      # rescore with all checks
                d = json.loads(f.read_text())
                d.pop("scores", None)
                f.write_text(json.dumps(d, indent=1, ensure_ascii=False))
            evaluate("score", ck, EVAL_TRACE_PAPERS=999, EVAL_JUDGE=1)
        # one summary over every checkpoint (score recomputes it from the saved scores)
        evaluate("score", *done, EVAL_TRACE_PAPERS=0, EVAL_JUDGE=0)
        progress(state="finished")
    except Exception as error:                                        # noqa: BLE001
        progress(state=f"failed: {type(error).__name__}: {error}")
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", *SYSTEM_SERVICES])
        subprocess.run(["systemctl", "--user", "start", *USER_SERVICES])
        progress(services="restarted")


if __name__ == "__main__":
    main()
