"""
Test every checkpoint as training goes, then give the GPUs back
===============================================================

Runs beside training. Each checkpoint, as soon as it is saved: writes the 20 held-back
papers with vLLM on GPU 0, and is scored (code checks on all 20 papers; the Opus fact check
on the first 4 papers, 2 per teacher). When training has finished, the services paused on
GPU 1 start again at once; then the best checkpoint (most facts kept) and the last one get the
Opus fact check and the blind judge on all 20 papers; finally the website's 9B model, stopped
to free GPU 0, is started again.

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
GPU_SERVICE = "model-vllm-smr-9b"                   # the website's 9B, stopped to free GPU 0


def progress(**fields):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "progress.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **fields, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))


def evaluate(*args, **env):
    e = {**os.environ, "EVAL_RUN": RUN_NAME, "EVAL_GPU": "0", **{k: str(v) for k, v in env.items()}}
    subprocess.run([PY, str(HERE / "evaluate.py"), *args], env=e, check=True,
                   stdout=open(OUT / "evaluate.log", "a"), stderr=subprocess.STDOUT)


def ready(folder: Path) -> bool:
    """A checkpoint folder that training has finished writing."""
    w = folder / "model.safetensors"
    return w.exists() and (folder / "config.json").exists() and time.time() - w.stat().st_mtime > 120


def finished() -> bool:
    state = str(json.loads((RUN / "status.json").read_text()).get("state", ""))
    return state.startswith(("finished", "crashed", "stopped"))


def main():
    done = list(json.loads((OUT / "progress.json").read_text()).get("done", [])) if (OUT / "progress.json").exists() else []
    services_back = False
    try:
        while True:
            end = finished()
            if end and not services_back:
                subprocess.run(["sudo", "-n", "systemctl", "start", *SYSTEM_SERVICES])
                subprocess.run(["systemctl", "--user", "start", *USER_SERVICES])
                services_back = True
                progress(services="GPU 1 services restarted")
            names = sorted(p.name for p in (RUN / "adapters").iterdir() if p.name.startswith("step-") and ready(p)) \
                if (RUN / "adapters").exists() else []
            if end and (RUN / "adapters/final").exists() and ready(RUN / "adapters/final"):
                names.append("final")
            todo = [n for n in names if n not in done]
            if not todo:
                if end:
                    break
                progress(state="testing as training goes: waiting for the next checkpoint", current=None, done=done)
                time.sleep(60)
                continue
            ck = todo[0]
            progress(state="testing as training goes" if not end else "testing the last checkpoints", current=ck, done=done)
            evaluate("generate", ck)
            evaluate("score", ck, EVAL_TRACE_PAPERS=4, EVAL_JUDGE=0)
            done.append(ck)
            progress(done=done)
        summary = json.loads((OUT / "summary.json").read_text())
        best = max((c for c in done if summary.get(c, {}).get("facts kept share") is not None),
                   key=lambda c: summary[c]["facts kept share"])
        progress(state=f"full test of the best ({best}) and the last checkpoints", best=best, current=None)
        for ck in dict.fromkeys([best, done[-1]]):
            for f in (OUT / ck).glob("*.json"):                      # rescore with every check
                d = json.loads(f.read_text())
                d.pop("scores", None)
                f.write_text(json.dumps(d, indent=1, ensure_ascii=False))
            evaluate("score", ck, EVAL_TRACE_PAPERS=999, EVAL_JUDGE=1)
        evaluate("score", *done, EVAL_TRACE_PAPERS=0, EVAL_JUDGE=0)      # one summary over every checkpoint
        progress(state="finished")
    except Exception as error:                                        # noqa: BLE001
        progress(state=f"failed: {type(error).__name__}: {error}")
    finally:
        if not services_back:
            subprocess.run(["sudo", "-n", "systemctl", "start", *SYSTEM_SERVICES])
            subprocess.run(["systemctl", "--user", "start", *USER_SERVICES])
        subprocess.run(["sudo", "-n", "systemctl", "start", GPU_SERVICE])
        progress(services="all restarted (GPU 1 services and the website's 9B)")


if __name__ == "__main__":
    main()
