"""
Train, and test each checkpoint on the same GPU as it is saved
==============================================================

Runs training/train.py with --pause-after-checkpoints: at every checkpoint training saves
everything and stops; this writes the held-back sections with that checkpoint (vLLM, a few
minutes), then starts training again, which continues at the next step. Scoring (code checks
and the Opus meaning check) runs separately (maker_evaluate.py watch) and needs no GPU.

    .venv/bin/python training/notes/bundle/train_with_tests.py -- <train.py arguments>
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
args = sys.argv[sys.argv.index("--") + 1:]
name = args[args.index("--name") + 1]
run = ROOT / "training/runs" / name
env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": os.environ.get("MAKER_GPU", "0")}
log = open(ROOT / "training/logs" / f"{name}.log", "a")


def training_running() -> bool:
    out = subprocess.run(["pgrep", "-f", f"train.py .*--name {name} "], capture_output=True, text=True).stdout
    return bool(out.strip())


def write_untested():
    """Write the held-back notes of every saved checkpoint that does not have them yet."""
    import json
    for ck in sorted(p.name for p in (run / "adapters").iterdir() if p.name.startswith("step-")):
        answers = HERE / "out" / "maker-eval" / ck / "answers.json"
        if answers.exists() and len(json.loads(answers.read_text())) >= 97:
            continue
        print(f"testing {ck}", flush=True)
        subprocess.run([str(ROOT / ".venv/bin/python"), str(HERE / "maker_evaluate.py"), "generate", ck],
                       env={**os.environ, "MAKER_GPU": env["CUDA_VISIBLE_DEVICES"]})


import time
while training_running():                            # a run started earlier: let it reach its next pause
    time.sleep(20)
if (run / "status.json").exists() and "paused" in (run / "status.json").read_text():
    write_untested()
while True:
    code = subprocess.run([str(ROOT / "training/.venv/bin/python"), str(ROOT / "training/train.py"), *args,
                           "--pause-after-checkpoints"], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 3:
        print(f"training ended (exit {code})", flush=True)
        break
    write_untested()
