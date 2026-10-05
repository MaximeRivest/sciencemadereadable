"""Turn a training checkpoint into one GGUF file that llama.cpp can serve on the CPU.

    .venv/bin/python export_gguf.py --base Qwen/Qwen3.5-0.8B --weights runs/qwen35-0.8b/adapters/step-00152 \
        --out exports/qwen35-0.8b-step-00152.gguf

The LoRA adapter is merged into the base weights (or the whole saved model is used,
for full training), saved as a normal Hugging Face folder with the tokenizer and
chat template, then converted with llama.cpp's own converter (8-bit weights, q8_0,
near-lossless and twice as fast as 16-bit on this CPU).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
LLAMA_CPP = Path(os.environ.get("LLAMA_CPP_SRC", HERE / "tools" / "llama.cpp"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--weights", required=True, help='an adapter or saved model folder, or "untrained"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--outtype", default="q8_0")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"{out} exists")
        return
    weights = Path(args.weights)
    if args.weights == "untrained":
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    elif (weights / "adapter_config.json").exists():
        import peft
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
        model = peft.PeftModel.from_pretrained(model, str(weights)).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(str(weights), dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory(dir=HERE / "exports") as tmp:
        model.save_pretrained(tmp)
        AutoTokenizer.from_pretrained(args.base).save_pretrained(tmp)
        out.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(LLAMA_CPP / "gguf-py")}
        subprocess.run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), tmp, "--outfile",
                        str(out) + ".tmp", "--outtype", args.outtype,
                        # the base's extra next-token-prediction layer (MTP, for speculative decoding)
                        # is not loaded by transformers, so it is not in the trained model
                        "--no-mtp"], check=True, env=env)
    shutil.move(str(out) + ".tmp", out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
