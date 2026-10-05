"""Give a trained Qwen3.5 checkpoint back the base model's multi-token-prediction (MTP)
head, so vLLM can use it for speculative decoding. Training drops the head; the base's
head still predicts the next words well enough to be useful, and with greedy decoding
vLLM verifies every drafted word, so the text is the model's own either way.

    .venv/bin/python graft_mtp.py TRAINED_DIR Qwen/Qwen3.5-9B OUT_DIR

OUT_DIR holds links to the trained files, mtp.safetensors (the base's head) and an
index that lists both.
"""
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

trained, base, out = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
for f in trained.iterdir():
    if f.name not in ("model.safetensors.index.json",) and not (out / f.name).exists():
        (out / f.name).symlink_to(f)

base_dir = Path(snapshot_download(base, allow_patterns=["*.safetensors", "*.json"]))
mtp = {}
for f in base_dir.glob("*.safetensors"):
    with safe_open(f, "pt") as st:
        for k in st.keys():
            if k.startswith("mtp."):
                mtp[k] = st.get_tensor(k)
save_file(mtp, out / "mtp.safetensors", metadata={"format": "pt"})

weight_map = {}
for f in sorted(trained.glob("*.safetensors")):
    with safe_open(f, "pt") as st:
        weight_map.update({k: f.name for k in st.keys()})
weight_map.update({k: "mtp.safetensors" for k in mtp})
(out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
print(f"{out}: {len(mtp)} MTP tensors from {base} + {len(weight_map) - len(mtp)} trained tensors")
