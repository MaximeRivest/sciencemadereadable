#!/usr/bin/env bash
# run_one.sh GPU MODEL NAME [train.py options]: one training run, restarted from its last checkpoint
# up to 3 times if it crashes (for example a rare out-of-memory on a long paper).
set -o pipefail
cd "$(dirname "$0")"
# wait until nothing else holds this GPU (e.g. a benchmark server), so training cannot run out of memory
while [ "$(nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 1500 ]; do
  echo "[$(date -Is)] GPU $1 busy, waiting"; sleep 60
done
for attempt in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES="$1" .venv/bin/python train.py --model "$2" --name "$3" "${@:4}" 2>&1 | tee -a "logs/$3.log" && exit 0
  echo "[$(date -Is)] $3 crashed (attempt $attempt); resuming from the last checkpoint in 60 s" | tee -a "logs/$3.log"
  sleep 60
done
exit 1
