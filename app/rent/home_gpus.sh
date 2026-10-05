#!/usr/bin/env bash
# Our models on lambda's own GPUs (2 x RTX 3090), as before launch day.
#   app/rent/home_gpus.sh off    stop the home worker and both home models (frees GPU 0 and GPU 1)
#   app/rent/home_gpus.sh on     start them again: 9B on GPU 0 (port 8015), 0.8B on GPU 1 (port 8014)
#   app/rent/home_gpus.sh status
# Same settings as the runs the site served until 2026-10-05. GPU 0 is the GPU InkType uses:
# pause it first (training/gpu_services.sh stop) before "on".
set -euo pipefail
APP=$(cd "$(dirname "$0")/.." && pwd)
T=$(cd "$APP/../training" && pwd)
IMAGE=fba1a5021627          # vllm/vllm-openai v0.29.0, the image the site's scored runs used
COMMON=(--dtype bfloat16 --max-model-len 65536 --enable-prefix-caching
        --override-generation-config '{"temperature": 0.0}' --default-chat-template-kwargs '{"enable_thinking":false}')
case "${1:-status}" in
  off)
    tmux kill-session -t smr-worker 2>/dev/null && echo "stopped the home worker" || true
    podman rm -f demo-9b demo-08b >/dev/null 2>&1 && echo "stopped the home 9B and 0.8B" || true
    ;;
  on)
    podman rm -f demo-9b demo-08b >/dev/null 2>&1 || true
    podman run -d --name demo-9b --device nvidia.com/gpu=0 --ipc=host --network=host -v "$T:$T:ro" $IMAGE \
      --model "$T/exports/qwen35-9b-glossary-mtp" --served-model-name our-9b --host 127.0.0.1 --port 8015 \
      --max-num-seqs 16 --gpu-memory-utilization 0.955 --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
      "${COMMON[@]}" >/dev/null
    podman run -d --name demo-08b --device nvidia.com/gpu=1 --ipc=host --network=host -v "$T:$T:ro" $IMAGE \
      --model "$T/exports/qwen35-0.8b-glossary-scratch-final-hf" --served-model-name our-0.8b --host 127.0.0.1 --port 8014 \
      --max-num-seqs 8 --gpu-memory-utilization 0.23 "${COMMON[@]}" >/dev/null
    echo "waiting for both models to load (a few minutes)"
    for i in $(seq 120); do curl -sf -m 3 localhost:8015/v1/models >/dev/null && curl -sf -m 3 localhost:8014/v1/models >/dev/null && break; sleep 5; done
    [ -f "$APP/worker/config.json" ] || cp "$APP/worker/config.home.example.json" "$APP/worker/config.json"
    tmux kill-session -t smr-worker 2>/dev/null || true
    tmux new -d -s smr-worker "cd $APP && node --conditions=functai-source worker/worker.ts 2>&1 | tee -a worker/worker.log"
    echo "the home GPUs are taking jobs"
    ;;
  status)
    podman ps --format "{{.Names}} {{.Status}}" | grep -E "demo-9b|demo-08b" || echo "home models: stopped"
    tmux has-session -t smr-worker 2>/dev/null && echo "home worker: running" || echo "home worker: stopped"
    ;;
esac
