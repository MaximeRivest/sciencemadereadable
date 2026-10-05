#!/usr/bin/env bash
# Serve a local model on GPU 1 with vLLM, OpenAI-compatible API on port 8001.
#   serve_local.sh Qwen/Qwen3.5-4B
#   serve_local.sh Qwen/Qwen3.5-9B fp8      (weights squeezed to 8 bits to fit)
# GPU 0 hosts the InkType Qwen 27B; GPU 1 also hosts speech/voice services (~6.5 GB),
# so this uses at most 16 GB.  Stop with: podman stop local-llm
MODEL="$1"; QUANT="${2:-}"
podman rm -f local-llm >/dev/null 2>&1
EXTRA=(); [ -n "$QUANT" ] && EXTRA=(--quantization "$QUANT")
podman run -d --name local-llm --device nvidia.com/gpu=1 --ipc=host -p 127.0.0.1:8001:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  docker.io/vllm/vllm-openai:v0.25.1 \
  --model "$MODEL" --served-model-name local --max-model-len 32768 --max-num-seqs 6 \
  --gpu-memory-utilization 0.64 --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking":false}' "${EXTRA[@]}"
echo "Starting $MODEL on GPU 1 (port 8001). Logs: podman logs -f local-llm"
