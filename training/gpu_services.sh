#!/usr/bin/env bash
# Pause or resume everything else that uses the GPUs on lambda, for training.
#   gpu_services.sh stop     # InkType (vLLM), Chattering semantic search, Kokoro speech, Parakeet dictation
#   gpu_services.sh start    # bring them all back
#   gpu_services.sh status
SERVICES="podman-inktype-vllm chattering-semantic kokoro-tts parakeet-server"
case "$1" in
  stop|start) sudo systemctl "$1" $SERVICES ;;
  status|"") for s in $SERVICES; do printf '%-24s %s\n' "$s" "$(systemctl is-active $s)"; done ;;
  *) echo "usage: $0 stop|start|status"; exit 1 ;;
esac
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
