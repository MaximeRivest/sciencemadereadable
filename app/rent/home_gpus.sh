#!/usr/bin/env bash
# Our 9B on lambda's GPU 0, in InkType's place (since 2026-10-05). Declared in ~/Projects/os/machines:
#   hosts/lambda/models.nix        model-vllm-smr-9b (system service; starts at boot)
#   hosts/lambda/ai-services.nix   inktypePaused = true (set false, and remove smr-9b, to give GPU 0 back)
#   hm/sciencemadereadable.nix     smr-server (the queue) and smr-worker-home (user services)
# This script only switches them for now (until the next reboot or rebuild):
#   app/rent/home_gpus.sh off      stop the 9B and the home worker, start InkType again
#   app/rent/home_gpus.sh on       stop InkType, start the 9B (about 6 minutes to load) and the home worker
#   app/rent/home_gpus.sh status
set -euo pipefail
case "${1:-status}" in
  off) systemctl --user stop smr-worker-home; sudo systemctl stop model-vllm-smr-9b; sudo systemctl start podman-inktype-vllm
       echo "GPU 0 is InkType's again (until the next reboot: see ~/Projects/os/machines/hosts/lambda/ai-services.nix)" ;;
  on)  sudo systemctl stop podman-inktype-vllm; sudo systemctl start model-vllm-smr-9b; systemctl --user start smr-worker-home
       echo "loading the 9B; the worker offers it as soon as it answers" ;;
  status) printf 'model-vllm-smr-9b   %s\n' "$(systemctl is-active model-vllm-smr-9b)"
          printf 'smr-worker-home     %s\n' "$(systemctl --user is-active smr-worker-home)"
          printf 'smr-server          %s\n' "$(systemctl --user is-active smr-server)"
          printf 'inktype             %s\n' "$(systemctl is-active podman-inktype-vllm)"
          curl -sf -m 3 localhost:8015/v1/models >/dev/null && echo "the 9B answers on :8015" || echo "the 9B does not answer" ;;
esac
