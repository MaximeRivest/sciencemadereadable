#!/usr/bin/env bash
# Run at every boot of lambda by the user unit training-after-boot.service, while
# training is not finished: pause the GPU services, cap both GPUs at 280 W (GPU 0
# fell off the bus at full power on 2026-10-01, likely power spikes), then resume
# the training runs and the dashboard. When every planned run has finished, it does
# nothing and disables itself.
cd "$(dirname "$0")"
log() { echo "[$(date -Is)] $*" | tee -a logs/after_boot.log; }
mkdir -p logs
all_done=1
for n in qwen35-4b-glossary-scratch qwen35-0.8b-glossary-scratch; do [ -d "runs/$n/adapters/final" ] || all_done=0; done
if [ "$all_done" = 1 ]; then
  log "all runs finished: nothing to do; disabling myself"
  systemctl --user disable training-after-boot.service
  exit 0
fi
for i in $(seq 60); do [ "$(nvidia-smi -L 2>/dev/null | wc -l)" = 2 ] && break; sleep 5; done
log "GPUs: $(nvidia-smi -L | tr '\n' ' ')"
./gpu_services.sh stop >> logs/after_boot.log 2>&1
sudo nvidia-smi -pl 280 >> logs/after_boot.log 2>&1
log "power limits: $(nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tr '\n' ' ')"
./start.sh >> logs/after_boot.log 2>&1
# Once only: the Opus paper-rewrite run was running before the 2026-10-01 reboot.
if [ ! -e logs/.paper_rewrite_restarted ]; then
  touch logs/.paper_rewrite_restarted
  ../paper_corpus/start.sh >> logs/after_boot.log 2>&1
  log "restarted the Opus paper-rewrite run"
fi
log "done"
