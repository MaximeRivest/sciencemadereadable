#!/usr/bin/env bash
# Stop the training runs (they resume from their last checkpoint with start.sh;
# up to 30 minutes of training since that checkpoint is lost). The dashboard keeps running.
for s in train-gpu0 train-gpu1; do tmux kill-session -t "$s" 2>/dev/null && echo "stopped $s"; done
echo "GPU services are still paused; bring them back with: $(dirname "$0")/gpu_services.sh start"
