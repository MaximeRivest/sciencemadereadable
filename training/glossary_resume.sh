#!/usr/bin/env bash
# Resume glossary_run.sh after the glossary step crashed (2026-10-01 21:30): glossaries,
# training conversations, then the 0.8B with the glossary on GPU 1 (the 2B has finished).
set -e
cd "$(dirname "$0")/.."
G=glossary
log() { echo "[$(date -Is)] $*"; }
log "glossaries, offline (resuming)"
.venv/bin/python $G/glossary.py --all
.venv/bin/python $G/glossary.py > /dev/null
log "training conversations with the glossary"
.venv/bin/python training/prepare.py --glossary
cd training
tmux kill-session -t train-gpu1 2>/dev/null || true
tmux new-session -d -s train-gpu1 "./run_one.sh 1 Qwen/Qwen3.5-0.8B qwen35-0.8b-glossary --full \
  --init-from runs/qwen35-0.8b-full/adapters/final --data data/examples-glossary.parquet --lr 1e-5; read"
log "started"
