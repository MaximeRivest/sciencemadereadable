#!/usr/bin/env bash
# The 0.8B with the reference glossary, end to end (started 2026-10-01):
#  1. wait for the offline Wikipedia download, build the local reference database
#  2. rebuild every paper's glossary offline (replaces the slower online run)
#  3. write the training conversations with the glossary as an input
#  4. stop the 2B (resumable later from its last checkpoint) and train the 0.8B on GPU 1:
#     all weights, starting from the finished 0.8B, one pass, gentler learning rate
set -e
cd "$(dirname "$0")/.."
G=glossary
log() { echo "[$(date -Is)] $*"; }
log "waiting for the Wikipedia download"
until grep -q "all downloaded" $G/data/offline/download.log 2>/dev/null; do sleep 60; done
[ "$(ls $G/data/offline/enwiki-*.json.bz2 | wc -l)" = 66 ] || { log "download incomplete"; exit 1; }
[ -e $G/data/offline/reference.sqlite ] || .venv/bin/python $G/build_offline.py
log "glossaries, offline"
tmux kill-session -t glossary-all 2>/dev/null || true
mkdir -p $G/out-online && mv $G/out/PMC*.json $G/out-online/ 2>/dev/null || true
.venv/bin/python $G/glossary.py --all
.venv/bin/python $G/glossary.py > /dev/null          # the 3 test papers' review.md
log "training conversations with the glossary"
.venv/bin/python training/prepare.py --glossary
log "stopping the 2B, starting the 0.8B with the glossary"
tmux kill-session -t train-gpu1 2>/dev/null || true
sleep 20
cd training
tmux new-session -d -s train-gpu1 "./run_one.sh 1 Qwen/Qwen3.5-0.8B qwen35-0.8b-glossary --full \
  --init-from runs/qwen35-0.8b-full/adapters/final --data data/examples-glossary.parquet --lr 1e-5; read"
log "started"
