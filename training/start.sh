#!/usr/bin/env bash
# Start (or resume) the training runs and the dashboard, each in its own tmux session.
# Safe to run again: a running session is left alone, a stopped run resumes from its
# last checkpoint (at most 30 minutes lost), a finished run is skipped.
#   GPU 0: Qwen3.5-4B
#   GPU 1: Qwen3.5-0.8B with every weight trained, then Qwen3.5-2B (LoRA)
#   (the 0.8B LoRA run was stopped by hand at step 215; it is not restarted)
# Dashboard: https://lambda.tail69222b.ts.net:18790
cd "$(dirname "$0")"
mkdir -p runs logs
cat > runs/plan.json <<'JSON'
[{"name": "qwen35-9b-full-glossary", "model": "Qwen/Qwen3.5-9B · full + glossary (Nebius B300)", "gpu": "cloud"},
 {"name": "qwen35-4b-glossary-scratch", "model": "Qwen/Qwen3.5-4B · LoRA + glossary, from original weights", "gpu": "0"},
 {"name": "qwen35-4b-glossary", "model": "Qwen/Qwen3.5-4B · LoRA + glossary, continued (stopped)", "gpu": "0"},
 {"name": "qwen35-4b",  "model": "Qwen/Qwen3.5-4B",   "gpu": "0"},
 {"name": "qwen35-0.8b-glossary-scratch", "model": "Qwen/Qwen3.5-0.8B · full + glossary, from original weights", "gpu": "1"},
 {"name": "qwen35-0.8b-glossary", "model": "Qwen/Qwen3.5-0.8B · full + glossary, continued (stopped)", "gpu": "1"},
 {"name": "qwen35-0.8b-full", "model": "Qwen/Qwen3.5-0.8B · full", "gpu": "1"},
 {"name": "qwen35-2b",  "model": "Qwen/Qwen3.5-2B",   "gpu": "1"},
 {"name": "qwen35-0.8b", "model": "Qwen/Qwen3.5-0.8B · LoRA", "gpu": "1"}]
JSON
start() {  # session, command
  if tmux has-session -t "$1" 2>/dev/null; then echo "$1 already running"; return; fi
  tmux new-session -d -s "$1" "$2; echo; echo 'Ended. Press Enter to close.'; read"
  echo "started $1"
}
start train-gpu0 "./run_one.sh 0 Qwen/Qwen3.5-4B qwen35-4b-glossary-scratch --data data/examples-glossary.parquet --lr 2e-4"
if [ -e data/examples-glossary.parquet ]; then
  start train-gpu1 "./run_one.sh 1 Qwen/Qwen3.5-0.8B qwen35-0.8b-glossary-scratch --full --data data/examples-glossary.parquet --lr 2e-5"
else
  start train-gpu1 "./run_one.sh 1 Qwen/Qwen3.5-0.8B qwen35-0.8b-full --full && ./run_one.sh 1 Qwen/Qwen3.5-2B qwen35-2b"
fi
start train-dashboard "python3 dashboard.py 8790"
echo "Dashboard: https://lambda.tail69222b.ts.net:18790   Watch a run: tmux attach -t train-gpu0"
