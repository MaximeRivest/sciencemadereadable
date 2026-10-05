#!/usr/bin/env bash
# Score training checkpoints with benchmark v0.3 on 3 test papers, one after the other.
#   bench.sh RUN STEP [RUN STEP ...]      e.g. bench.sh qwen35-4b step-00076 qwen35-0.8b-full step-00228
# For each: the checkpoint becomes a GGUF file served by llama.cpp on the CPU (the GPUs
# are training), the student rewrites the papers, the Opus judge scores them.
# Results: runs/RUN/benchmark/STEP.json (shown on the dashboard).
cd "$(dirname "$0")"
LLAMA=/nix/store/8zgm2wxi6s5vxx8q334h0hgpx3wbq7ab-llama-cpp-99999/bin/llama-server
ROOT=$(cd .. && pwd)
while [ $# -ge 2 ]; do
  RUN=$1; STEP=$2; shift 2
  NAME="student-$RUN-$STEP"
  OUT="runs/$RUN/benchmark/$STEP.json"
  [ -e "$OUT" ] && { echo "$NAME: already scored"; continue; }
  BASE=$(python3 -c "import json;print(json.load(open('runs/$RUN/config.json'))['model'])")
  GGUF="exports/$RUN-$STEP.gguf"
  echo "[$(date -Is)] $NAME: exporting"
  .venv/bin/python export_gguf.py --base "$BASE" --weights "runs/$RUN/adapters/$STEP" --out "$GGUF" > "logs/bench-$NAME.log" 2>&1 || { echo "export failed (logs/bench-$NAME.log)"; continue; }
  CUDA_VISIBLE_DEVICES="" "$LLAMA" -m "$GGUF" --port 8011 --host 127.0.0.1 -ngl 0 -t 24 -tb 24 -np 8 -c 196608 \
    --jinja --reasoning off --temp 0 >> "logs/bench-$NAME.log" 2>&1 &
  SERVER=$!
  for i in $(seq 120); do curl -sf localhost:8011/health > /dev/null && break; sleep 3; done
  echo "[$(date -Is)] $NAME: rewriting 3 papers on the CPU"
  (cd "$ROOT" && .venv/bin/python training/eval_student.py generate "$NAME" --papers 3) 2>&1 | grep -v -i warn | tee -a "logs/bench-$NAME.log"; GEN=${PIPESTATUS[0]}
  kill $SERVER; wait $SERVER 2>/dev/null
  rm -f "$GGUF"
  [ "$GEN" = 0 ] || { echo "[$(date -Is)] $NAME: rewriting failed, not judged"; continue; }
  echo "[$(date -Is)] $NAME: judging"
  (cd "$ROOT" && .venv/bin/python training/eval_student.py judge "$NAME" --papers 3 --out "training/$OUT") 2>&1 | grep -v -i warn | tee -a "logs/bench-$NAME.log"
done
