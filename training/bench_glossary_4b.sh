#!/usr/bin/env bash
# 4B step 152 with the reference glossary (no retraining), same 3 papers.
cd "$(dirname "$0")"
NAME=student-qwen35-4b-step-00152-glossary
LLAMA=/nix/store/8zgm2wxi6s5vxx8q334h0hgpx3wbq7ab-llama-cpp-99999/bin/llama-server
.venv/bin/python export_gguf.py --base Qwen/Qwen3.5-4B --weights runs/qwen35-4b/adapters/step-00152 --out exports/qwen35-4b-step-00152.gguf > logs/bench-$NAME.log 2>&1
CUDA_VISIBLE_DEVICES="" $LLAMA -m exports/qwen35-4b-step-00152.gguf --port 8011 --host 127.0.0.1 -ngl 0 -t 24 -tb 24 -np 8 -c 196608 --jinja --reasoning off --temp 0 >> logs/bench-$NAME.log 2>&1 &
SERVER=$!
for i in $(seq 120); do curl -sf localhost:8011/health > /dev/null && break; sleep 3; done
echo "[$(date -Is)] $NAME: rewriting 3 papers on the CPU"
(cd .. && .venv/bin/python training/eval_student.py generate $NAME --papers 3 --glossary) 2>&1 | grep -v -i warn
GEN=${PIPESTATUS[0]}
kill $SERVER; wait $SERVER 2>/dev/null; rm -f exports/qwen35-4b-step-00152.gguf
[ "$GEN" = 0 ] || { echo "rewriting failed"; exit 1; }
echo "[$(date -Is)] $NAME: judging"
(cd .. && .venv/bin/python training/eval_student.py judge $NAME --papers 3 --out training/runs/qwen35-4b/benchmark/step-00152-glossary.json) 2>&1 | grep -v -i warn
