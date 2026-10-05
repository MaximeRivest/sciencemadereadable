#!/usr/bin/env bash
# bench_vllm.sh RUN NAME [--glossary]: serve runs/RUN's final model with vLLM on GPU 1, rewrite the
# 3 benchmark papers, judge them. Leaves other GPU 1 services alone (uses at most 65% of the GPU).
cd "$(dirname "$0")"
RUN=$1; NAME=$2; EXTRA=$3
podman rm -f student-llm >/dev/null 2>&1
podman run -d --name student-llm --device nvidia.com/gpu=1 --ipc=host --network=host \
  -v "$PWD/exports/$RUN-final-hf:/model:ro" fba1a5021627 --model /model --served-model-name local \
  --host 127.0.0.1 --port 8012 --max-model-len 32768 --max-num-seqs 32 --gpu-memory-utilization 0.65 \
  --enable-prefix-caching --override-generation-config '{"temperature": 0.0}' --default-chat-template-kwargs '{"enable_thinking":false}' >/dev/null
for i in $(seq 90); do curl -sf -m 5 localhost:8012/v1/models >/dev/null && break; sleep 5; done
rm -rf ../model_baselines/rewrites/$NAME; podman logs student-llm 2>&1 | grep -i -m1 "sampling param"
(cd .. && .venv/bin/python training/eval_student.py generate $NAME --papers 3 --port 8012 $EXTRA) 2>&1 | grep -v -i warn
podman stop student-llm >/dev/null
mkdir -p runs/$RUN/benchmark
(cd .. && .venv/bin/python training/eval_student.py judge $NAME --papers 3 --out training/runs/$RUN/benchmark/final${EXTRA:+-glossary}.json) 2>&1 | grep -v -i warn
