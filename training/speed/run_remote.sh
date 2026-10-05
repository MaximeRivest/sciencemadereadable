#!/usr/bin/env bash
# run_remote.sh HOST LABEL LOCALPORT: speed of the three students on a rented 1-GPU machine.
# Models are expected in ~/models/{m08,m4,m9,m4-mtp,m9-mtp} on HOST (see upload in the README);
# vLLM 0.29.0 (same version as on lambda) runs there in Docker. The client (bench_speed.py)
# runs on lambda through an SSH tunnel; every request is a few hundred ms of network at most,
# against seconds to minutes of generation.
# For each model: latency (3 test papers, one at a time) and throughput (128 papers at once),
# then the MTP variants of the 4B and 9B for latency only.
H=srl@$1; LABEL=$2; LP=$3
cd "$(dirname "$0")/../.."
LOG=training/speed/$LABEL.log
SETTINGS='--dtype bfloat16 --max-model-len 65536 --max-num-seqs 512 --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.92 --enable-prefix-caching --override-generation-config {"temperature":0.0}
  --default-chat-template-kwargs {"enable_thinking":false}'

ssh -n -f -N -o ExitOnForwardFailure=yes -L $LP:127.0.0.1:8000 $H
local_size() { stat -c %s "$1"; }
wait_upload() {  # wait_upload DIR LOCALFILE
  for i in $(seq 240); do
    r=$(ssh -n $H "stat -c %s ~/models/$1/model.safetensors 2>/dev/null; pgrep -f '[r]sync.*models/$1' >/dev/null && echo busy")
    [ "$r" = "$(local_size $2)" ] && return 0; sleep 15
  done; return 1
}
serve() {  # serve DIR [SPEC]
  ssh -n $H "until grep -q pulled ~/pull.log; do sleep 5; done; sudo docker rm -f llm >/dev/null 2>&1
    sudo docker run -d --name llm --gpus all --ipc=host -p 127.0.0.1:8000:8000 -v ~/models:/models:ro \
      vllm/vllm-openai:v0.29.0 --model /models/$1 --served-model-name local --host 0.0.0.0 --port 8000 \
      $(printf '%q ' $SETTINGS) ${2:+--speculative-config $(printf '%q' "$2")} >/dev/null"
  for i in $(seq 120); do curl -sf -m 5 localhost:$LP/v1/models >/dev/null && break; sleep 5; done
  curl -sf -m 5 localhost:$LP/v1/models >/dev/null || { echo "server $1 failed" | tee -a $LOG; ssh -n $H "sudo docker logs llm 2>&1 | tail -20" >> $LOG; return 1; }
  # warm-up request, so the first timed paper does not pay for any first-call setup
  curl -s -m 300 localhost:$LP/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"local","messages":[{"role":"user","content":"Say hello."}],"max_tokens":20}' >/dev/null
}
bench() {  # bench MODE NAME
  .venv/bin/python training/bench_speed.py $1 $2 $LP 2>&1 \
    | grep --line-buffered -v -i warn | tee -a $LOG
}
T=training
echo "[$(date -Is)] $LABEL $(ssh -n $H 'nvidia-smi --query-gpu=name --format=csv,noheader')" | tee -a $LOG
for m in m08:0.8b:$T/exports/qwen35-0.8b-glossary-scratch-final-hf m4:4b:$T/exports/qwen35-4b-glossary-scratch-final-hf \
         m9:9b:$T/runs/qwen35-9b-full-glossary/models/final; do
  IFS=: read dir size path <<< "$m"
  wait_upload $dir $path/model.safetensors || { echo "upload of $dir never finished" | tee -a $LOG; continue; }
  echo "[$(date -Is)] $size" | tee -a $LOG
  serve $dir || continue
  bench latency $LABEL-$size
  bench throughput $LABEL-$size
  ssh -n $H "sudo docker logs llm 2>&1 | grep -E 'KV cache size|Maximum concurrency' | tail -2" | cut -c1-200 >> $LOG
done
for m in m4-mtp:4b m9-mtp:9b; do
  IFS=: read dir size <<< "$m"
  echo "[$(date -Is)] $size + MTP" | tee -a $LOG
  serve $dir '{"method":"mtp","num_speculative_tokens":2}' || continue
  bench latency $LABEL-$size-mtp
  ssh -n $H "sudo docker logs llm 2>&1 | grep 'Mean acceptance length' | tail -3" | cut -c1-200 >> $LOG
done
ssh -n $H "sudo docker rm -f llm" >/dev/null
echo "[$(date -Is)] $LABEL done" | tee -a $LOG
