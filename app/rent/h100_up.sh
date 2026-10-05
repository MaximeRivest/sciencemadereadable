#!/usr/bin/env bash
# Rent one H100 on Nebius and make it the GPU of sciencemadereadable.com.
#   app/rent/h100_up.sh          (then app/rent/h100_down.sh to stop paying)
# Cost: Nebius on-demand H100 about $4.50/hour, about $108 a day (the disk adds a few cents).
#
# What it does:
#   1. creates the machine (eu-north1, 1 x H100, Ubuntu with CUDA);
#   2. starts our 9B there, downloaded from Hugging Face at a fixed version (the same weights,
#      checked by fingerprint, that the home GPU served), and our 0.8B, uploaded from here (2 GB);
#      vLLM 0.29.0 and the settings of the scored runs (bf16, greedy, MTP for the 9B);
#   3. opens an SSH tunnel from lambda to it (the machine opens no port to the internet);
#   4. starts the worker for it on lambda (tmux "smr-worker-h100"). The home worker (the 9B on GPU 0)
#      keeps working too: the queue gives each paper to whichever worker has a free place and the model.
# The queue, the paper service and the public door (Tailscale Funnel) stay on lambda.
set -euo pipefail
export PATH=$HOME/.nebius/bin:$PATH
APP=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$APP/.." && pwd)
P=project-e00fbrcapr00yr015zfmw3                     # eu-north1 (H100 on-demand)
SUBNET=vpcsubnet-e00bxcq5ftx3hr16bb
NAME=smr-h100
M9=maximerivest/qwen3.5-9b-paper-rewriter
M9_REV=213d5dd67759fb9b0c91882238bd3fd5f9a74f0a       # weights checked against the served model, 2026-10-05
M08=$ROOT/training/exports/qwen35-0.8b-glossary-scratch-final-hf
VLLM=vllm/vllm-openai:v0.29.0
# (the JSON is quoted for the rented machine's shell, which reads these commands)
COMMON="--dtype bfloat16 --max-model-len 65536 --enable-prefix-caching --override-generation-config '{\"temperature\":0.0}' --default-chat-template-kwargs '{\"enable_thinking\":false}'"

cat > /tmp/smr-cloudinit.yaml <<CI
#cloud-config
users:
  - name: srl
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat ~/.ssh/id_ed25519.pub)
CI
echo "[$(date +%T)] creating $NAME (1 x H100, on-demand)"
nebius compute instance get-by-name --parent-id $P --name $NAME >/dev/null 2>&1 || \
nebius compute instance create --parent-id $P --name $NAME \
  --resources-platform gpu-h100-sxm --resources-preset 1gpu-16vcpu-200gb --on-demand \
  --boot-disk-attach-mode read_write --boot-disk-managed-disk-name $NAME-boot \
  --boot-disk-managed-disk-size-gibibytes 200 --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-source-image-family-image-family ubuntu24.04-cuda13.0 \
  --network-interfaces "[{\"name\": \"eth0\", \"ip_address\": {}, \"public_ip_address\": {}, \"subnet_id\": \"$SUBNET\"}]" \
  --cloud-init-user-data "$(cat /tmp/smr-cloudinit.yaml)" --async --format json >/dev/null
for i in $(seq 60); do
  read STATE IP < <(nebius compute instance get-by-name --parent-id $P --name $NAME --format json | python3 -c "
import sys,json;d=json.load(sys.stdin);s=d.get('status',{});ip=[n.get('public_ip_address',{}).get('address','').split('/')[0] for n in s.get('network_interfaces',[])]
print(s.get('state'), (ip or [''])[0] or '-')")
  echo "[$(date +%T)] $STATE $IP"; [ "$STATE" = RUNNING ] && [ "$IP" != - ] && break
  [ "$STATE" = STOPPED ] && [ $i -gt 10 ] && { echo "Nebius couldn't start it (no H100 free?). Run h100_down.sh, try later or another region."; exit 1; }
  sleep 15
done
H=srl@$IP; SSH="ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes"
until $SSH -n $H true 2>/dev/null; do sleep 10; done
echo "$IP" > "$APP/rent/h100_ip"

echo "[$(date +%T)] pulling vLLM and downloading the 9B there; uploading the 0.8B from here"
$SSH -n $H "mkdir -p ~/m08 ~/hf; ( (sudo docker pull -q $VLLM && \
  sudo docker run --rm -v ~/hf:/root/.cache/huggingface --entrypoint python3 $VLLM -c \
    'from huggingface_hub import snapshot_download as d; d(\"$M9\", revision=\"$M9_REV\")') \
  && echo READY || echo FAILED) > ~/prep.log 2>&1 &"
rsync -aL -e "$SSH" "$M08/" $H:m08/
until $SSH -n $H 'grep -q READY ~/prep.log' 2>/dev/null; do
  $SSH -n $H 'grep -q FAILED ~/prep.log' 2>/dev/null && { echo "the download failed:"; $SSH -n $H 'tail -20 ~/prep.log'; exit 1; }
  sleep 10
done

# The 9B first (80% of the GPU's memory), then the 0.8B in what is left (12%): vLLM checks free memory at start.
echo "[$(date +%T)] starting the 9B"
$SSH -n $H "sudo docker rm -f llm9 llm08 >/dev/null 2>&1; sudo docker run -d --name llm9 --restart unless-stopped --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 -v ~/hf:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 $VLLM \
  --model $M9 --revision $M9_REV --served-model-name our-9b --host 0.0.0.0 --port 8000 $COMMON \
  --max-num-seqs 128 --gpu-memory-utilization 0.80 --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}' >/dev/null"
$SSH -n $H 'for i in $(seq 120); do curl -sf -m 5 localhost:8000/v1/models >/dev/null && exit 0; sleep 5; done; exit 1' || \
  { echo "the 9B didn't start:"; $SSH -n $H "sudo docker logs llm9 2>&1 | tail -30"; exit 1; }
echo "[$(date +%T)] starting the 0.8B"
$SSH -n $H "sudo docker run -d --name llm08 --restart unless-stopped --gpus all --ipc=host \
  -p 127.0.0.1:8001:8000 -v ~/m08:/model:ro $VLLM \
  --model /model --served-model-name our-0.8b --host 0.0.0.0 --port 8000 $COMMON \
  --max-num-seqs 32 --gpu-memory-utilization 0.12 >/dev/null"
$SSH -n $H 'for i in $(seq 120); do curl -sf -m 5 localhost:8001/v1/models >/dev/null && exit 0; sleep 5; done; exit 1' || \
  { echo "the 0.8B didn't start:"; $SSH -n $H "sudo docker logs llm08 2>&1 | tail -30"; exit 1; }

tmux kill-session -t h100-tunnel 2>/dev/null || true
tmux new -d -s h100-tunnel "while true; do $SSH -N -o ServerAliveInterval=20 -o ExitOnForwardFailure=yes -L 8016:127.0.0.1:8000 -L 8017:127.0.0.1:8001 $H; sleep 3; done"
for i in $(seq 30); do curl -sf -m 5 localhost:8016/v1/models >/dev/null && curl -sf -m 5 localhost:8017/v1/models >/dev/null && break; sleep 2; done
echo '{"name": "h100", "queue": "http://127.0.0.1:8795", "papers": "http://127.0.0.1:8795", "models": {"our-9b": "http://127.0.0.1:8016/v1", "our-0.8b": "http://127.0.0.1:8017/v1"}, "parallel": 24}' > "$APP/worker/config-h100.json"
tmux kill-session -t smr-worker-h100 2>/dev/null || true
tmux new -d -s smr-worker-h100 "cd $APP && WORKER_CONFIG=$APP/worker/config-h100.json node --conditions=functai-source worker/worker.ts 2>&1 | tee -a worker/worker-h100.log"
sleep 5
echo "[$(date +%T)] the H100 is taking jobs (up to 24 papers at once). Stop paying: app/rent/h100_down.sh"
