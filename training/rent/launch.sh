#!/usr/bin/env bash
# On the rented machine (Nebius 8 x B200 spot, Ubuntu with NVIDIA drivers):
#   bash launch.sh        set up once, install a restart-at-boot job, run in tmux session "train"
# Spot: Nebius sends SIGTERM 60 s before stopping the VM. Every GPU process gets it
# (they are started directly, not through torchrun), saves the last finished step, and exits.
# The disk is kept; when the VM is started again, the @reboot job resumes from that save.
# A save cut short by the stop never damages the previous one (written aside, then renamed).
# Watch:  tail -f ~/srl/train.log
set -e
cd ~/srl
if [ -z "$INSIDE" ]; then
  # restart after a reboot (spot machines); optional: some images do not let users use cron
  (crontab -l 2>/dev/null | grep -v srl/launch.sh; echo "@reboot cd ~/srl && bash launch.sh") | crontab - 2>/dev/null \
    || echo "note: no restart-at-boot job (crontab not allowed); fine for an on-demand machine"
  tmux has-session -t train 2>/dev/null && { echo "already running"; exit; }
  tmux new-session -d -s train "INSIDE=1 bash launch.sh 2>&1 | tee -a train.log"; echo "started in tmux 'train'"; exit
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -e .venv/ok ] || { uv venv -p 3.12 .venv; VIRTUAL_ENV=.venv uv pip install torch transformers peft accelerate \
  flash-linear-attention safetensors pyarrow pandas huggingface_hub && touch .venv/ok; }
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
.venv/bin/hf download Qwen/Qwen3.5-9B > /dev/null
N=$(nvidia-smi -L | wc -l)

ranks() {   # ranks NAME ARGS...: one process per GPU; stop signals go to all of them
  local pids=() code=0
  for r in $(seq 0 $((N - 1))); do
    RANK=$r LOCAL_RANK=$r WORLD_SIZE=$N MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
      .venv/bin/python train_multi.py "$@" &
    pids+=($!)
  done
  trap 'kill -TERM "${pids[@]}" 2>/dev/null' TERM INT HUP
  for p in "${pids[@]}"; do wait $p || code=$?; done
  return $code
}

if [ ! -e smoke.ok ]; then
  echo "== $(date -Is) smoke test: 6 steps on $N GPUs"
  rm -rf runs/smoke-9b
  ranks --model Qwen/Qwen3.5-9B --name smoke-9b --data data/examples-glossary.parquet \
    --max-steps 6 --validation-limit 16 --checkpoint-minutes 0
  grep -E '"train"|"checkpoint"' runs/smoke-9b/log.jsonl | tail -4
  rm -rf runs/smoke-9b; touch smoke.ok
  echo "== smoke test passed. Check the seconds per step above before the real run continues."
fi
echo "== $(date -Is) real run: 3 passes (resumes from runs/qwen35-9b-full-glossary/last if present)"
set +e
ranks --model Qwen/Qwen3.5-9B --name qwen35-9b-full-glossary --data data/examples-glossary.parquet \
  --epochs 3 --lr 1e-5 --save-every 4 --checkpoint-minutes 20
code=$?
[ $code = 3 ] && { echo "== $(date -Is) interrupted and saved; resumes at next boot"; exit 3; }
[ $code = 0 ] && { (crontab -l 2>/dev/null | grep -v srl/launch.sh | crontab - 2>/dev/null) || true; echo "== $(date -Is) DONE"; }
exit $code
