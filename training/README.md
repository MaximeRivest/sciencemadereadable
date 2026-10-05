# Training the student model

One small model learns both steps of the rewrite recipe (opening, then each section),
by plain next-token training on the Opus and GPT-6 Astra rewrites (LoRA).

| file | what |
|---|---|
| `prepare.py` | builds `data/examples.parquet` with dpyr + functai (main venv): the exact prompts functai will send the student |
| `train.py` | one training run (this folder's `.venv`: torch, transformers, peft, flash-linear-attention) |
| `start.sh` / `stop.sh` | start or resume all runs and the dashboard in tmux / stop the runs |
| `gpu_services.sh stop\|start\|status` | pause or bring back InkType, Chattering semantic search, Kokoro, Parakeet |
| `dashboard.py` | https://lambda.tail69222b.ts.net:18790 |
| `functai_feedback.md` | what functai's `bake` needs to run this job itself |

First runs (2026-10-01): data snapshot of 3,159 papers (15,089 conversations; 40
papers held out for validation; the 500 test papers were never in the corpus).
GPU 0: Qwen3.5-4B. GPU 1: Qwen3.5-0.8B, then Qwen3.5-2B. Same settings for all:
LoRA rank 32 on every linear layer, lr 2e-4 warmup-stable-decay, 16 conversations
per step, one pass, conversations over 24,576 tokens left out (126).

Each run writes `runs/<name>/`: `log.jsonl`, `status.json`, `last/` (resume point,
every 30 min) and `adapters/step-N/` (at each of the 12 validations) to generate
and score with the benchmark afterwards.
