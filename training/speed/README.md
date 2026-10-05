# Time and price per paper (2026-10-04)

Our three students (glossary-trained 0.8B, 4B, 9B) on rented Nebius GPUs, against the
three API writers with the v8 recipe they were scored with. `python3 summarize.py`
prints the table from the JSON files here.

| | GPU / route | s per paper | with MTP draft head | papers/h (GPU full) | $ per typical paper |
|---|---|---|---|---|---|
| our 0.8B | RTX PRO 6000 ($1.80/h) | 13 | – | 1,662 | 0.0011 |
| our 0.8B | H100 ($4.50/h) | 9 | – | 3,117 | 0.0014 |
| our 4B | RTX PRO 6000 | 30 | 21 | 502 | 0.0036 |
| our 4B | H100 | 20 | 9 (2 papers) | 1,064 | 0.0042 |
| our 9B | RTX PRO 6000 | 50 | 29 | 392 | 0.0046 |
| our 9B | H100 | 27 | 15 | 807 | 0.0056 |
| Opus 5.5 | API ($4/$20 per M) | 83 | | | 0.97 (batch 0.49) |
| GPT-6 Astra | API via Codex login ($10/$50) | 301 (106–510) | | | 1.21 (batch 0.60) |
| GPT-6 Luna | API via Codex login ($0.10/$0.50) | 82 | | | 0.011 (batch 0.005) |

**Method**
- Time: the 3 benchmark test papers (~4,500 words), one at a time, alone on the server
  (`bench_speed.py latency`; API: `speed_api.py`). Opening first, then the 4 sections at once.
- Students' price: GPU on-demand price / papers per hour with 128 unused corpus papers
  (~7,100 words each) sent at once (`bench_speed.py throughput`). API price: measured tokens
  x list price (standard tier). All prices scaled to a typical 7,100-word paper.
- vLLM 0.29.0 (same as lambda), bf16, greedy, `--max-num-seqs 512 --max-num-batched-tokens 16384
  --gpu-memory-utilization 0.92`, max length 65,536 (32,768 refused ~2.5% of requests on long
  papers: `first-try-32k/`). Orchestration: `run_remote.sh`.
- MTP: the trained checkpoints lost the multi-token-prediction head; `graft_mtp.py` adds the
  base model's head back. Mean acceptance ~2.4 tokens per step. Judge, 9B, 3 papers:
  scored run 7.78, plain H100 rerun 7.88, MTP 7.72 (`judge-speed-*.json`): within run-to-run
  noise. Throughput with MTP was not measured. n-gram speculation garbled text on the 0.8B
  (lambda test) and was dropped.

**Caveats**
- Greedy text is not bit-reproducible across servers (~80-86% similarity between reruns).
- The GPU price assumes the GPU is kept busy; idle hours cost the same.
- Astra and Luna went through the ChatGPT/Codex login; the paid API (or its fast mode) may
  have other speeds. Opus went through the Claude login.
- Not tested: L40S, B300/B200 (likely fastest per paper), FP8 weights.
- 0.8B: ~10 of 640 sections per run ran on to the length limit (counted in time and price).
