"""Train a student model (LoRA) on the rewrite conversations.

    training/.venv/bin/python training/train.py --model Qwen/Qwen3.5-4B --name qwen35-4b
    (normally started by training/start.sh, one run per GPU)

Plain next-token training on the teacher's answers only (the prompt is not
graded), one conversation per forward pass (no padding), several conversations
per optimizer step, the loss averaged over every answer token of the step.

Written for long examples (up to ~25k tokens) on a 24 GB RTX 3090:
- base weights in bf16, frozen; a LoRA adapter on every linear layer;
- gradient checkpointing; the vocabulary scores (248k words) computed in chunks
  of answer tokens and never all kept at once;
- Qwen3.5's linear-attention layers run on the flash-linear-attention kernels
  (refuses to start if they are missing: the fallback is >10x slower).

Everything a dashboard needs is in runs/<name>/:
  config.json   the settings, sizes and planned number of steps
  log.jsonl     one line per optimizer step ("train") and per validation ("validation")
  status.json   current state, rewritten every step
  last/         the newest checkpoint (adapter + optimizer + position): restarting resumes from it
  adapters/step-N/   the adapter at each validation, to generate and score later
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib")   # NixOS: Triton finds libcuda here
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import inspect
import json
import math
import random
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "examples.parquet"
RUNS = HERE / "runs"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- data

def encode(tokenizer, prompt: list[dict], answer: str) -> tuple[list[int], int]:
    """Token ids of prompt + answer + end of turn, and where the answer starts
    (the same rule as functai.bake.sft.example_ids)."""
    kwargs = {"enable_thinking": False}
    head = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True,
                                         return_dict=False, **kwargs)
    full = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": answer}], tokenize=True,
                                         return_dict=False, **kwargs)
    head, full = list(head), list(full)
    if full[:len(head)] != head:
        full = head + tokenizer(answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    return full, len(head)


def load_examples(tokenizer, max_tokens: int, log, data: Path = None) -> tuple[list[dict], list[dict], dict]:
    """Tokenized examples, cached per tokenizer (all Qwen3.5 sizes share one)."""
    import hashlib
    DATA = data or globals()["DATA"]
    same_tokenizer = hashlib.sha256(f"{tokenizer.chat_template}{len(tokenizer)}{tokenizer.eos_token_id}".encode())
    cache = HERE / "data" / f"tokens-{DATA.stem}-{same_tokenizer.hexdigest()[:12]}-{DATA.stat().st_mtime_ns}.pt"
    if cache.exists():
        rows = torch.load(cache)
    else:
        log(f"tokenizing {DATA.name} (cached afterwards in {cache.name})")
        frame = pd.read_parquet(DATA)
        rows = []
        for r in frame.itertuples():
            ids, start = encode(tokenizer, json.loads(r.prompt), r.answer)
            rows.append(dict(ids=torch.tensor(ids, dtype=torch.int32), start=start, split=r.split, step=r.step,
                             section=r.section, writer=r.writer, paper_id=r.paper_id))
        torch.save(rows, cache)
    too_long = [r for r in rows if len(r["ids"]) > max_tokens]
    rows = [r for r in rows if len(r["ids"]) <= max_tokens]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    info = {"train_conversations": len(train), "validation_conversations": len(val),
            "dropped_too_long": len(too_long), "max_tokens": max_tokens,
            "train_tokens": int(sum(len(r["ids"]) for r in train)),
            "train_answer_tokens": int(sum(len(r["ids"]) - r["start"] for r in train))}
    return train, val, info


# ---------------------------------------------------------------- model

def load_model(name: str, lora_rank: int, device: str, full: bool = False, init_from: str | None = None,
               init_adapter: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    kernel = inspect.getclosurevars(qwen.torch_chunk_gated_delta_rule).nonlocals.get("is_new_implementation")
    if not kernel:
        raise SystemExit("flash-linear-attention is not in use (Triton could not start?): refusing the slow path")
    tokenizer = AutoTokenizer.from_pretrained(name)
    # init_from: a whole model saved by an earlier full-training run, to continue from it
    model = AutoModelForCausalLM.from_pretrained(init_from or name, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    n_params = sum(p.numel() for p in model.parameters())
    if full:                                 # every weight trains: fp32 master weights, bf16 compute
        model = model.float()
    else:
        if init_adapter:                     # continue training an adapter saved by an earlier run
            model = peft.PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        else:
            model = peft.get_peft_model(model, peft.LoraConfig(
                r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0, bias="none",
                target_modules="all-linear", task_type="CAUSAL_LM"))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()      # the adapter trains in fp32; the frozen base stays bf16
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, tokenizer, n_params, trainable


def _chunk_loss(head_weight, hidden, target):
    logits = F.linear(hidden, head_weight).float()
    return F.cross_entropy(logits, target, reduction="sum")


def answer_loss_sum(model, ids: torch.Tensor, start: int, chunk: int = 2048) -> tuple[torch.Tensor, int]:
    """Sum of the cross-entropy over the answer tokens of one conversation, and
    their number. Vocabulary scores are computed chunk by chunk and recomputed in
    the backward pass, so at most one chunk of them is ever in memory."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = base.model(input_ids=ids[None]).last_hidden_state[0]
    hidden = hidden[start - 1:-1]
    target = ids[start:].long()
    weight = base.get_output_embeddings().weight
    total = hidden.new_zeros((), dtype=torch.float32)
    for a in range(0, len(target), chunk):
        total = total + checkpoint(_chunk_loss, weight, hidden[a:a + chunk], target[a:a + chunk],
                                   use_reentrant=False)
    return total, len(target)


@torch.no_grad()
def validate(model, val: list[dict], device: str) -> dict:
    model.eval()
    sums: dict[str, list[float]] = {}
    for r in val:
        loss, n = answer_loss_sum(model, r["ids"].to(device), r["start"])
        for key in ("all", r["step"], r["writer"], f"{r['step']}/{r['writer']}"):
            s = sums.setdefault(key, [0.0, 0])
            s[0] += loss.item()
            s[1] += n
    model.train()
    return {k: s[0] / s[1] for k, s in sums.items()}


# ---------------------------------------------------------------- schedule

def learning_rate(step: int, total: int, peak: float, warmup: int, decay_share: float) -> float:
    """Warmup, constant, then a linear decay to 10% over the last `decay_share` of
    the steps (warmup-stable-decay)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int(total * (1 - decay_share))
    if step < decay_start:
        return peak
    return peak * (1 - 0.9 * (step - decay_start) / max(1, total - decay_start))


# ---------------------------------------------------------------- checkpoints

def save_weights(model, folder: Path):
    """The adapter (LoRA), or the whole model in bf16 (full training)."""
    if hasattr(model, "peft_config"):
        model.save_pretrained(folder)
    else:
        model.save_pretrained(folder, state_dict={k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})


def save_adapter(model, folder: Path):
    tmp = folder.with_name(folder.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    save_weights(model, tmp)
    shutil.rmtree(folder, ignore_errors=True)
    tmp.rename(folder)


def save_last(model, optimizer, state: dict, run: Path):
    tmp = run / "last.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    if hasattr(model, "peft_config"):
        model.save_pretrained(tmp / "adapter")
    else:                                    # full training: the fp32 master weights, to resume exactly
        tmp.mkdir(parents=True)
        torch.save(model.state_dict(), tmp / "weights.pt")
    torch.save({"optimizer": optimizer.state_dict(), "state": state,
                "rng": {"torch": torch.get_rng_state(), "python": random.getstate()}}, tmp / "trainer.pt")
    shutil.rmtree(run / "last", ignore_errors=True)
    tmp.rename(run / "last")


def resume(model, optimizer, run: Path):
    import peft
    last = run / "last"
    if not (last / "trainer.pt").exists():
        return None
    if hasattr(model, "peft_config"):
        peft.set_peft_model_state_dict(model, peft.load_peft_weights(str(last / "adapter")))
    else:
        model.load_state_dict(torch.load(last / "weights.pt", map_location="cpu"))
    saved = torch.load(last / "trainer.pt", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rng"]["torch"])
    random.setstate(saved["rng"]["python"])
    return saved["state"]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lr", type=float, default=None, help="default 2e-4 (LoRA), 2e-5 (--full)")
    ap.add_argument("--full", action="store_true", help="train every weight instead of a LoRA adapter")
    ap.add_argument("--init-from", default=None, help="start from this saved whole model (full training)")
    ap.add_argument("--init-adapter", default=None, help="continue training this saved LoRA adapter")
    ap.add_argument("--data", default=None, help="examples file (default data/examples.parquet)")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--conversations-per-step", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=24_576)
    ap.add_argument("--evals", type=int, default=12, help="validations over the run (plus one before training)")
    ap.add_argument("--checkpoint-minutes", type=float, default=30)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--validation-limit", type=int, default=None)
    ap.add_argument("--longest-first", action="store_true", help="smoke test: start with the longest conversations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "adapters" / "final").exists():
        print(f"{args.name} has already finished ({run / 'adapters' / 'final'})")
        return
    device = "cuda:0"                      # the GPU is chosen with CUDA_VISIBLE_DEVICES
    logf = (run / "log.jsonl").open("a")

    def log_event(**event):
        logf.write(json.dumps({"at": now(), **event}) + "\n")
        logf.flush()

    def say(text):
        print(f"[{now()}] {text}", flush=True)
        log_event(kind="message", text=text)

    def status(**fields):
        (run / "status.json.tmp").write_text(json.dumps({"name": args.name, "updated": now(), **fields}, indent=1))
        (run / "status.json.tmp").replace(run / "status.json")

    status(state="loading")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.lr = args.lr or (2e-5 if args.full else 2e-4)
    model, tokenizer, n_params, trainable = load_model(args.model, args.lora_rank, device, args.full,
                                                       args.init_from, args.init_adapter)
    data = Path(args.data).resolve() if args.data else DATA
    train, val, info = load_examples(tokenizer, args.max_tokens, say, data)
    if args.validation_limit:
        val = random.Random(1).sample(val, min(args.validation_limit, len(val)))

    order: list[int] = []
    for epoch in range(math.ceil(args.epochs)):
        o = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(o)
        order += o
    order = order[:int(len(train) * args.epochs)]
    if args.longest_first:
        order = sorted(order, key=lambda i: -len(train[i]["ids"]))
    per_step = args.conversations_per_step
    total_steps = math.ceil(len(order) / per_step)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    eval_every = max(1, total_steps // args.evals)
    warmup = max(1, min(30, total_steps // 20))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    state = resume(model, optimizer, run) or {"step": 0, "seen_tokens": 0, "seen_answer_tokens": 0,
                                               "train_seconds": 0.0}
    gpu = torch.cuda.get_device_name(0)
    config = {"model": args.model, "name": args.name, "parameters": n_params, "trainable_parameters": trainable,
              "method": "full" if args.full else "lora", "lora_rank": None if args.full else args.lora_rank, "lr": args.lr, "conversations_per_step": per_step,
              "epochs": args.epochs, "total_steps": total_steps, "eval_every": eval_every, "warmup": warmup,
              "schedule": "warmup-stable-decay (last 15% linear to 10%)", "gpu": gpu,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "host": socket.gethostname(),
              "data": str(data), "init_from": args.init_from or args.init_adapter, **info, "started": now(), "args": vars(args)}
    (run / "config.json").write_text(json.dumps(config, indent=1))
    say(f"{args.model}: {n_params / 1e9:.2f}B parameters, {'all weights' if args.full else 'LoRA'} {trainable / 1e6:.0f}M trainable, on {gpu}; "
        f"{len(train):,} conversations, {total_steps} steps of {per_step}"
        + (f"; resuming at step {state['step']}" if state["step"] else ""))

    eval_seconds = 0.0
    if state["step"] == 0 and not args.max_steps:
        status(state="validating", step=0, total_steps=total_steps)
        t = time.time()
        v = validate(model, val, device)
        eval_seconds = time.time() - t
        log_event(kind="validation", step=0, seen_answer_tokens=0, seconds=round(time.time() - t, 1), **v)
        say(f"step 0 validation loss {v['all']:.4f} ({time.time() - t:.0f}s)")

    model.train()
    last_checkpoint = time.time()
    recent: list[tuple[float, int]] = []          # (seconds, tokens) of recent steps, for speed and ETA
    while state["step"] < total_steps:
        step = state["step"]
        lr = learning_rate(step, total_steps, args.lr, warmup, 0.15)
        for g in optimizer.param_groups:
            g["lr"] = lr
        batch = [train[i] for i in order[step * per_step:(step + 1) * per_step]]
        n_answer = sum(len(r["ids"]) - r["start"] for r in batch)
        n_tokens = sum(len(r["ids"]) for r in batch)
        t = time.time()
        loss_sum = 0.0
        parts: dict[str, list[float]] = {}
        skipped = 0
        for r in batch:
            try:
                loss, n = answer_loss_sum(model, r["ids"].to(device), r["start"])
                (loss / n_answer).backward()
            except torch.OutOfMemoryError:
                loss = n = None
                torch.cuda.empty_cache()
                skipped += 1
                say(f"out of memory on a {len(r['ids']):,}-token conversation ({r['paper_id']} {r['section']}): skipped")
                continue
            loss_sum += loss.item()
            p = parts.setdefault(r["step"], [0.0, 0])
            p[0] += loss.item()
            p[1] += n
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        seconds = time.time() - t
        state["step"] += 1
        state["seen_tokens"] += n_tokens
        state["seen_answer_tokens"] += n_answer
        state["train_seconds"] += seconds
        recent = (recent + [(seconds, n_tokens)])[-20:]
        rate = sum(s for s, _ in recent) / len(recent)
        log_event(kind="train", step=state["step"], loss=loss_sum / n_answer, lr=lr, grad_norm=grad_norm,
                  seconds=round(seconds, 2), tokens=n_tokens, answer_tokens=n_answer,
                  tokens_per_second=round(n_tokens / seconds), seen_tokens=state["seen_tokens"],
                  seen_answer_tokens=state["seen_answer_tokens"],
                  max_memory_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                  **{f"loss_{k}": v[0] / v[1] for k, v in parts.items()})
        remaining_evals = (total_steps - state["step"]) // eval_every
        status(state="training", step=state["step"], total_steps=total_steps, loss=loss_sum / n_answer, lr=lr,
               seconds_per_step=round(rate, 1), train_hours=round(state["train_seconds"] / 3600, 2),
               eta_hours=round(((total_steps - state["step"]) * rate + remaining_evals * eval_seconds) / 3600, 2),
               max_memory_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))

        done = state["step"] == total_steps
        if state["step"] % eval_every == 0 or done:
            status(state="validating", step=state["step"], total_steps=total_steps)
            t = time.time()
            v = validate(model, val, device)
            eval_seconds = time.time() - t
            log_event(kind="validation", step=state["step"], seen_answer_tokens=state["seen_answer_tokens"],
                      seconds=round(time.time() - t, 1), **v)
            say(f"step {state['step']}/{total_steps} validation loss {v['all']:.4f} ({time.time() - t:.0f}s)")
            save_adapter(model, run / "adapters" / f"step-{state['step']:05d}")
        if done or time.time() - last_checkpoint > args.checkpoint_minutes * 60:
            save_last(model, optimizer, state, run)
            last_checkpoint = time.time()

    save_adapter(model, run / "adapters" / "final")
    status(state="finished", step=state["step"], total_steps=total_steps,
           train_hours=round(state["train_seconds"] / 3600, 2))
    say("finished")


if __name__ == "__main__":
    main()
