"""Full fine-tuning (every weight) on several GPUs: the model, its gradients and the
optimizer state are split across the GPUs (PyTorch FSDP2), so a 9B model fits on 8 GPUs.

    torchrun --nproc_per_node 8 train_multi.py --model Qwen/Qwen3.5-9B --name qwen35-9b-full-glossary \
        --data data/examples-glossary.parquet --epochs 3
    (normally started by rent/launch.sh on the rented machine)

Same training as train.py (answer tokens only, one conversation per forward pass, no
padding, warmup-stable-decay schedule, same validation papers, same log format, so the
dashboard reads it), with these differences:
- each optimizer step's conversations are dealt to the GPUs so every GPU gets the same
  number and about the same number of tokens (longest first, to the least loaded GPU);
  a GPU short of a conversation runs a copy with zero weight, so all GPUs stay in step;
- the loss is the mean over all answer tokens of the step, across all GPUs;
- weights are kept in fp32 (split across GPUs), computed in bf16, gradients averaged in fp32;
- runs/<name>/last/ is a sharded checkpoint (torch.distributed.checkpoint) to resume from,
  written every --checkpoint-minutes; runs/<name>/models/ gets whole models (bf16,
  Hugging Face format) at the end and every --save-every validations.
"""
from __future__ import annotations

import os
from pathlib import Path

if Path("/run/opengl-driver/lib/libcuda.so").exists():                 # NixOS (lambda): Triton finds libcuda here
    os.environ.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import signal
import math
import random
import shutil
import socket
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import train as single            # load_examples, learning_rate, now: shared with the one-GPU script

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
INTERRUPTED = 3          # exit code: stopped by a signal after saving; run again to resume

# Spot machines: the cloud sends SIGTERM before stopping the machine (Nebius: 60 s). The
# handler only raises a flag; the GPUs agree on it after each conversation, save the last
# completed step and exit. A half-done step is dropped (its gradients are not applied).
STOP = {"signal": None}


def _on_signal(signum, frame):
    STOP["signal"] = signum


for _s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
    signal.signal(_s, _on_signal)


def stop_requested(device) -> bool:
    flag = torch.tensor([1.0 if STOP["signal"] else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


# ---------------------------------------------------------------- the model, with the loss inside

def _chunk_loss(weight, hidden, target):
    return F.cross_entropy(F.linear(hidden, weight).float(), target, reduction="sum")


class AnswerLoss(torch.nn.Module):
    """Wraps the causal LM so that the whole loss runs inside one forward call (FSDP2
    gathers the root's embedding / output layer only during the root's forward)."""

    def __init__(self, lm, chunk: int = 2048):
        super().__init__()
        self.lm, self.chunk = lm, chunk

    def forward(self, ids: torch.Tensor, start: int) -> torch.Tensor:
        hidden = self.lm.model(input_ids=ids[None]).last_hidden_state[0][start - 1:-1]
        target = ids[start:].long()
        weight = self.lm.get_output_embeddings().weight
        total = hidden.new_zeros((), dtype=torch.float32)
        for a in range(0, len(target), self.chunk):      # vocabulary scores one chunk at a time, recomputed in backward
            total = total + checkpoint(_chunk_loss, weight, hidden[a:a + self.chunk], target[a:a + self.chunk],
                                       use_reentrant=False)
        return total


def build(model_name: str, device: torch.device, allow_slow: bool):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import inspect
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
    fast = inspect.getclosurevars(qwen.torch_chunk_gated_delta_rule).nonlocals.get("is_new_implementation")
    if not fast and not allow_slow:
        raise SystemExit("flash-linear-attention is not in use (Triton could not start?): refusing the slow path")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, attn_implementation="sdpa")
    lm.config.use_cache = False
    n_params = sum(p.numel() for p in lm.parameters())
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = AnswerLoss(lm)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in lm.model.layers:
        fully_shard(layer, mp_policy=policy)
    fully_shard(model, mp_policy=policy)                 # root: embeddings, final norm, output layer
    model.to(device)
    return model, tokenizer, n_params


# ---------------------------------------------------------------- dealing conversations to GPUs

def deal(batch: list[dict], world: int) -> list[list[tuple[dict, float]]]:
    """Each GPU gets ceil(len/world) (conversation, weight) pairs, balanced by length;
    missing places are zero-weight copies of the shortest conversation."""
    per = math.ceil(len(batch) / world)
    order = sorted(batch, key=lambda r: -len(r["ids"]))
    hands: list[list[tuple[dict, float]]] = [[] for _ in range(world)]
    loads = [0] * world
    for r in order:
        k = min((i for i in range(world) if len(hands[i]) < per), key=lambda i: loads[i])
        hands[k].append((r, 1.0))
        loads[k] += len(r["ids"])
    filler = order[-1]
    for h in hands:
        while len(h) < per:
            h.append((filler, 0.0))
    return hands


# ---------------------------------------------------------------- checkpoints

def save_resume(model, optimizer, state: dict, run: Path, rank: int):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict
    msd, osd = get_state_dict(model, optimizer)
    tmp = run / "last.tmp"
    if rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)
    dist.barrier()
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(tmp))
    if rank == 0:
        (tmp / "state.json").write_text(json.dumps(state))
        shutil.rmtree(run / "last", ignore_errors=True)
        tmp.rename(run / "last")
    dist.barrier()


def load_resume(model, optimizer, run: Path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    if not (run / "last" / "state.json").exists():
        return None
    msd, osd = get_state_dict(model, optimizer)
    sd = {"model": msd, "optim": osd}
    dcp.load(sd, checkpoint_id=str(run / "last"))
    set_state_dict(model, optimizer, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
    return json.loads((run / "last" / "state.json").read_text())


def save_whole(model, tokenizer, folder: Path, rank: int):
    """A normal Hugging Face model folder (bf16), written by GPU 0."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        sd = {k.removeprefix("lm."): v.to(torch.bfloat16) for k, v in sd.items()}
        tmp = folder.with_name(folder.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        model.lm.save_pretrained(tmp, state_dict=sd)
        tokenizer.save_pretrained(tmp)
        shutil.rmtree(folder, ignore_errors=True)
        tmp.rename(folder)
    dist.barrier()


# ---------------------------------------------------------------- validation

@torch.no_grad()
def validate(model, val: list[dict], device, rank: int, world: int) -> dict:
    model.eval()
    keys = ["all", "opening", "section", "opus", "astra", "opening/opus", "opening/astra", "section/opus",
            "section/astra"]
    sums = torch.zeros(2 * len(keys), device=device, dtype=torch.float64)
    per = math.ceil(len(val) / world)
    mine = val[rank * per:(rank + 1) * per]
    mine = [(r, 1.0) for r in mine] + [(val[0], 0.0)] * (per - len(mine))     # same number of forwards everywhere
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for r, w in mine:
            loss = model(r["ids"].to(device), r["start"]).double() * w
            n = (len(r["ids"]) - r["start"]) * w
            for key in ("all", r["step"], r["writer"], f"{r['step']}/{r['writer']}"):
                i = keys.index(key)
                sums[2 * i] += loss
                sums[2 * i + 1] += n
    dist.all_reduce(sums)
    model.train()
    return {k: (sums[2 * i] / sums[2 * i + 1]).item() for i, k in enumerate(keys) if sums[2 * i + 1] > 0}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=str(HERE / "data" / "examples-glossary.parquet"))
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--conversations-per-step", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=24_576)
    ap.add_argument("--evals", type=int, default=12, help="validations over the run (plus one before training)")
    ap.add_argument("--save-every", type=int, default=0, help="also write a whole model every N validations (0: end only)")
    ap.add_argument("--checkpoint-minutes", type=float, default=20)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--validation-limit", type=int, default=None)
    ap.add_argument("--allow-slow", action="store_true", help="tests only: allow the slow linear-attention path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # started by torchrun, or directly with RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    # (rent/launch.sh does the latter, so a stop signal reaches every GPU process directly)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60),
                            device_id=torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    run = RUNS / args.name
    if (run / "models" / "final").exists():
        if rank == 0:
            print(f"{args.name} has already finished")
        dist.destroy_process_group()
        return
    run.mkdir(parents=True, exist_ok=True)
    logf = (run / "log.jsonl").open("a") if rank == 0 else None

    def log_event(**e):
        if logf:
            logf.write(json.dumps({"at": single.now(), **e}) + "\n")
            logf.flush()

    def say(text):
        if rank == 0:
            print(f"[{single.now()}] {text}", flush=True)
            log_event(kind="message", text=text)

    def status(**fields):
        if rank == 0:
            (run / "status.json.tmp").write_text(json.dumps({"name": args.name, "updated": single.now(), **fields}, indent=1))
            (run / "status.json.tmp").replace(run / "status.json")

    status(state="loading")
    torch.manual_seed(args.seed)
    model, tokenizer, n_params = build(args.model, device, args.allow_slow)
    # tokenized data: GPU 0 builds the cache, the others wait and read it
    if rank == 0:
        train, val, info = single.load_examples(tokenizer, args.max_tokens, say, Path(args.data).resolve())
    dist.barrier()
    if rank != 0:
        train, val, info = single.load_examples(tokenizer, args.max_tokens, lambda s: None, Path(args.data).resolve())
    if args.validation_limit:
        val = random.Random(1).sample(val, min(args.validation_limit, len(val)))

    order: list[int] = []
    for epoch in range(math.ceil(args.epochs)):
        o = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(o)
        order += o
    order = order[:int(len(train) * args.epochs)]
    per_step = args.conversations_per_step
    total_steps = math.ceil(len(order) / per_step)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    eval_every = max(1, total_steps // args.evals)
    warmup = max(1, min(30, total_steps // 20))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0, fused=True)
    state = load_resume(model, optimizer, run) or {"step": 0, "seen_tokens": 0, "seen_answer_tokens": 0,
                                                    "train_seconds": 0.0}
    gpu = torch.cuda.get_device_name(device)
    if rank == 0:
        (run / "config.json").write_text(json.dumps({
            "model": args.model, "name": args.name, "parameters": n_params, "trainable_parameters": n_params,
            "method": "full", "lora_rank": None, "lr": args.lr, "conversations_per_step": per_step,
            "epochs": args.epochs, "total_steps": total_steps, "eval_every": eval_every, "warmup": warmup,
            "schedule": "warmup-stable-decay (last 15% linear to 10%)", "gpu": f"{world} x {gpu}",
            "host": socket.gethostname(), "data": args.data, "init_from": None, **info,
            "started": single.now(), "args": vars(args)}, indent=1))
    say(f"{args.model}: {n_params / 1e9:.2f}B parameters, all trained, on {world} x {gpu}; {len(train):,} "
        f"conversations, {total_steps} steps of {per_step}" + (f"; resuming at step {state['step']}" if state["step"] else ""))

    eval_seconds, n_evals = 0.0, 0
    if state["step"] == 0 and not args.max_steps:
        status(state="validating", step=0, total_steps=total_steps)
        t = time.time()
        v = validate(model, val, device, rank, world)
        eval_seconds = time.time() - t
        log_event(kind="validation", step=0, seen_answer_tokens=0, seconds=round(eval_seconds, 1), **v)
        say(f"step 0 validation loss {v['all']:.4f} ({eval_seconds:.0f}s)")

    model.train()
    last_checkpoint, recent = time.time(), []
    while state["step"] < total_steps:
        step = state["step"]
        lr = single.learning_rate(step, total_steps, args.lr, warmup, 0.15)
        for g in optimizer.param_groups:
            g["lr"] = lr
        batch = [train[i] for i in order[step * per_step:(step + 1) * per_step]]
        n_answer = sum(len(r["ids"]) - r["start"] for r in batch)
        n_tokens = sum(len(r["ids"]) for r in batch)
        hand = deal(batch, world)[rank]
        t = time.time()
        local_loss = torch.zeros(1, device=device, dtype=torch.float64)      # summed answer-token loss
        parts = torch.zeros(4, device=device, dtype=torch.float64)           # opening loss, opening n, section loss, section n
        interrupted = False
        for r, w in hand:
            if stop_requested(device):
                interrupted = True
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(r["ids"].to(device), r["start"])
            # gradients are averaged over GPUs by FSDP: scale so the result is the mean over all answer tokens
            (loss * w * world / n_answer).backward()
            n = (len(r["ids"]) - r["start"]) * w
            local_loss[0] += loss.detach().double() * w
            k = 0 if r["step"] == "opening" else 2
            parts[k] += loss.detach().double() * w
            parts[k + 1] += n
        if interrupted:
            optimizer.zero_grad(set_to_none=True)
            say(f"signal {STOP['signal']}: saving step {state['step']} and stopping")
            status(state="interrupted, saving", step=state["step"], total_steps=total_steps)
            t = time.time()
            save_resume(model, optimizer, state, run, rank)
            say(f"saved in {time.time() - t:.0f}s; run again to resume")
            status(state="interrupted (saved; resumes when restarted)", step=state["step"], total_steps=total_steps)
            dist.destroy_process_group()
            raise SystemExit(INTERRUPTED)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        grad_norm = grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else grad_norm.item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dist.all_reduce(local_loss)
        dist.all_reduce(parts)
        torch.cuda.synchronize()
        seconds = time.time() - t
        state["step"] += 1
        state["seen_tokens"] += n_tokens
        state["seen_answer_tokens"] += n_answer
        state["train_seconds"] += seconds
        recent = (recent + [seconds])[-20:]
        rate = sum(recent) / len(recent)
        mem = torch.tensor([torch.cuda.max_memory_allocated(device) / 2 ** 30], device=device)
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        extra = {}
        if parts[1] > 0:
            extra["loss_opening"] = (parts[0] / parts[1]).item()
        if parts[3] > 0:
            extra["loss_section"] = (parts[2] / parts[3]).item()
        loss_value = local_loss[0].item() / n_answer
        log_event(kind="train", step=state["step"], loss=loss_value, lr=lr, grad_norm=grad_norm,
                  seconds=round(seconds, 2), tokens=n_tokens, answer_tokens=n_answer,
                  tokens_per_second=round(n_tokens / seconds), seen_tokens=state["seen_tokens"],
                  seen_answer_tokens=state["seen_answer_tokens"], max_memory_gb=round(mem.item(), 2), **extra)
        remaining_evals = (total_steps - state["step"]) // eval_every
        status(state="training", step=state["step"], total_steps=total_steps, loss=loss_value, lr=lr,
               seconds_per_step=round(rate, 1), train_hours=round(state["train_seconds"] / 3600, 2),
               eta_hours=round(((total_steps - state["step"]) * rate + remaining_evals * eval_seconds) / 3600, 2),
               max_memory_gb=round(mem.item(), 2))

        done = state["step"] == total_steps
        if state["step"] % eval_every == 0 or done:
            status(state="validating", step=state["step"], total_steps=total_steps)
            t = time.time()
            v = validate(model, val, device, rank, world)
            eval_seconds = time.time() - t
            n_evals += 1
            log_event(kind="validation", step=state["step"], seen_answer_tokens=state["seen_answer_tokens"],
                      seconds=round(eval_seconds, 1), **v)
            say(f"step {state['step']}/{total_steps} validation loss {v['all']:.4f} ({eval_seconds:.0f}s)")
            if args.save_every and n_evals % args.save_every == 0 and not done:
                save_whole(model, tokenizer, run / "models" / f"step-{state['step']:05d}", rank)
        # the time check is decided on GPU 0 and shared, so every GPU joins the same checkpoints
        flag = torch.tensor([int(done or time.time() - last_checkpoint > args.checkpoint_minutes * 60)], device=device)
        dist.broadcast(flag, 0)
        if flag.item():
            t = time.time()
            save_resume(model, optimizer, state, run, rank)
            last_checkpoint = time.time()
            log_event(kind="checkpoint", step=state["step"], seconds=round(last_checkpoint - t, 1))

    save_whole(model, tokenizer, run / "models" / "final", rank)
    status(state="finished", step=state["step"], total_steps=total_steps,
           train_hours=round(state["train_seconds"] / 3600, 2))
    say("finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
