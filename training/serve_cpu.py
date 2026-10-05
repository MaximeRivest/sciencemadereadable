"""Serve a trained checkpoint on the CPU, with an OpenAI-compatible chat API, so the
benchmark can call it through functai while both GPUs are busy training.

    .venv/bin/python serve_cpu.py --base Qwen/Qwen3.5-0.8B --weights runs/qwen35-0.8b/adapters/step-00152
    .venv/bin/python serve_cpu.py --base Qwen/Qwen3.5-0.8B --weights runs/qwen35-0.8b-full/adapters/step-00076

--weights is a LoRA adapter folder (merged into the base at load) or a whole saved
model (full training). Greedy decoding, thinking off, like the student was trained.
Requests that arrive together are generated as one batch (left padding), which is
what makes the CPU usable: the weights are read once per token for the whole batch.
"""
from __future__ import annotations

import sys

sys.modules["fla"] = None        # the GPU kernels cannot run on the CPU: use transformers' torch path

import argparse
import json
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--weights", default=None)
ap.add_argument("--port", type=int, default=8011)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--threads", type=int, default=24)
ap.add_argument("--wait", type=float, default=2.0, help="seconds to gather a batch")
args = ap.parse_args()
torch.set_num_threads(args.threads)

tokenizer = AutoTokenizer.from_pretrained(args.base)
tokenizer.padding_side = "left"
weights = Path(args.weights) if args.weights else None
if weights and (weights / "adapter_config.json").exists():
    import peft
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    model = peft.PeftModel.from_pretrained(model, str(weights)).merge_and_unload()
else:
    model = AutoModelForCausalLM.from_pretrained(str(weights) if weights else args.base, dtype=torch.float32)
model.eval()
EOS = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
jobs: "queue.Queue[tuple[list[int], int, Future]]" = queue.Queue()
print(f"loaded {args.base} + {weights}", flush=True)


def batcher():
    while True:
        batch = [jobs.get()]
        deadline = time.time() + args.wait
        while len(batch) < args.batch and time.time() < deadline:
            try:
                batch.append(jobs.get(timeout=max(0.01, deadline - time.time())))
            except queue.Empty:
                break
        try:
            width = max(len(ids) for ids, _, _ in batch)
            input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for i, (ids, _, _) in enumerate(batch):
                input_ids[i, width - len(ids):] = torch.tensor(ids)
                mask[i, width - len(ids):] = 1
            t = time.time()
            with torch.inference_mode():
                out = model.generate(input_ids=input_ids, attention_mask=mask, do_sample=False,
                                     max_new_tokens=max(n for _, n, _ in batch), eos_token_id=EOS,
                                     pad_token_id=tokenizer.pad_token_id)
            new_total = 0
            for i, (ids, limit, fut) in enumerate(batch):
                new = out[i, width:width + limit].tolist()
                for k, tok in enumerate(new):
                    if tok in EOS:
                        new, finish = new[:k], "stop"
                        break
                else:
                    finish = "length"
                new_total += len(new)
                fut.set_result((tokenizer.decode(new, skip_special_tokens=True), len(ids), len(new), finish))
            print(f"batch of {len(batch)}: {width:,} prompt tokens wide, {new_total:,} new tokens, "
                  f"{time.time() - t:.0f}s", flush=True)
        except Exception as error:      # noqa: BLE001 - every waiting request gets the error
            for _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(error)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"object": "list", "data": [{"id": "local", "object": "model"}]})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else
                     "".join(p.get("text", "") for p in m["content"])} for m in request["messages"]]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                            return_dict=False, enable_thinking=False)
        limit = int(request.get("max_completion_tokens") or request.get("max_tokens") or 4096)
        fut: Future = Future()
        jobs.put((list(ids), limit, fut))
        try:
            text, n_in, n_out, finish = fut.result()
        except Exception as error:      # noqa: BLE001
            self._send(500, {"error": {"message": f"{type(error).__name__}: {error}"}})
            return
        self._send(200, {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
                         "created": int(time.time()), "model": "local",
                         "choices": [{"index": 0, "finish_reason": finish,
                                      "message": {"role": "assistant", "content": text}}],
                         "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                                   "total_tokens": n_in + n_out}})

    def log_message(self, *a):
        pass


threading.Thread(target=batcher, daemon=True).start()
print(f"serving on http://127.0.0.1:{args.port}/v1", flush=True)
ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
