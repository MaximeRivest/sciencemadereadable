# functai `bake(method="sft")`: what a long-text distillation needs

Context (2026-10-01): Scholar's Reading List wants to train a small model to
replace Opus in two AI functions, `rewrite_opening` and `rewrite_section`
(`rewrite_benchmark/translator.py`). The training data is a corpus of
Opus and GPT-6 Astra rewrites:

| | examples | input tokens (median / p90 / p99) | output tokens (median / p90 / p99) |
|---|---|---|---|
| opening (whole paper in) | ~3,200 | 10.6k / 15.5k / 22.2k | 1.2k / 1.8k / 2.6k |
| section (section + rewritten opening in) | ~11,900 | 3.5k / 6.2k / 10.6k | 2.6k / 5.2k / 9.0k |

About 125M tokens per pass. Students: Qwen3.5 0.8B / 2B / 4B / 9B. Hardware:
lambda, 2× RTX 3090 (24 GB each, no NVLink).

`bake(method="sft")` already does the important things right: the loss is on the
answer only, the training conversations come from the same layout the
function is called with, LoRA is chosen when needed, and the result is used
with `fn.using(lm=baked)`. The gaps below are what stops it from running this
job. They are listed most blocking first.

## Blocking

1. **The base model is loaded in fp32.** A 4B model is 16 GB before any
   activations; 9B does not fit at all. Needed: load in bf16 when training
   LoRA, and an option for 4-bit base weights (QLoRA) for 9B on 24 GB.
2. **Batches are padded to a fixed number of rows.** With examples of
   2k–25k tokens, `batch_size=8` runs out of memory and padding wastes most
   of the compute. Needed: batches built by token budget (for example
   `max_tokens_per_batch=16_000`), ideally with packing and variable-length
   attention (no padding at all).
3. **Nothing is saved or reported during a pass.** One pass here takes
   15–30 hours. Needed: training loss every N steps, validation loss every
   N steps, a checkpoint every N steps, resuming from the last checkpoint
   after a crash, and the loss curve written to a file (CSV/JSONL) that a
   dashboard can read while training is running.
4. **The test report uses exact-match accuracy.** For long text, every
   answer counts as wrong. Needed: a `metric=` argument (any function, or a
   functai evaluation / judge) for the report, and the option to skip the
   report and evaluate separately.
5. **Lengths are capped at 4,096 tokens.** `max_new_tokens` is capped at 4,096
   (our sections need up to ~9k) and `serve()` passes
   `--max-model-len 4096` (our prompts reach 22k). Needed: take both from the
   data, or let the caller set them.

## Important

6. **One function per model.** Our two functions are the same skill, and
   the opening function alone has only ~3,200 examples. Needed: train one
   student on several functions (a list of `(fn, rows)`, or bake a module),
   each still called through its own layout.
7. **One GPU only.** Needed: data-parallel training across visible GPUs
   (`torchrun`/DDP). With LoRA the gradients are small (tens of MB), so it
   works well even without NVLink.
8. **Constant inputs are repeated in every example.** Our functions take a
   long `writing_brief`, `reader_habits` and `section_guidance` that never
   change. Needed: a way to mark inputs as fixed so the student's prompt
   leaves them out ("baked in"). The workaround is a separate student
   function with fewer inputs, which loses the "same function, different
   executor" guarantee.
9. **Qwen3.5 needs fast kernels for its linear-attention layers**
   (`flash-linear-attention`, `causal-conv1d`). Without them, transformers
   falls back to a slow path without saying so. Needed: check for them at
   start and warn (or refuse) when a hybrid model would train on the slow
   path. On NixOS, Triton also needs `TRITON_LIBCUDA_PATH` (as `serve()`
   already sets).

## Nice to have

10. A learning-rate schedule that allows branching (warmup, constant, short
    decay at the end). Then checkpoints taken partway through give a fair
    "quality versus amount of data" curve, which a cosine schedule does not.
11. A `weights=` column (or a way to tag rows), so examples from different
    teachers can be weighted or labelled.
12. Export the training conversations (`bake.examples(fn, rows) -> JSONL`)
    so that an external trainer (TRL, Axolotl, Unsloth) can be used with
    functai's exact layout, and `bake.load()` can wrap the resulting HF
    folder. This bridge would have unblocked us today.
