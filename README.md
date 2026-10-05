# Science made readable

**https://sciencemadereadable.com**: search open research papers and read them rewritten in plain
words, next to the original, with the paper's own figures and tables.

The rewrite is a translation, not a summary: the same paper, in the authors' voice, with the same
information, uncertainty, headings and order, written so that a curious 14-year-old can follow it.
It is done by a small model we trained (Qwen3.5 9B, fine-tuned on about 4,400 papers rewritten by
large models), which runs on a single GPU.

- **Models:** [qwen3.5-9b-paper-rewriter](https://huggingface.co/maximerivest/qwen3.5-9b-paper-rewriter) (the one the site uses),
  [qwen3.5-4b-paper-rewriter](https://huggingface.co/maximerivest/qwen3.5-4b-paper-rewriter)
- **Training data:** [paper-rewrites-for-young-readers](https://huggingface.co/datasets/maximerivest/paper-rewrites-for-young-readers)
- **Write-up:** `training/figures/blog/x_article.md`, with its figures in `training/figures/blog/out/`

## How it fits together

```
paper_corpus/  open-licence papers (Europe PMC): harvest, split into sections, large-model rewrites
glossary/      reference entries for each paper (offline Wikipedia and Wiktionary), given to the model
rewrite_benchmark/  the rewrite recipe (prompts) and the judge that scores rewrites
model_baselines/    other models' rewrites and scores on the same test papers
training/      training the small models on the corpus; speed and quality measurements
  v3/          the next training set (in progress): every fact from the paper, the glossary,
               or what a 14-year-old knows; ⟦markers⟧ for anything else
app/           the website, the job queue, the GPU worker (app/README.md)
notebooks/     tutorials: how the rewrite recipe was built and tested
explainers/, article-tokens/, pine-nut-visuals/   side experiments
```

The prompts are the same everywhere: the app's TypeScript programs send byte for byte what the
Python training and benchmark code sent (`app/tools/check_prompts.ts`), so the model in production
reads exactly what it was trained and scored on.

## Running it

Python parts (from this folder):

```
uv sync
.venv/bin/python app/server.py              # queue, paper service and the page: http://127.0.0.1:8795
```

The page and the worker (Node 22+). They use [functai](https://github.com/MaximeRivest/functai)'s
TypeScript library, which is not on npm yet: clone it next to this folder first.

```
git clone https://github.com/MaximeRivest/functai ../functai
(cd ../functai/ts && npm install)
cd app && npm install && node tools/build.mjs
npm run worker                              # takes jobs, runs the model through vLLM (app/worker/)
```

Serving the model: any OpenAI-compatible server, for example
`vllm serve maximerivest/qwen3.5-9b-paper-rewriter --dtype bfloat16 --max-model-len 65536
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
--override-generation-config '{"temperature":0.0}' --default-chat-template-kwargs '{"enable_thinking":false}'`.
`app/rent/h100_up.sh` rents one H100 on Nebius and does all of this.

## What is not in this repository

Data and models live next to the code on the working machine and are left out (`.gitignore`):
paper copies, the large-model rewrites, the offline Wikipedia copy (46 GB,
`glossary/data/offline/download.sh`), training data and checkpoints (about 200 GB), and run outputs.
The trained models are on Hugging Face (above).

## Next version

`training/v3/` builds the next training set: every fact in an answer must come from the paper, the
glossary given with it, or what a curious 14-year-old knows; anything else becomes a ⟦marker⟧ to be
explained afterwards, so the small model writes instead of recalling. Design: `training/dataset_v3.md`.
A 50-paper pilot (`training/v3/pilot50.py`) finished on 2026-10-05: in a blind comparison the judge
still preferred the current answers (159 to 40, 41 ties), so the recipe needs more work before training.

## Status

A research project that went live on 2026-10-05. The rewrites read well and keep the papers'
structure, but they can still contain factual errors: always check the original, shown alongside.

## Licence

Code and notes: [Apache 2.0](LICENSE). The papers shown and rewritten are open-access articles under
CC BY; their rewrites credit the original authors and link to the paper.
