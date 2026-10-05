"""Write README.md (the Hugging Face model card) into both upload folders.

    python3 training/hf/make_cards.py OWNER
"""
import sys
from pathlib import Path

OWNER = sys.argv[1] if len(sys.argv) > 1 else "OWNER"
HERE = Path(__file__).resolve().parent

# Opus judge, 3 held-out papers (15 parts), 0-10: faithful, understandable, pleasant, structure, mean
REF = {"Claude Opus 5.5": (8.3, 8.5, 8.1, 8.9, 8.5), "GPT-6 Astra": (8.5, 8.1, 7.4, 8.7, 8.2),
       "GPT-6 Luna": (7.2, 6.6, 7.7, 8.5, 7.5)}
MODELS = {
    "4b": dict(base="Qwen/Qwen3.5-4B", scores=(6.1, 7.0, 7.2, 8.5, 7.2), untrained=3.8, serious=11.3,
               method="LoRA (rank 32, learning rate 2e-4) on one RTX 3090, 38 hours, merged into the full weights",
               h100=(20, 9, 1064), rtx=(30, 21, 502), price_rtx=3.6, other="9b"),
    "9b": dict(base="Qwen/Qwen3.5-9B", scores=(7.1, 7.7, 7.8, 8.6, 7.8), untrained=5.8, serious=5.7,
               method="all weights (learning rate 1e-5) on one B300, 6 hours",
               h100=(27, 15, 807), rtx=(50, 29, 392), price_rtx=4.6, other="4b"),
}

for size, m in MODELS.items():
    name = f"qwen3.5-{size}-paper-rewriter"
    other = f"{OWNER}/qwen3.5-{m['other']}-paper-rewriter"
    table = [("**This model**", m["scores"])] + list(REF.items()) + [(f"{m['base'].split('/')[1]}, untrained", (None,) * 4 + (m["untrained"],))]
    best = [max(v[i] for _, v in table if v[i] is not None) for i in range(5)]
    cell = lambda v, i: "" if v is None else (f"**{v}**" if v == best[i] else f"{v}")
    rows = "\n".join(f"| {k} | " + " | ".join(cell(v[i], i) for i in range(5)) + " |" for k, v in table)
    card = f"""---
license: apache-2.0
base_model: {m['base']}
language: [en]
pipeline_tag: text-generation
library_name: transformers
tags: [science-communication, text-simplification, rewriting, qwen3.5]
---

# Qwen3.5-{size.upper()} Paper Rewriter

Rewrites a research paper so a curious 12–14-year-old can read it: the same paper, in the
authors' own voice, keeping every result, number and uncertainty, in plain words.
It rewrites, it does not summarise. Fine-tuned from [{m['base']}](https://huggingface.co/{m['base']}).
See also the {m['other'].upper()} version: [{other}](https://huggingface.co/{other}).

## Quality

An Opus judge scored each rewrite from 0 to 10 on four aspects, on the same 3 held-out
papers (15 parts) for every model. The large models used our refined prompts.

| Model | Faithful | Clear | Pleasant | Structure | Average |
|---|---|---|---|---|---|
{rows}

Bold: best in each column. Faithful = faithful and exact; Clear = understandable;
Structure = structure and voice.

Part by part, against each model (better, same, or worse on each of the 15 parts):

![Win rate of this model against Claude Opus 5.5, GPT-6 Astra, GPT-6 Luna and the untrained base model, on the average score and on faithfulness](winrate.png)

Serious problems found by the judge: about {m['serious']} per paper (Opus 0.7, Astra 2.0, Luna 9.0).

**Read these numbers with care:** three papers is a small test, so gaps under about half a
point may be luck; and the judge is itself a frontier model, so it may favour the style
of the synthetic rewrites this model learned from.

## Speed and cost

One paper (~4,500 words) sent alone, vLLM 0.29, bf16:

| GPU | Seconds per paper | With the draft head (MTP) | Papers per hour, GPU kept full |
|---|---|---|---|
| H100 80GB | {m['h100'][0]} | {m['h100'][1]} | {m['h100'][2]:,} |
| RTX PRO 6000 | {m['rtx'][0]} | {m['rtx'][1]} | {m['rtx'][2]:,} |

About ${m['price_rtx']} per 1,000 typical papers (~7,100 words) on an RTX PRO 6000 rented at
$1.80/h and kept busy.

## How to use

The model was trained on two calls per paper, and expects exactly that layout.
`rewrite.py` in this repository does both calls with the standard library only.

```bash
vllm serve {OWNER}/{name} --served-model-name rewriter --max-model-len 65536 \\
  --default-chat-template-kwargs '{{"enable_thinking": false}}' \\
  --speculative-config '{{"method": "mtp", "num_speculative_tokens": 2}}'   # optional, ~2x faster for one reader

python rewrite.py paper.json > rewrite.md
```

`paper.json` holds the paper's parts as plain text: `title`, `abstract`,
`introduction_first` (first paragraph of the introduction), `introduction_rest`,
`methods`, `results`, `discussion`, `conclusion`, and an optional `glossary`.

1. **The opening:** title, abstract, first introduction paragraph and conclusion, from the whole paper.
2. **The other sections,** each on its own (they can run at the same time), given the rewritten opening.

Use greedy decoding (temperature 0) and turn thinking off. The `writer` field picks
a writing style; `rewrite.py` uses `writer_a`, the main style of the training data
(for the 9B, the judge scored it the same as the label used in our benchmark). The exact system and user
messages are in `rewrite.py`. `rewrite.py` reproduces the training prompts character for
character; we checked this on 15 prompts.

### The reference glossary

Each request carries a `reference_glossary`: definitions of the passage's technical
terms, one per entry, as `- term (long form): definition [source]`. We built ours from an
offline copy of Wikipedia. Almost all training examples had one (62 of 21,069 had it
empty), so the model works best with a glossary. Without one it still runs, but that
setting was rarely seen in training. Give each request only the entries whose term
appears in its text (`rewrite.py` does this).

### Draft head

`mtp.safetensors` is the multi-token-prediction head of the original {m['base']}.
Our fine-tuning did not keep this head, so we added the base model's back. vLLM uses it for speculative decoding with the
`--speculative-config` line above. With greedy decoding, every drafted word is checked
by the model; the judge scored our 9B the same with and without it. Transformers ignores
these weights.

## Training

- **Data:** [maximerivest/paper-rewrites-for-young-readers](https://huggingface.co/datasets/maximerivest/paper-rewrites-for-young-readers):
  20,394 training conversations (190M tokens) from 4,416 open-access papers in
  PubMed Central, all under CC BY. The target rewrites are synthetic, written by frontier
  models with our refined prompts, each with a reference glossary.
- **Method:** one pass over the data, {m['method']}.
- **Held out:** the benchmark papers were never trained on.

## Limitations

- **Accuracy is its weakest point.** Typical errors: a number attached to the wrong group,
  a hedge turned into a firm claim, two similar technical terms confused (in one plant
  paper, "hermaphroditic" and "monoecious"). Check important facts against the paper.
- English only. Tested on life-science and environmental papers.
- Not a source of medical or other professional advice.
- Very long sections can run into the length limit; `rewrite.py` caps each reply at about
  twice the section's length.

## License

Apache 2.0, like the base model (see `LICENSE`). Rewrites of CC BY papers should credit
the original authors and say the text was adapted.
"""
    (HERE / name / "README.md").write_text(card)
    print("wrote", HERE / name / "README.md")
