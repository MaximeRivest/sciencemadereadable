"""Few-shot probe of Ettin-decoder-1B (a base model, not instruction-tuned).

Each paragraph is rewritten alone: the prompt shows two example pairs (an original
paragraph and its Opus rewrite, from a training paper) and then the paragraph to
rewrite. Headings and table rows are copied as they are. The paragraphs are put back
in order and saved in the same format as the other candidates.

Host side first (writes ettin_input.json, and afterwards turns the JSON results into parquet):
    .venv/bin/python model_baselines/ettin_io.py prepare
Then inside the vLLM container (it has torch and transformers), on GPU 1:
    podman run --rm --device nvidia.com/gpu=1 -v ~/.cache/huggingface:/root/.cache/huggingface \\
      -v $PWD:/work -w /work --entrypoint python3 docker.io/vllm/vllm-openai:v0.25.1 \\
      model_baselines/ettin_probe.py
    .venv/bin/python model_baselines/ettin_io.py collect
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/work")
BASE = ROOT / "model_baselines"
OUT = BASE / "rewrites/ettin1b/rewrites"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "jhu-clsp/ettin-decoder-1b"
SECTIONS = ["title", "abstract", "introduction_first", "introduction_rest", "methods", "results", "discussion", "conclusion"]


def paragraphs(text):
    return [p for p in text.split("\n\n") if p.strip()]


def is_heading_or_table(p):
    s = p.strip()
    return s.startswith("|") or (len(s.split()) <= 12 and not s.endswith((".", ":", ";")))


def example_pairs(rows):
    """Two aligned (original, rewrite) paragraph pairs from an Opus-rewritten training paper."""
    for r in rows:
        if r["section"] != "introduction_rest":
            continue
        o, n = paragraphs(r["original"]), paragraphs(r["rewrite"])
        if len(o) == len(n) and len(o) >= 2:
            pairs = [(a, b) for a, b in zip(o, n) if 60 <= len(a.split()) <= 150 and not is_heading_or_table(a)]
            if len(pairs) >= 2:
                return pairs[:2]
    raise RuntimeError("no aligned example pairs found")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()

    data = json.loads((BASE / "ettin_input.json").read_text())   # written by ettin_prepare (host side)
    shots = example_pairs(data["training_sections"])
    header = ("Below, paragraphs from scientific papers are rewritten, as if by their own authors, so that "
              "a curious 13-year-old can understand them. Every fact and number is kept.\n\n")
    header += "".join(f"Original: {a}\nRewritten: {b}\n\n" for a, b in shots)

    papers = data["test_papers"]
    for paper in papers:
        target = OUT / f"{paper['paper_id']}.json"
        if target.exists():
            continue
        jobs = []   # (section, index, text or None for copy)
        pieces = {s: [] for s in SECTIONS}
        for s in SECTIONS:
            for i, p in enumerate(paragraphs(paper.get(s) or "")):
                pieces[s].append(p)
                if not is_heading_or_table(p) or s == "title":
                    jobs.append((s, i, p))
        for start in range(0, len(jobs), 12):
            batch = jobs[start:start + 12]
            prompts = [header + f"Original: {p}\nRewritten:" for _, _, p in batch]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=7000).to("cuda")
            longest = max(len(tok(p).input_ids) for _, _, p in batch)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=min(900, int(longest * 1.8) + 40), do_sample=False,
                                     repetition_penalty=1.1, pad_token_id=tok.pad_token_id)
            for (s, i, _), seq in zip(batch, out[:, enc.input_ids.shape[1]:]):
                text = tok.decode(seq, skip_special_tokens=True)
                text = text.split("\nOriginal:")[0].split("\n\n")[0].strip()
                pieces[s][i] = text
        rows = [{"paper_id": paper["paper_id"], "section": s, "original": paper[s] or "",
                 "rewrite": "\n\n".join(pieces[s]), "editorial_notes": [], "input_tokens": 0, "output_tokens": 0,
                 "model": MODEL, "recipe": "few-shot-paragraphs", "paper_minutes": 0.0, "finished_at": ""}
                for s in SECTIONS if paper.get(s)]
        target.write_text(json.dumps(rows))
        print(paper["paper_id"], "done", flush=True)


if __name__ == "__main__":
    main()
