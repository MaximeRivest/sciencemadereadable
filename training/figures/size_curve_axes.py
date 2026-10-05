"""Square PNG, one panel per judge aspect: Qwen 0.8B / 4B / 9B untrained vs trained
(glossary data, glossary given at rewrite time), against Luna, Astra and Opus with
their best prompts (v8). Same 3 benchmark papers and Opus judge throughout.

    uv run --with matplotlib training/figures/size_curve_axes.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

T = Path(__file__).resolve().parents[1]
load = lambda p: json.loads((T / p).read_text())["candidates"]

trained = {"0.8B": load("runs/qwen35-0.8b-glossary-scratch/benchmark/final-glossary.json")["student-0.8b-glossary-final-greedy"],
           "4B": load("runs/qwen35-4b-glossary-scratch/benchmark/final-glossary.json")["student-4b-glossary-final-greedy"],
           "9B": load("runs/qwen35-9b-full-glossary/benchmark/final-glossary.json")["student-9b-glossary-final-greedy"]}
untrained = {"0.8B": None,   # all 15 answers empty (logs/judge-qwen08b.json)
             "4B": load("runs/qwen35-4b-glossary-scratch/benchmark/final-glossary.json")["qwen4b"],
             "9B": load("logs/judge-qwen9b.json")["qwen9b"]}   # 14 of 15 parts (one empty)
refs = {"Opus": (load("runs/qwen35-4b-glossary-scratch/benchmark/final-glossary.json")["opus"], "#b5651d"),
        "Astra": (load("logs/judge-v8-astra.json")["v8-astra"], "#7b5ea7"),
        "Luna": (load("logs/judge-v8-luna.json")["v8-luna"], "#3b7dd8")}

ASPECTS = [("faithful_and_exact", "Faithful and exact"), ("understandable", "Understandable"),
           ("pleasant_to_read", "Pleasant to read"), ("structure_and_voice", "Structure and voice")]
sizes = ["0.8B", "4B", "9B"]

fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
for ax, (key, title) in zip(axes.flat, ASPECTS):
    w = 0.36
    for i, s in enumerate(sizes):
        for dx, src, color, label in [(-w / 2, untrained, "#a9b4ad", "untrained"), (w / 2, trained, "#2a7a55", "trained")]:
            v = src[s][key] if src[s] else None
            x = i + dx
            if v is None:
                ax.bar(x, 0.08, w * 0.95, color=color)
                ax.text(x, 0.3, "no usable\nanswer", ha="center", va="bottom", fontsize=7, zorder=5)
            else:
                ax.bar(x, v, w * 0.95, color=color, zorder=3)
                ax.text(x, v - 0.15, f"{v:.1f}", ha="center", va="top", fontsize=10, fontweight="bold", zorder=4,
                        color="white" if src is trained else "black")
    placed = []
    for name, (r, color) in sorted(refs.items(), key=lambda kv: -kv[1][0][key]):
        v = r[key]
        ax.plot([-0.6, 2.5], [v, v], color=color, ls="--", lw=1.6, zorder=2)
        ty = v
        while any(abs(ty - p) < 0.32 for p in placed):
            ty -= 0.32
        placed.append(ty)
        ax.text(2.62, ty, f"{name} {v:.1f}", color=color, va="center", fontsize=9, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(3), [f"Qwen {s}" for s in sizes], fontsize=11)
    ax.set_xlim(-0.6, 3.15)
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].set_ylabel("Opus judge score (0–10)")
axes[1, 0].set_ylabel("Opus judge score (0–10)")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [Patch(color="#a9b4ad", label="untrained"), Patch(color="#2a7a55", label="trained")] + \
          [Line2D([], [], color=c, ls="--", lw=1.6, label=f"{n} (best prompts)") for n, (_, c) in refs.items()]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10)
fig.suptitle("Student models before and after training, against Luna, Astra and Opus", fontsize=15, fontweight="bold")
fig.text(0.5, 0.045, "Same 3 test papers (15 parts) and Opus judge for every model. Trained = one pass over the glossary data; "
         "glossary given when rewriting.\n4B trained with LoRA, 0.8B and 9B with all weights. Untrained 9B: 14 of 15 parts "
         "(one came back empty). With 3 papers, gaps under ~0.5 may be noise.", ha="center", fontsize=8.5, alpha=0.8)
fig.tight_layout(rect=(0, 0.08, 1, 0.96))
out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=150)
print(out)
