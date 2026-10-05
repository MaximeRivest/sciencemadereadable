"""Head-to-head win rate for the model cards: on each of the 15 judged parts, does our model
score higher, the same, or lower than each other model? Two panels: the average of the four
aspects, and faithfulness alone. Writes winrate.png into both upload folders.

    uv run --no-project --with matplotlib training/hf/make_winrate.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D = json.loads((HERE.parent / "figures/blog/data.json").read_text())
A = D["aspects"]
WIN, TIE, LOSS, INK, MUTED = "#2a7a55", "#c9d1d9", "#c0504d", "#1f2328", "#6a737d"
plt.rcParams.update({"font.size": 12, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "text.parse_math": False})


def per(label, aspect=None):
    return {(r["paper"], r["unit"]): (r["scores"][aspect] if aspect else sum(r["scores"][a] for a in A) / 4)
            for r in D["units"][label] if r["scores"]}


for size, me, base in [("4b", "q4_trained", "q4_untrained"), ("9b", "q9_trained", "q9_untrained")]:
    opponents = [("Claude Opus 5.5", "opus_v8"), ("GPT-6 Astra", "astra_v8"), ("GPT-6 Luna", "luna_v8"),
                 (f"Qwen3.5-{size.upper()}, untrained", base)]
    keys = sorted(per("opus_v8"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (title, aspect) in zip(axes, [("Average of the four aspects", None), ("Faithful and exact", "faithful_and_exact")]):
        mine = per(me, aspect)
        for i, (name, lab) in enumerate(opponents):
            other = per(lab, aspect)
            w = t = l = 0
            for k in keys:   # a missing part is an empty rewrite: it loses
                x, y = mine.get(k, -1), other.get(k, -1)
                w, t, l = w + (x > y), t + (x == y), l + (x < y)
            y = len(opponents) - 1 - i
            left = 0
            for n, c, tc in [(w, WIN, "white"), (t, TIE, INK), (l, LOSS, "white")]:
                ax.barh(y, n, left=left, color=c, height=0.62)
                if n:
                    ax.text(left + n / 2, y, str(n), ha="center", va="center", color=tc, fontweight="bold")
                left += n
            if ax is axes[0]:
                ax.text(-0.4, y, f"vs {name}", ha="right", va="center", fontsize=12.5, fontweight="bold")
        ax.set_xlim(0, len(keys)), ax.set_xticks([0, 5, 10, 15]), ax.set_yticks([])
        ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
        ax.set_xlabel(f"Parts of the 3 test papers (of {len(keys)})")
    fig.text(0.02, 0.96, f"Qwen3.5-{size.upper()} Paper Rewriter, part by part: ", fontsize=15, fontweight="bold", va="top")
    fig.text(0.44 if size == "9b" else 0.44, 0.96, "■ better", color=WIN, fontsize=14, fontweight="bold", va="top")
    fig.text(0.53, 0.96, "■ same", color="#8a959e", fontsize=14, fontweight="bold", va="top")
    fig.text(0.61, 0.96, "■ worse", color=LOSS, fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 0.02, "Opus judge, 0–10, same 15 parts for every model. Large models with refined prompts. "
             "Three papers is a small test.", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0.17, 0.05, 1, 0.9))
    out = HERE / f"qwen3.5-{size}-paper-rewriter" / "winrate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)
