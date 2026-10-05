"""Blog / course figures from data.json (made by export_data.py). PNG (1920x1200) + SVG.

    uv run --no-project --with matplotlib training/figures/blog/make_figures.py
"""
import json
import math
import random
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
D = json.loads((HERE / "data.json").read_text())
A = D["aspects"]
ASPECT_NAMES = {"faithful_and_exact": "Faithful and exact", "understandable": "Understandable",
                "pleasant_to_read": "Pleasant to read", "structure_and_voice": "Structure and voice"}

GREEN, GREEN_L, GREY = "#2a7a55", "#7fc4a0", "#a9b4ad"
OPUS, ASTRA, LUNA = "#b5651d", "#7b5ea7", "#3b7dd8"
INK, MUTED = "#1f2328", "#6a737d"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#888",
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
                     "text.parse_math": False})
NOTE = "Opus judge, 0–10, same 3 held-out papers (15 parts) for every model."


def f1(v):
    """One decimal, halves rounded up (8.45 -> 8.5), as in the written results."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def scores(label, aspect=None):
    """Per-part scores: the mean of the 4 aspects, or one aspect."""
    out = []
    for r in D["units"][label]:
        if r["scores"]:
            out.append(r["scores"][aspect] if aspect else mean(r["scores"][a] for a in A))
    return out


def avg(label, aspect=None):
    s = scores(label, aspect)
    return mean(s) if s else None


def ci(label, aspect=None, n=4000):
    """Rough 95% range of the mean, resampling the parts (optimistic: parts of one paper are related)."""
    s = scores(label, aspect)
    if not s:
        return None
    rng = random.Random(0)
    ms = sorted(mean(rng.choice(s) for _ in s) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def serious_per_paper(label):
    return sum(sev in ("major", "critical") for r in D["units"][label] for sev in r["issues"]) / 3


def figure(title, subtitle, size=(12, 7.5)):
    fig = plt.figure(figsize=size)
    fig.text(0.04, 0.955, title, fontsize=20, fontweight="bold", va="top")
    fig.text(0.04, 0.905, subtitle, fontsize=13, color=MUTED, va="top")
    return fig


def save(fig, name, note=NOTE):
    fig.text(0.04, 0.018, note, fontsize=10.5, color=MUTED, va="bottom")
    fig.savefig(OUT / f"{name}.png", dpi=160)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print(name)


def refline(ax, y, color, label, x=1.0, dy=0.0, xmin=0, xmax=1):
    ax.axhline(y, xmin=xmin, xmax=xmax, color=color, ls="--", lw=1.8, zorder=1)
    ax.text(x, y + dy, f" {label} {f1(y)}", color=color, fontweight="bold", va="center",
            transform=ax.get_yaxis_transform(), fontsize=12)


def bar_with_ci(ax, x, label, color, width, text_color="white"):
    v, r = avg(label), ci(label)
    ax.bar(x, v, width, color=color, zorder=3)
    ax.errorbar([x], [v], yerr=[[v - r[0]], [r[1] - v]], fmt="none", ecolor=INK, elinewidth=1.4, capsize=5,
                alpha=0.7, zorder=4)
    ax.text(x, 0.35, f"{f1(v)}", ha="center", va="bottom", color=text_color, fontweight="bold", fontsize=15, zorder=5)
    return v


# ------------------------------------------------------------------ 1. pipeline
def pipeline():
    fig = figure("How we built small science translators",
                 "Big models write the examples, small open models learn from them, and a judge scores everyone the same way")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    ax.set_xlim(0, 100), ax.set_ylim(0, 60), ax.axis("off")
    boxes = [
        (1, 30, 19, 22, "#eef3f8", "5,000 open\npapers", "PubMed Central,\nCC BY licence"),
        (24.5, 30, 19, 22, "#f6efe7", "Frontier models\nrewrite them", f"{D['data']['papers']:,} papers →\n{D['data']['examples']:,} examples"),
        (48, 30, 19, 22, "#eef6f1", "Small Qwen 3.5\nmodels learn", "0.8B, 4B and 9B\none pass over the data"),
        (72, 30, 25, 22, "#f3eef8", "Opus judges\nevery rewrite", "3 held-out papers, 15 parts,\n4 scores out of 10"),
    ]
    for x, y, w, h, c, head, sub in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2", fc=c, ec="#c9d1d9", lw=1.2))
        ax.text(x + w / 2, y + h - 5, head, ha="center", va="top", fontsize=14.5, fontweight="bold")
        ax.text(x + w / 2, y + 6, sub, ha="center", va="bottom", fontsize=11.5, color=MUTED)
    for x0, x1 in [(20.8, 23.6), (44.3, 47.2), (67.8, 71.2)]:
        ax.annotate("", xy=(x1, 41), xytext=(x0, 41), arrowprops=dict(arrowstyle="-|>", lw=2, color=INK))
    ax.add_patch(FancyBboxPatch((22, 5), 47, 15, boxstyle="round,pad=0.6,rounding_size=2", fc="#fffbe6", ec="#e3d48a"))
    ax.text(45.5, 16, "A reference glossary for every passage", ha="center", fontsize=13.5, fontweight="bold")
    ax.text(45.5, 7.5, "definitions of the paper's terms from an offline Wikipedia copy,\n"
            "given to the writers and to the small models",
            ha="center", fontsize=11.5, color=MUTED)
    ax.annotate("", xy=(33, 28.5), xytext=(33, 21), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#b8a540"))
    ax.annotate("", xy=(57, 28.5), xytext=(57, 21), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#b8a540"))
    save(fig, "01_pipeline", note="Rewrites keep the authors' voice and every result, in words a curious 14-year-old can follow.")


# ------------------------------------------------------------------ 2. prompts
def prompts():
    fig = figure("Better instructions lifted every big model by 1.5 to 2.8 points",
                 "Same models, same papers: our first prompts (v1) against the refined ones (v8)")
    ax = fig.add_axes([0.25, 0.13, 0.62, 0.68])
    rows = [("Opus", "opus_v1", "opus_v8", OPUS), ("Astra", "astra_v1", "astra_v8", ASTRA), ("Luna", "luna_v1", "luna_v8", LUNA)]
    for i, (name, a, b, c) in enumerate(rows):
        y = len(rows) - 1 - i
        va, vb = avg(a), avg(b)
        ax.plot([va, vb], [y, y], color=c, lw=4, alpha=0.35, zorder=1)
        ax.scatter([va], [y], s=260, color="white", edgecolor=c, lw=3, zorder=3)
        ax.scatter([vb], [y], s=260, color=c, zorder=3)
        ax.text(va - 0.15, y, f"{f1(va)}", ha="right", va="center", color=c, fontsize=13)
        ax.text(vb + 0.15, y, f"{f1(vb)}", ha="left", va="center", color=c, fontsize=14, fontweight="bold")
        ax.text(3.35, y, name, ha="right", va="center", fontsize=16, fontweight="bold", color=c)
        ax.text((va + vb) / 2, y + 0.22, f"+{f1(vb - va)}", ha="center", color=c, fontsize=12)
    ax.set_xlim(3.5, 9.6), ax.set_ylim(-0.6, 2.6), ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Average judge score (0–10)")
    ax.scatter([], [], s=120, color="white", edgecolor=MUTED, lw=2, label="first prompts (v1)")
    ax.scatter([], [], s=120, color=MUTED, label="refined prompts (v8)")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "02_prompts_matter", note=NOTE + " Part of the gap is a change of goal: v1 also corrected the paper and allowed lists.")


# ------------------------------------------------------------------ 3. training lifts every size
def training_lift():
    fig = figure("Training turns small open models into usable science writers",
                 "Average judge score before and after one pass over the examples, against the big models")
    ax = fig.add_axes([0.08, 0.12, 0.74, 0.7])
    w = 0.36
    for i, (name, u, t) in enumerate([("Qwen 0.8B", "q08_untrained", "q08_trained"), ("Qwen 4B", "q4_untrained", "q4_trained"),
                                      ("Qwen 9B", "q9_untrained", "q9_trained")]):
        if avg(u) is None:
            ax.bar(i - w / 2, 0.08, w * 0.95, color=GREY)
            ax.text(i - w / 2, 0.3, "no usable\nanswer", ha="center", va="bottom", fontsize=10.5, zorder=5)
        else:
            bar_with_ci(ax, i - w / 2, u, GREY, w * 0.95, text_color=INK)
        bar_with_ci(ax, i + w / 2, t, GREEN, w * 0.95)
        ax.text(i, -0.75, name, ha="center", fontsize=15, fontweight="bold")
    for name, lab, c, dy in [("Opus", "opus_v8", OPUS, 0.12), ("Astra", "astra_v8", ASTRA, -0.12), ("Luna", "luna_v8", LUNA, 0)]:
        refline(ax, avg(lab), c, name, dy=dy, xmax=0.92)
    ax.set_xticks([]), ax.set_ylim(0, 10), ax.set_xlim(-0.6, 2.75)
    ax.set_ylabel("Average judge score (0–10)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GREY, label="before training"), Patch(color=GREEN, label="after training")],
              loc="upper left", frameon=False, ncol=2)
    save(fig, "03_training_lift", note=NOTE + " Thin lines: rough 95% range. Big models use their refined prompts.")


# ------------------------------------------------------------------ 4. four aspects
def four_aspects():
    fig = figure("Where the small models catch up, and where they don't",
                 "Training closes the gap on clarity, readability and structure; accuracy is the last mile")
    w = 0.36
    for k, a in enumerate(A):
        ax = fig.add_axes([0.07 + (k % 2) * 0.46, 0.47 - (k // 2) * 0.39, 0.36, 0.29])
        for i, (u, t) in enumerate([("q08_untrained", "q08_trained"), ("q4_untrained", "q4_trained"), ("q9_untrained", "q9_trained")]):
            for dx, lab, c, tc in [(-w / 2, u, GREY, INK), (w / 2, t, GREEN, "white")]:
                v = avg(lab, a)
                if v is None:
                    ax.bar(i + dx, 0.08, w * 0.95, color=c)
                    continue
                ax.bar(i + dx, v, w * 0.95, color=c, zorder=3)
                ax.text(i + dx, v - 0.2, f"{f1(v)}", ha="center", va="top", color=tc, fontsize=10.5, fontweight="bold", zorder=5)
        placed = []
        for name, lab, c in sorted([("Opus", "opus_v8", OPUS), ("Astra", "astra_v8", ASTRA), ("Luna", "luna_v8", LUNA)],
                                   key=lambda r: -avg(r[1], a)):
            v = avg(lab, a)
            ax.plot([-0.6, 2.5], [v, v], color=c, ls="--", lw=1.5, zorder=2)
            ty = v
            while any(abs(ty - p) < 0.7 for p in placed):
                ty -= 0.7
            placed.append(ty)
            ax.text(2.55, ty, f"{name} {f1(v)}", color=c, fontsize=10.5, fontweight="bold", va="center")
        ax.set_title(ASPECT_NAMES[a], loc="left")
        ax.set_xticks(range(3), ["0.8B", "4B", "9B"]), ax.set_ylim(0, 10), ax.set_xlim(-0.6, 3.2)
        ax.set_yticks([0, 5, 10])
    fig.text(0.04, 0.835, "■ before training", color="#8a958e", fontsize=13, fontweight="bold")
    fig.text(0.21, 0.835, "■ after training", color=GREEN, fontsize=13, fontweight="bold")
    save(fig, "04_four_aspects", note=NOTE + " Grey: before training (0.8B gave no usable answer). Green: after training.")


# ------------------------------------------------------------------ 5. glossary
def glossary():
    fig = figure("A reference glossary helps the small models a little",
                 "Same model, rewriting with and without definitions of the paper's terms")
    ax = fig.add_axes([0.3, 0.13, 0.62, 0.68])
    rows = [("0.8B, fully trained", "q08_trained_noglossary", "q08_trained", "glossary run also had 40% more data"),
            ("4B, fully trained", "q4_trained_noglossary", "q4_trained", "glossary run also had 40% more data"),
            ("4B, early checkpoint", "q4_s152_noglossary", "q4_s152_glossary", "glossary only at rewrite time")]
    for i, (name, a, b, note) in enumerate(rows):
        y = len(rows) - 1 - i
        va, vb = avg(a), avg(b)
        ax.plot([va, vb], [y, y], color=GREEN, lw=4, alpha=0.3)
        ax.scatter([va], [y], s=220, color="white", edgecolor=GREEN, lw=3, zorder=3)
        ax.scatter([vb], [y], s=220, color=GREEN, zorder=3)
        ax.text(va - 0.08, y, f"{f1(va)}", ha="right", va="center", color=GREEN)
        ax.text(vb + 0.08, y, f"{f1(vb)}", ha="left", va="center", color=GREEN, fontweight="bold")
        ax.text(4.35, y + (0.1 if note else 0), name, ha="right", va="center", fontsize=14, fontweight="bold")
        if note:
            ax.text(4.35, y - 0.2, note, ha="right", va="center", fontsize=10.5, color=MUTED)
        sa, sb = serious_per_paper(a), serious_per_paper(b)
        ax.text((va + vb) / 2, y - 0.25, f"serious errors per paper: {sa:.0f} → {sb:.0f}", ha="center", va="center",
                fontsize=11, color=MUTED)
    ax.set_xlim(4.5, 8.0), ax.set_ylim(-0.6, 2.6), ax.set_yticks([]), ax.spines["left"].set_visible(False)
    ax.set_xlabel("Average judge score (0–10)")
    ax.scatter([], [], s=110, color="white", edgecolor=GREEN, lw=2, label="without glossary")
    ax.scatter([], [], s=110, color=GREEN, label="with glossary")
    ax.legend(loc="upper right", frameon=False)
    save(fig, "05_glossary", note=NOTE + " Gains of 0.3–0.8 on 3 papers: suggestive, not yet proven.")


# ------------------------------------------------------------------ 6. more training
def more_training():
    fig = figure("The judge stopped improving long before the loss did",
                 "Validation loss keeps falling to the end, but checkpoints from early on already score as well")
    ax = fig.add_axes([0.08, 0.13, 0.78, 0.68])
    runs = [("q4_first", "4B (first run)", GREEN, "q4_s152_noglossary", 152, "q4_trained_noglossary"),
            ("q9", "9B", "#14432e", "q9_s424", 424, "q9_trained")]
    for key, name, c, early, step, final in runs:
        t = D["train"][key]
        xs = [v["step"] / t["steps"] for v in t["validation"]]
        ys = [v["loss"] for v in t["validation"]]
        ax.plot(xs, ys, color=c, lw=2.8, zorder=2)
        ax.text(1.03, ys[-1], name, color=c, fontweight="bold", va="center", fontsize=13)
        for x, lab in [(step / t["steps"], early), (1.0, final)]:
            y = min(t["validation"], key=lambda v: abs(v["step"] / t["steps"] - x))["loss"]
            ax.scatter([x], [y], s=150, color=c, edgecolor="white", lw=2, zorder=4)
            ax.annotate(f"judge {f1(avg(lab))}", (x, y), xytext=(0, 18), textcoords="offset points", ha="center",
                        fontsize=12.5, fontweight="bold", color=c,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c, lw=1))
    ax.set_xlabel("Share of the training pass"), ax.set_ylabel("Validation loss (lower = closer to the teachers)")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["start", "¼", "½", "¾", "end"]), ax.set_xlim(-0.02, 1.14)
    save(fig, "06_more_training", note="Judge: " + NOTE + " 4B first run: no glossary, 16% vs end. "
                                       "9B: glossary, 33% vs end.")


# ------------------------------------------------------------------ 7. quality vs price
def quality_vs_price():
    sp = D["speed"]
    words = sp["typical_words"]
    price = {}
    for r in sp["rows"]:
        if r["where"] in ("rtx6000", "API"):
            price[r["model"]] = r["usd_per_word"] * words * 1000
    fig = figure("Luna-level quality at under half Luna's price, 200× cheaper than Opus",
                 "Quality against the price of rewriting 1,000 typical papers (log scale)")
    ax = fig.add_axes([0.09, 0.13, 0.84, 0.68])
    pts = [("Our 0.8B", "q08_trained", price["our 0.8b"], GREEN_L), ("Our 4B", "q4_trained", price["our 4b"], GREEN),
           ("Our 9B", "q9_trained", price["our 9b"], "#14432e"), ("Luna", "luna_v8", price["luna"], LUNA),
           ("Opus", "opus_v8", price["opus"], OPUS), ("Astra", "astra_v8", price["astra"], ASTRA)]
    place = {"Our 0.8B": (1.3, 0, "left"), "Our 4B": (0.77, 0, "right"), "Our 9B": (0.77, 0.15, "right"),
             "Luna": (1.3, 0, "left"), "Opus": (1.25, 0.22, "left"), "Astra": (1.25, -0.22, "left")}
    for name, lab, p, c in pts:
        v, r = avg(lab), ci(lab)
        ax.plot([p, p], r, color=c, lw=2, alpha=0.5)
        ax.scatter([p], [v], s=330, color=c, zorder=3, edgecolor="white", lw=1.5)
        dx, dy, ha = place[name]
        ax.text(p * dx, v + dy, f"{name}\n${p:,.2f}" if p < 100 else f"{name}\n${p:,.0f}", va="center", ha=ha,
                fontsize=12.5, fontweight="bold", color=c)
    for name, lab, p, c in pts[3:]:
        ax.scatter([p / 2], [avg(lab)], s=150, facecolor="none", edgecolor=c, lw=1.5, zorder=2)
        ax.plot([p / 2, p], [avg(lab)] * 2, color=c, lw=1, ls=":", alpha=0.7)
    ax.text(price["opus"] / 2, avg("opus_v8") + 0.2, "batch price\n(slow, half off)", ha="center", fontsize=10, color=MUTED)
    ax.set_xscale("log"), ax.set_xlim(0.5, 4000), ax.set_ylim(4.5, 9.2)
    ax.set_xticks([1, 10, 100, 1000], ["$1", "$10", "$100", "$1,000"])
    ax.set_xlabel("Price per 1,000 papers (~7,100 words each)"), ax.set_ylabel("Average judge score (0–10)")
    ax.grid(alpha=0.25)
    save(fig, "07_quality_vs_price",
         note="Our models: rented RTX PRO 6000 at $1.80/h, kept busy. Big models: list prices × tokens measured on our papers. "
              "Vertical lines: rough 95% range.")


# ------------------------------------------------------------------ 8. time per paper
def time_per_paper():
    rows = {(r["model"], r["where"]): r for r in D["speed"]["rows"]}
    fig = figure("A paper in 15 seconds instead of 80",
                 "Time for one reader to get one paper rewritten (fastest setup we measured)")
    ax = fig.add_axes([0.2, 0.13, 0.72, 0.68])
    items = [("Astra", mean(rows[("astra", "API")]["seconds"]), ASTRA, "API"),
             ("Opus", mean(rows[("opus", "API")]["seconds"]), OPUS, "API"),
             ("Luna", mean(rows[("luna", "API")]["seconds"]), LUNA, "API"),
             ("Our 9B", mean(rows[("our 9b", "h100")]["seconds_mtp"]), "#14432e", "H100 + draft head"),
             ("Our 4B", mean(rows[("our 4b", "h100")]["seconds_mtp"]), GREEN, "H100 + draft head"),
             ("Our 0.8B", mean(rows[("our 0.8b", "h100")]["seconds"]), GREEN_L, "H100")]
    for i, (name, s, c, where) in enumerate(items):
        y = len(items) - 1 - i
        ax.barh(y, s, color=c, height=0.62)
        ax.text(s + 4, y, f"{s:.0f} s  ({where})", va="center", fontsize=12)
        ax.text(-6, y, name, ha="right", va="center", fontsize=14, fontweight="bold", color=c)
    a = rows[("astra", "API")]["seconds"]
    ax.text(mean(a) + 4, len(items) - 1 - 0.38, f"varied from {min(a):.0f} to {max(a):.0f} s", fontsize=10.5, color=MUTED, va="center")
    ax.set_yticks([]), ax.spines["left"].set_visible(False), ax.set_xlim(0, 380)
    ax.set_xlabel("Seconds per paper (mean of 3 papers, ~4,500 words)")
    save(fig, "08_time_per_paper",
         note="Big models through our Claude and ChatGPT logins. Draft head: the base model's multi-token-prediction head "
              "(judge: 9B 7.7 with it, 7.8–7.9 without). 4B with draft head: 2 papers.")


# ------------------------------------------------------------------ 9. GPU choice
def gpu_choice():
    rows = {(r["model"], r["where"]): r for r in D["speed"]["rows"]}
    words = D["speed"]["typical_words"]
    fig = figure("The cheaper GPU is the cheaper way to rewrite papers",
                 "One rented GPU kept full: RTX PRO 6000 ($1.80/h) against H100 ($4.50/h)")
    w = 0.36
    for k, (title, f, fmt) in enumerate([
            ("Papers per hour", lambda r: r["papers_per_hour"], "{:,.0f}"),
            ("Price per 1,000 papers", lambda r: r["usd_per_word"] * words * 1000, "${:.2f}")]):
        ax = fig.add_axes([0.07 + k * 0.48, 0.13, 0.4, 0.64])
        for i, m in enumerate(["our 0.8b", "our 4b", "our 9b"]):
            for dx, g, c in [(-w / 2, "rtx6000", GREEN), (w / 2, "h100", "#5b6770")]:
                v = f(rows[(m, g)])
                ax.bar(i + dx, v, w * 0.95, color=c)
                ax.text(i + dx, v, fmt.format(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(range(3), ["0.8B", "4B", "9B"]), ax.set_title(title, loc="left")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.text(0.07, 0.835, "■ RTX PRO 6000", color=GREEN, fontsize=13, fontweight="bold")
    fig.text(0.22, 0.835, "■ H100", color="#5b6770", fontsize=13, fontweight="bold")
    save(fig, "09_gpu_choice", note="128 unused papers sent at once; vLLM 0.29, bf16. Nebius on-demand prices, October 2026.")


# ------------------------------------------------------------------ 10. break-even
def break_even():
    rows = {(r["model"], r["where"]): r for r in D["speed"]["rows"]}
    words = D["speed"]["typical_words"]
    r9 = rows[("our 9b", "rtx6000")]
    cap, gpu = r9["papers_per_hour"], D["speed"]["gpu_per_hour"]["rtx6000"]
    fig = figure("A rented GPU pays off once it stays busy",
                 "Price per paper as the average number of papers per hour grows (our 9B on RTX PRO 6000, rented around the clock)")
    ax = fig.add_axes([0.09, 0.13, 0.8, 0.68])
    vols = [10 ** (i / 50) for i in range(0, 176)]
    cost = [math.ceil(v / cap) * gpu / v for v in vols]
    ax.plot(vols, cost, color="#14432e", lw=3, label="Our 9B, rented GPUs")
    for name, m, c in [("Opus", "opus", OPUS), ("Luna", "luna", LUNA)]:
        p = rows[(m, "API")]["usd_per_word"] * words
        ax.axhline(p, color=c, ls="--", lw=2)
        ax.text(60 if name == "Opus" else 1.15, p * 1.15, f"{name}: ${p:.3f} per paper" if p < 0.1 else f"{name}: ${p:.2f} per paper", color=c,
                fontweight="bold")
        be = gpu / p
        ax.scatter([be], [p], color=c, s=80, zorder=4)
        ax.annotate(f"cheaper than {name}\nabove {be:,.0f} papers/hour" if be >= 10 else f"cheaper than {name}\nabove {be:.1f} papers/hour",
                    (be, p), xytext=((4, 1.9) if name == "Opus" else (be * 1.6, p * 2.6)), fontsize=11, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=1))
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(1, 3000), ax.set_ylim(0.002, 3)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1,000"])
    ax.set_yticks([0.003, 0.01, 0.03, 0.1, 0.3, 1, 3], ["$0.003", "$0.01", "$0.03", "$0.10", "$0.30", "$1", "$3"])
    ax.set_xlabel("Average papers per hour"), ax.set_ylabel("Price per typical paper (log scale)")
    ax.grid(alpha=0.25, which="major")
    save(fig, "10_break_even", note=f"One GPU rewrites up to {cap:.0f} papers/hour; above that, add GPUs. "
                                    "Big models at list prices; batch (half price, slow) halves their lines.")


# ------------------------------------------------------------------ 11. serious errors
def errors():
    fig = figure("Our 9B: fewer serious mistakes than Luna, more than Opus or Astra",
                 "Major and critical problems the judge found, per paper")
    ax = fig.add_axes([0.2, 0.13, 0.72, 0.68])
    items = [("Opus", "opus_v8", OPUS), ("Astra", "astra_v8", ASTRA), ("Luna", "luna_v8", LUNA),
             ("Our 9B", "q9_trained", "#14432e"), ("Our 4B", "q4_trained", GREEN), ("Our 0.8B", "q08_trained", GREEN_L),
             ("Qwen 9B, untrained", "q9_untrained", GREY), ("Qwen 4B, untrained", "q4_untrained", GREY)]
    for i, (name, lab, c) in enumerate(items):
        y = len(items) - 1 - i
        v = serious_per_paper(lab)
        ax.barh(y, v, color=c, height=0.62)
        ax.text(v + 0.4, y, f"{f1(v)}", va="center", fontsize=12)
        ax.text(-0.6, y, name, ha="right", va="center", fontsize=13.5, fontweight="bold", color=c if c != GREY else MUTED)
    ax.set_yticks([]), ax.spines["left"].set_visible(False)
    ax.set_xlabel("Serious problems per paper (fewer is better)")
    save(fig, "11_serious_errors", note=NOTE + " Typical problems: a number attached to the wrong group, two similar terms confused.")


# ------------------------------------------------------------------ 12-13. head to head (Opus judge)
H2H = json.loads((HERE.parents[1] / "head_to_head" / "results.json").read_text())
H_COLORS = {"Opus": OPUS, "Sonnet": "#d9965b", "Astra": ASTRA, "Sol": "#a58fd0", "Terra 5.6": "#8fb3e8",
            "Luna": LUNA, "Our 9B": "#14432e", "Our 4B": GREEN, "Our 0.8B": GREEN_L}
H_NAMES = {"Opus": "Claude Opus 5.5", "Sonnet": "Claude Sonnet 5.5", "Astra": "GPT-6 Astra", "Sol": "GPT-6 Sol",
           "Terra 5.6": "GPT-5.6 Terra", "Luna": "GPT-6 Luna", "Our 9B": "Our 9B", "Our 4B": "Our 4B",
           "Our 0.8B": "Our 0.8B"}


def h2h_tally():
    t = {n: {"w": 0, "l": 0, "t": 0} for n in H2H["writers"]}
    for r in H2H["outcomes"]:
        for me, other in [(r["x"], r["y"]), (r["y"], r["x"])]:
            k = "t" if r["winner"] is None else "w" if r["winner"] == me else "l"
            t[me][k] += 1
    return t


def head_to_head():
    t = h2h_tally()
    rate = {n: (v["w"] + 0.5 * v["t"]) / (v["w"] + v["l"] + v["t"]) for n, v in t.items()}
    order = sorted(t, key=lambda n: -rate[n])
    fig = figure("Head to head: our 9B lands mid-pack, ahead of Luna and Terra",
                 "Every model against every other, part by part; which rewrite would a curious 14-year-old prefer?")
    ax = fig.add_axes([0.2, 0.165, 0.66, 0.655])
    for i, n in enumerate(order):
        y = len(order) - 1 - i
        v = t[n]; tot = v["w"] + v["l"] + v["t"]
        w, d, l = v["w"] / tot, v["t"] / tot, v["l"] / tot
        ax.barh(y, w, color=H_COLORS[n], height=0.66)
        ax.barh(y, d, left=w, color="#d8dde2", height=0.66)
        ax.barh(y, l, left=w + d, color="#f1f3f5", height=0.66)
        if w > 0.06:
            ax.text(w / 2, y, f"{v['w']}", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(1.02, y, f"{rate[n]:.0%}", va="center", fontsize=14, fontweight="bold", color=H_COLORS[n])
        ax.text(-0.02, y, H_NAMES[n], ha="right", va="center", fontsize=13.5, fontweight="bold",
                color=H_COLORS[n] if not n.startswith("Our 0") else "#4f9a72")
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xlim(0, 1), ax.set_yticks([]), ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of matchups (each model faces 8 others on 15 parts = 120 matchups)")
    fig.text(1.0 - 0.115, 0.83, "win rate", ha="center", fontsize=11, color=MUTED)
    fig.text(0.2, 0.845, "■ wins", color=GREEN, fontsize=12, fontweight="bold")
    fig.text(0.27, 0.845, "■ no clear winner", color="#b4bcc4", fontsize=12, fontweight="bold")
    fig.text(0.42, 0.845, "□ losses", color=MUTED, fontsize=12, fontweight="bold")
    save(fig, "12_head_to_head",
         note="Judge: Opus, blind; both orders asked, a win counts only if it holds both ways; win rate counts no clear winner as half.\n"
              "Same 3 held-out papers. Opus is also one of the writers, and tends to prefer rewrites that explain more.")


def head_to_head_matrix():
    t = h2h_tally()
    rate = {n: (v["w"] + 0.5 * v["t"]) / (v["w"] + v["l"] + v["t"]) for n, v in t.items()}
    order = sorted(t, key=lambda n: -rate[n])
    g = H2H["grid"]
    fig = figure("Who beats whom", "Row model's win rate against column model, 15 parts each (no clear winner counts half)",
                 size=(12, 9))
    ax = fig.add_axes([0.2, 0.08, 0.72, 0.72])
    import matplotlib.colors as mc
    cmap = mc.LinearSegmentedColormap.from_list("wl", ["#c45c4a", "#f4f1ea", "#2a7a55"])
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if a == b:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color="#e9ecef"))
                continue
            c = g[f"{a}|{b}"]; n = c["wins"] + c["losses"] + c["no_clear_winner"]
            r = (c["wins"] + 0.5 * c["no_clear_winner"]) / n
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap(r), ec="white", lw=2))
            ax.text(j + 0.5, i + 0.42, f"{r:.0%}", ha="center", va="center", fontsize=12.5, fontweight="bold",
                    color="white" if abs(r - 0.5) > 0.3 else INK)
            ax.text(j + 0.5, i + 0.72, f"{c['wins']}–{c['losses']}", ha="center", va="center", fontsize=9,
                    color="white" if abs(r - 0.5) > 0.3 else MUTED)
    ax.set_xlim(0, len(order)), ax.set_ylim(len(order), 0)
    ax.set_xticks([k + 0.5 for k in range(len(order))], order, rotation=35, ha="left", fontsize=11.5)
    ax.xaxis.tick_top()
    ax.set_yticks([k + 0.5 for k in range(len(order))], [H_NAMES[n] for n in order], fontsize=12)
    for lab, n in zip(ax.get_yticklabels(), order):
        lab.set_color(H_COLORS[n]); lab.set_fontweight("bold")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    save(fig, "13_head_to_head_matrix",
         note="Judge: Opus, blind, both orders. Small numbers: wins–losses out of 15 parts. Read across a row: green = that model usually wins.")


for f in [head_to_head, head_to_head_matrix]:
    f()
for f in [pipeline, prompts, training_lift, four_aspects, glossary, more_training, quality_vs_price, time_per_paper,
          gpu_choice, break_even, errors]:
    f()
