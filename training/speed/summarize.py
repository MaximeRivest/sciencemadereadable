"""Time and price per paper for the three students (rented GPUs) and the three API writers.

    python3 research/training/speed/summarize.py

Time per paper: the 3 benchmark test papers, one at a time (what one reader waits).
Price per paper: students = GPU price per hour / papers per hour with the GPU kept full
(128 unused corpus papers sent at once); API writers = their measured tokens x list price.
Papers differ in length (test papers ~4,500 words, the 128-paper set ~7,000), so prices
are given for a typical paper of TYPICAL_WORDS words, each scaled by words.
"""
import json
from pathlib import Path
from statistics import mean

S = Path(__file__).resolve().parent
GPU_PER_HOUR = {"rtx6000": 1.80, "h100": 4.50}   # Nebius on-demand, 2026-10-04 (nebius.com/prices)
API = {  # USD per million tokens, standard tier (batch: half), checked 2026-10-04
    "opus": (4.0, 20.0, "Claude Opus 5.5 (platform.claude.com pricing)"),
    "astra": (10.0, 50.0, "GPT-6 Astra (developers.openai.com)"),
    "luna": (0.10, 0.50, "GPT-6 Luna (developers.openai.com)"),
    "sonnet": (2.0, 10.0, "Claude Sonnet 5.5 (platform.claude.com pricing)"),
    "sol": (2.0, 10.0, "GPT-6 Sol (developers.openai.com)"),
    "terra": (2.0, 12.0, "GPT-5.6 Terra (developers.openai.com)"),
}
load = lambda p: json.loads((S / p).read_text()) if (S / p).exists() else None


def valid_seconds(run):
    """Per-paper times, leaving out a paper whose request never reached the server
    (connection error): its time would be missing a section."""
    bad = {c["paper"] for c in run["calls"] if str(c["ok"]).startswith(("TransportError", "Connection"))}
    return [r["total_seconds"] for r in run["per_paper"] if r["paper"] not in bad]


test_words = None
rows = []
for size in ["0.8b", "4b", "9b"]:
    for gpu, price in GPU_PER_HOUR.items():
        lat, thr, mtp = load(f"latency-{gpu}-{size}.json"), load(f"throughput-{gpu}-{size}.json"), \
            load(f"latency-{gpu}-{size}-mtp.json")
        if not lat:
            continue
        test_words = mean(r["words_in"] for r in lat["per_paper"])
        row = {"model": f"our {size}", "where": gpu, "seconds": valid_seconds(lat)}
        if mtp:
            row["seconds_mtp"] = valid_seconds(mtp)
        if thr:
            words = mean(r["words_in"] for r in thr["per_paper"])
            pph = thr["papers"] / thr["wall_seconds"] * 3600
            row.update(papers_per_hour=pph, words_per_paper=words, usd_per_word=price / pph / words,
                       failed_calls=thr["failed_calls"], calls=len(thr["calls"]))
        rows.append(row)
for w, (pin, pout, src) in API.items():
    d = load(f"api-{w}.json")
    if not d:
        continue
    words = test_words or 4460
    usd = [sum(c["usage"].get("input_tokens", 0) for c in p["calls"]) * pin / 1e6 +
           sum(c["usage"].get("output_tokens", 0) for c in p["calls"]) * pout / 1e6 for p in d["papers"]]
    rows.append({"model": w, "where": "API", "seconds": [p["seconds"] for p in d["papers"]],
                 "usd_per_test_paper": mean(usd), "usd_per_word": mean(usd) / words})

TYPICAL_WORDS = round(mean(r["words_per_paper"] for r in rows if "words_per_paper" in r), -2) if \
    any("words_per_paper" in r for r in rows) else 7000
print(f"time = mean of the 3 test papers (~{test_words:.0f} words), one at a time; "
      f"price = for a typical {TYPICAL_WORDS:.0f}-word paper\n")
print(f"{'model':<10}{'where':<9}{'s/paper':>9}{'(range)':>12}{'with MTP':>10}{'papers/h':>10}{'$/paper':>10}"
      f"{'$/1000 papers':>15}")
for r in rows:
    usd = r.get("usd_per_word", 0) * TYPICAL_WORDS if "usd_per_word" in r else None
    rng = f"{min(r['seconds']):.0f}-{max(r['seconds']):.0f}"
    m = f"{mean(r['seconds_mtp']):.0f}" if "seconds_mtp" in r else ""
    pph = f"{r['papers_per_hour']:.0f}" if "papers_per_hour" in r else ""
    print(f"{r['model']:<10}{r['where']:<9}{mean(r['seconds']):>9.0f}{rng:>12}{m:>10}{pph:>10}"
          f"{(f'{usd:.4f}' if usd is not None else ''):>10}{(f'{usd * 1000:,.2f}' if usd is not None else ''):>15}")
(S / "summary.json").write_text(json.dumps({"typical_words": TYPICAL_WORDS, "test_words": test_words,
                                            "gpu_per_hour": GPU_PER_HOUR, "api_prices": API, "rows": rows}, indent=1))
