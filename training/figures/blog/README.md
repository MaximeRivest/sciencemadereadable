# Figures for the blog / course

Thirteen figures, in story order. Each is in `out/` as PNG (1920×1200) and SVG (editable text).
Rebuild: `export_data.py` (collects the numbers, no model calls), then `make_figures.py`.

**Applies to every quality figure:** an Opus judge scores each rewrite from 0 to 10 on four aspects. Every model is judged on the same 3 held-out papers (15 parts). Three papers is a small test, so gaps under about half a point may be luck. Thin lines show a rough 95% range.

---

## 1. The idea

![Diagram: 5,000 open papers, rewritten by frontier models into 21,069 examples, used to train small Qwen 3.5 models, judged by Opus; a reference glossary feeds both the writers and the small models.](out/01_pipeline.png)

**Caption:** Big models write the examples, small open models learn from them, and one judge scores everyone the same way. A glossary built from an offline copy of Wikipedia gives writers and students the definitions of each paper's terms.

## 2. Instructions matter before anything else

![Dumbbell chart: the refined prompts raise Opus from 5.6 to 8.5, Astra from 5.8 to 8.2 and Luna from 6.0 to 7.5.](out/02_prompts_matter.png)

**Caption:** Before training anything, we rewrote the instructions: from 5.6–6.0 to 7.5–8.5. Part of the gain comes from a change of goal (the first prompts also corrected papers and allowed lists).

## 3. Training works at every size

![Bar chart: before training, Qwen 0.8B gives no usable answer, 4B scores 3.8 and 9B 5.8. After training they score 5.7, 7.2 and 7.8. Reference lines: Opus 8.5, Astra 8.2, Luna 7.5.](out/03_training_lift.png)

**Caption:** One pass over the examples turns models that can't do the task into usable science writers. The trained 9B (7.8) scores just above Luna (7.5).

## 4. What improves, and what's left

![Four panels, one per judging aspect. The trained 9B reaches 7.1 on accuracy, against 7.2 for Luna and about 8.3–8.5 for Opus and Astra. It matches or beats the big models on structure and readability.](out/04_four_aspects.png)

**Caption:** Structure and readability catch up first. Accuracy (keeping every number and claim right) is the last mile.

## 5. The glossary

![Dumbbell chart: with the glossary, 0.8B goes from 4.9 to 5.7, the fully trained 4B from 6.9 to 7.2, an early 4B checkpoint from 6.9 to 7.3; serious errors drop a little in each case.](out/05_glossary.png)

**Caption:** Giving the model definitions of the paper's terms helps a little each time, and removes a few serious errors. The two fully trained comparisons also differ in data (the glossary runs had 40% more), so this is suggestive, not proven.

## 6. When to stop training

![Validation loss for the 4B (first run) and 9B falls steadily to the end of training, but the judge gives the same score to an early checkpoint and the final model: 6.9 and 6.9 for the 4B, 7.9 and 7.8 for the 9B.](out/06_more_training.png)

**Caption:** The usual training metric (validation loss) kept improving to the end, but the judge didn't: checkpoints 16% (4B) and 33% (9B) of the way through already scored as well. Most of the training time bought nothing we could measure.

## 7. Quality against price (headline)

![Scatter, log price axis: our 0.8B at $1.08 per 1,000 papers (5.7), 4B at $3.57 (7.2), 9B at $4.58 (7.8); Luna $10.72 (7.5); Opus $975 (8.5); Astra $1,206 (8.2). Hollow circles show the big models' half-price batch option.](out/07_quality_vs_price.png)

**Caption:** Our 9B scores just above Luna at under half its price, and costs about 200 times less than Opus. Opus and Astra are still more accurate.

## 8. Speed

![Horizontal bars, seconds per paper: Astra 301, Opus 83, Luna 82, our 9B 15, our 4B 9, our 0.8B 9.](out/08_time_per_paper.png)

**Caption:** On one rented H100, our 9B rewrites a paper in about 15 seconds. The big models take about 80 seconds (Astra 100–500). The 4B and 9B reach this speed with a "draft head" that guesses a few words ahead; the judge scored the 9B the same with it.

## 9. Which GPU to rent

![Two bar panels. Papers per hour, RTX PRO 6000 vs H100: 1,662 vs 3,117 (0.8B), 502 vs 1,064 (4B), 392 vs 807 (9B). Price per 1,000 papers: $1.08 vs $1.44, $3.57 vs $4.22, $4.58 vs $5.56.](out/09_gpu_choice.png)

**Caption:** The H100 is about twice as fast, but the RTX PRO 6000 costs 40% as much, so it rewrites each paper 15–25% cheaper.

## 10. When owning beats paying per paper

![Log-log line: price per paper for the 9B on a GPU rented around the clock falls as volume grows; it is cheaper than Opus above 1.8 papers per hour and cheaper than Luna above 168 papers per hour.](out/10_break_even.png)

**Caption:** A rented GPU costs the same per hour whether it works or not. It beats Opus almost immediately, but it only beats Luna once it rewrites more than about 170 papers an hour on average.

## 11. Serious mistakes

![Horizontal bars, serious problems per paper: Opus 0.7, Astra 2.0, Luna 9.0, our 9B 5.7, our 4B 11.3, our 0.8B 32.7, untrained Qwen 9B 25.0, untrained Qwen 4B 44.7.](out/11_serious_errors.png)

**Caption:** Our 9B makes fewer serious mistakes than Luna, but still several times more than Opus or Astra. The typical ones: a number attached to the wrong group, or two similar terms confused.

## 12. Head to head (headline for quality)

![Stacked bars of win rate in blind head-to-head comparisons, judged by Opus: Claude Opus 5.5 88%, GPT-6 Astra 75%, Claude Sonnet 5.5 67%, GPT-6 Sol 59%, our 9B 55%, our 4B 40%, GPT-5.6 Terra 35%, GPT-6 Luna 29%, our 0.8B 2%.](out/12_head_to_head.png)

**Caption:** Every model against every other, part by part, judged blind. Our 9B lands in the middle of the pack: behind Opus, Astra, Sonnet and (narrowly) Sol, ahead of Terra and Luna. Opus is also one of the writers and tends to favour rewrites that explain more.

## 13. Who beats whom

![Heatmap of each model's win rate against each other model, 15 parts per pair. Our 9B beats Luna and Terra 11 to 4, is level with Sol (7 to 8), and loses to Sonnet (5 to 7), Astra (3 to 10) and Opus (1 to 11).](out/13_head_to_head_matrix.png)

**Caption:** Read across a row: green means that model usually wins. Our 9B's closest contest is with GPT-6 Sol, which costs about 40 times more per paper.

---

## Fine print (for a methods box)

- **Models:** Qwen 3.5 0.8B (all weights), 4B (LoRA), 9B (all weights), one pass over 20,394 training conversations. Big models with their refined (v8) prompts: Claude Opus 5.5, GPT-6 Astra (low reasoning), GPT-6 Luna.
- **Prices:** Nebius on-demand (RTX PRO 6000 $1.80/h, H100 $4.50/h) and API list prices (Opus $4/$20, Astra $10/$50, Luna $0.10/$0.50 per million tokens in/out), October 2026. Our models' price assumes the GPU is kept busy. A "typical paper" is about 7,100 words.
- **Speed:** time for one paper sent alone, mean of 3 papers. Big models went through our Claude and ChatGPT logins; their paid APIs may be faster.
- **Head to head:** 9 writers, 15 parts, every pair asked in both orders (1,080 Opus calls); a win only counts if it holds both ways. A second run with GPT-6 Astra as judge ranked models very differently (it put itself first and Opus sixth), so we report the Opus judge and say so.
- **Not tested:** other GPUs (L40S, B200/B300), 8-bit models, more test papers.
