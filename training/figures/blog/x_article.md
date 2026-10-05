# Small open models can rewrite science papers for curious readers. Here's what it took.

Research papers are written for other researchers. We want a curious 14-year-old to be able to read them too: the same paper, in the authors' own voice, with every result kept, but in words they can follow.

The big models already do this well. Claude Opus rewrites a paper beautifully, but it costs about a dollar per paper and takes about 80 seconds. So we asked: can a small open model, running on a single rented GPU, learn to do it?

Short answer: yes, mostly. Our 9-billion-parameter model scores slightly above GPT-6 Luna on our test, for under half Luna's price and 200 times less than Opus. It still gets more facts wrong than Opus or Astra.

Here is how we got there, with everything we measured.

[FIGURE 01_pipeline]

## How it works

1. **5,000 open-access papers** from PubMed Central, all under a CC BY licence, so rewriting and republishing them is allowed.
2. **Big models write the examples.** Opus and GPT-6 Astra rewrote 4,416 of them, giving 21,069 training examples (the opening of each paper, then each section).
3. **A reference glossary.** For every passage, we look up the paper's terms in an offline copy of Wikipedia and give the definitions to the writer.
4. **Small models learn.** We fine-tuned Qwen 3.5 at three sizes, 0.8B, 4B and 9B, with one pass over the examples.
5. **One judge scores everyone.** Opus grades each rewrite from 0 to 10 on four things: faithful and exact, understandable, pleasant to read, and structure and voice. Every model is judged on the same 3 held-out papers (15 parts).

## Step zero: better instructions

Before training anything, we spent time on the instructions. That alone lifted every big model by 1.5 to 2.8 points.

[FIGURE 02_prompts_matter]

## Training works at every size

Untrained, the 0.8B couldn't do the task at all, and the 4B and 9B were weak. One pass over the examples changed that: 5.7, 7.2 and 7.8. The 9B lands just above Luna (7.5).

[FIGURE 03_training_lift]

Broken down by aspect, structure and readability catch up first. Accuracy (keeping every number and claim right) is the last mile.

[FIGURE 04_four_aspects]

## The glossary helps a little

Giving the model definitions of the paper's terms adds a few tenths of a point each time and removes a few serious errors. With only 3 test papers, that's suggestive, not proven.

[FIGURE 05_glossary]

It also has limits. In one plant-biology paper, all our small models (and Luna, in one section) mixed up "hermaphroditic" and "monoecious" plants, and our glossary's definitions were too vague to prevent it. Better definitions are next on our list.

## Most of the training time bought nothing we could measure

The usual training metric kept improving until the very end. The judge didn't: checkpoints a sixth (4B) and a third (9B) of the way through scored as well as the final models.

[FIGURE 06_more_training]

## Quality against price

This is the chart that matters most to us. Our models cost what a rented GPU costs per hour, divided by the papers it can rewrite in that hour. The big models cost their list price times the tokens they actually used on our papers.

[FIGURE 07_quality_vs_price]

## Speed

On one rented H100, our 9B rewrites a paper in about 15 seconds. The big models take about 80 seconds, and Astra anywhere from 100 to 500. The 4B and 9B get there with a "draft head" that guesses a few words ahead while the model checks them; the judge scored the 9B the same with and without it.

[FIGURE 08_time_per_paper]

The H100 is the faster GPU, but the cheaper RTX PRO 6000 rewrites each paper 15–25% cheaper.

[FIGURE 09_gpu_choice]

## The catch: a rented GPU only pays off when it's busy

A GPU costs the same per hour whether it works or not. Ours beats Opus's price almost immediately, but it only beats Luna once it rewrites more than about 170 papers an hour on average.

[FIGURE 10_break_even]

## Mistakes

Our 9B makes fewer serious mistakes than Luna, but several times more than Opus or Astra. The typical ones: a number attached to the wrong group, or two similar terms confused.

[FIGURE 11_serious_errors]

## What we'd keep in mind

- **Three test papers is a small test.** Gaps under half a point could be luck. A 30-paper test is next.
- **Opus is both a teacher and the judge.** It could favour writing like its own. Astra scoring higher than Opus on accuracy suggests the effect is small, but it's there.
- **Speeds for the big models** were measured through our Claude and ChatGPT logins; their paid APIs may be faster.

## What's next

Better glossary definitions for terms that are easy to confuse, a bigger test, and shorter training runs, since the extra training didn't help. And then: a reading list of real papers that curious kids can actually read.
