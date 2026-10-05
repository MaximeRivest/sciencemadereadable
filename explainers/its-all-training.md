# It's All Training: A Fully Synthetic Single-Stage Recipe for LLMs: explained

*A plain-language explainer of [It's All Training: A Fully Synthetic Single-Stage Recipe for LLMs](https://arxiv.org/abs/2609.37891v1). Written by Claude Opus from the paper; for the full details, read the original.*

# Teaching a Language Model Entirely with Made-Up Practice Material: An Explainer

## First, some background

A **large language model** (LLM), such as the systems behind ChatGPT, is a computer program that learns to write by reading enormous amounts of text. During **training**, the model sees a piece of text and keeps trying to guess the next word. Each time it guesses wrong, its internal settings are nudged slightly. Those settings are called **parameters**, and big models have millions or billions of them. Text is measured in **tokens**, which are chunks of words, roughly a word or part of a word each.

Building a modern model usually happens in several stages:

- **Pre-training**: the model reads a huge pile of general text, mostly copied from the web, so it learns language and facts.
- **Mid-training**: extra, more targeted material is added, such as step-by-step reasoning examples.
- **Post-training**: the model is taught to behave like an assistant. It learns to follow instructions, answer in a requested format, and refuse when appropriate. This is often done with "supervised fine-tuning" (showing it good example answers) and "reinforcement learning" (rewarding good behaviour).

This paper is by a team mostly from a company called PleIAs. They ask whether all of that could be collapsed into **one single training stage**, using a dataset that is entirely **synthetic**, meaning text written by other AI models rather than by people.

## The problem the authors want to solve

The authors point to several issues with the usual approach:

1. **Web data is messy and hard to control.** You can't easily choose what facts a model learns from a random web crawl. Much of the web is also copyrighted, and some website owners now block AI companies from collecting their pages.
2. **Web data wasn't designed for later training stages.** It contains very little explicit reasoning, the "show your work" style of thinking that newer models are expected to do.
3. **Big labs already make synthetic data, but keep it secret.** Companies generate their own internal datasets, for example reasoning examples produced by a strong existing model, but none of these are public. So outsiders don't understand well how such data affects what a model learns, especially for small models.
4. **Simply copying a big model's outputs has risks.** If one model's answers are used to train another, any mistakes get copied too. Repeatedly training models on AI-written text can cause **model collapse**, where outputs become narrow and repetitive over generations.

## What earlier work did

The paper situates itself among a few lines of research:

- **Rephrasing**: taking real text and having an AI rewrite it, which can help models memorize facts better than just repeating the original.
- **Fully synthetic training**: Microsoft's earlier "Phi" models were trained mostly on AI-written textbooks and did surprisingly well. The authors note, though, that relying on a single style ("write a textbook") leaves the data lacking variety, and later Phi versions added web data back in.
- **Controlled experiments on memory**: researchers have estimated how much a model can memorize per parameter (roughly 2–3.6 "bits" per parameter in different studies). One finding the authors highlight is that mixing useful data with junk drastically reduces how much useful knowledge a model can store. They treat this as motivation for building training data carefully from chosen sources.

## What they built: the Synth dataset

**Synth** is the paper's main creation: almost **80 billion tokens** of synthetic text in 8 languages. All of it grows out of a fixed set of **"seeds"**, real documents that the generated text is based on.

**The seeds.** The core is about 50,000 Wikipedia "vital articles", a community-chosen list of the most important topics. The authors added about 8,700 specialist articles in law, medicine and chemistry (the paper's total of 58,698 articles includes these). They also added about 3,700 Wikibooks pages, mostly cooking and practical know-how, plus 130 documents about the model itself, recent events and AI research.

**The key trick: "back-translation."** Normally you start with a question and get an answer. Synth works backwards. It starts from a real Wikipedia passage, the "answer material", and generates questions that this passage could answer. Then it produces a reasoned answer grounded in that passage. Because every answer is tied to a real source text, the authors can control exactly which facts the model is meant to learn.

**How the data is generated.** There are two stages:

- **Stage 1: training helper models.** The authors take a mid-sized open model (Gemma-3-12B) and lightly adapt it into two specialized "auxiliary" helpers. They use a cheap add-on technique called LoRA and train it on examples produced by a stronger "frontier" model. One helper writes **questions**. The other writes **answers with reasoning**.
- **Stage 2: generating at scale.** The helpers run over every seed passage. Each passage is "amplified" 100 times, meaning roughly 100 different generated examples per piece.

**Keeping the data varied.** The authors found that simply prompting one big model with changing instructions produced repetitive phrasing. So for each question, they randomly pick settings along six "dials":

- question type
- difficulty
- the kind of user asking
- what the ideal response should be
- language
- writing style

This forces variety along chosen directions rather than whatever the AI finds easiest to vary.

**Teaching the model to say "I don't know."** One design choice the authors specifically flag: about **20%** of questions are deliberately ones where the right response is a refusal, a correction or a hedge. Examples include questions based on a false premise, absurd questions and ambiguous ones. Without these, every training question would have a confident answer, and the model would learn to always answer, even when it shouldn't.

**Mixing in a neighbouring passage.** The answer-writing helper sees the question, its seed passage, and also the most similar *other* passage in the collection. That passage is found with a search tool that compares meaning (bge-m3 plus FAISS). The idea is to create links between related topics that a single passage wouldn't provide.

**A shorthand for reasoning.** The reasoning in Synth isn't written in ordinary sentences. It uses a compact symbol system:

- arrows and "therefore" signs for logic
- symbols marking how **certain** or **uncertain** a claim is (called **epistemic markers**; "epistemic" means "about knowledge")
- check-marks for verification steps

These symbols were added as special tokens, so a whole reasoning step takes only one or two tokens. That helps a small model fit more thinking into its limited working memory. The traces also include made-up "uncertainty level" tags at decision points. These are only labels in the training data; they don't control anything when the model is used.

**Other kinds of tasks.** Besides memorizing facts, Synth includes:

- **RAG** (retrieval-augmented generation): practice answering using up to ten supplied documents and citing them, versus answering from memory
- **arithmetic**: about 3,000 maths problem templates with randomized numbers, solved by another model, Qwen-3-8B, which the authors found more accurate at maths
- **creative writing** with odd constraints, such as avoiding a letter or shaping a poem visually
- **editing**: translation, extracting information, fixing errors, changing style
- **multiple-choice questions**
- **cooking recipes**

About 20% of the data isn't in English, mostly European languages. The reasoning parts stay in English.

**Quality check.** The authors used an outside automatic judge of data quality, called Propella-1, to compare Synth with other public training datasets. By that judge, Synth came out on top for educational value and reasoning, by a wide margin. It tied a cleaned Wikipedia dataset (FineWiki) on safety. The authors acknowledge the reasoning score is partly boosted simply because Synth contains reasoning traces. They say the educational-value lead doesn't depend on that.

## The models they trained

They trained four models from scratch, only on Synth, with **no separate fine-tuning or reinforcement learning stage**:

- **Monad** (56 million parameters): a deliberately tiny "stress test".
- **Baguettotron-350M** (about 321 million parameters): unusually deep (80 layers), testing a hunch that deeper models might benefit more from reasoning data.
- **Baguettotron-600M** (about 594 million): a more conventional design and the main model in the paper.
- **Baguettotron-MoE**: a **Mixture-of-Experts** model. It has 16 "expert" sub-networks totalling 13.2 billion parameters. For each token, a small "router" picks just one expert, so only about 1 billion parameters are actually used at a time. This gives a big model's storage at a smaller model's running cost.

The MoE reached the same training error as the 600M model using only about a third as many tokens (around 50 billion).

**Going past the usual "rule of thumb".** A well-known study (nicknamed "Chinchilla") suggested the best use of computing power is roughly 20 training tokens per parameter. That would be about 12 billion tokens for a 600M model. The authors trained theirs on 264 times its parameter count, far more, and say its error was still falling at the end with no instability. They also compared runs on Synth against two real-text datasets. They describe this as **preliminary**, but say it suggests Synth allows a lower eventual error. They caution that error numbers aren't directly comparable across datasets, which is why they also compare on actual tasks.

## How well do the models do?

**Against other small open models.** The authors tested on 22 multiple-choice and 8 open-ended tasks. They compared against four open models of 270M–600M parameters: Gemma-3-270M, SmolLM2-360M, LFM2.5-350M and Qwen3-0.6B. They note these models differ in many ways, so the comparison is "context" rather than a controlled test.

- Their best models trail the strongest comparison, Qwen3-0.6B, by **4.7 points** on multiple-choice and only **0.7 points** on open-ended tasks, while being trained on **80–700 times fewer tokens**.
- Tiny Monad beats Gemma-3-270M and SmolLM2-360M on the multiple-choice average.
- Results vary a lot by test. The models do relatively well on truthfulness and some specialized science and industry quizzes, such as NuclearQA. They do noticeably worse on tests needing broad general web knowledge, such as ARC-Challenge and GeoBench. The authors attribute this to the limited set of seed sources.

**A fairer test: same model, different data.** To check that the data, not the model design, is responsible, they trained two identical 600M models on real text instead: one on Wikipedia text (FineWiki) and one on educational PDFs (FinePDFs-Edu).

- Without post-training, these web-trained models couldn't even produce valid multiple-choice answers. So the authors gave them a short post-training stage with a conversation dataset plus practice questions.
- Even then, the Synth model led by **16–17 points** on multiple-choice and **11–14 points** on open-ended tasks.
- The authors' takeaway: web pre-training needs a second stage to become usable, but Synth's single stage already includes instruction-following. They argue the advantage goes beyond just knowing the answer format.

**Do the reasoning traces matter?** They retrained the 600M model on Synth with the reasoning removed, keeping everything else matched.

- Multiple-choice accuracy barely changed (41.8% vs 42.2%).
- Open-ended accuracy dropped from 24.3% to 22.0%.
- The losses were concentrated in truthfulness (−9.1 points on TruthfulQA) and domain reasoning (−10.0 on NuclearQA). Fact recall was basically unaffected.

## Are the models' facts actually correct?

Because every training answer is grounded in a real Wikipedia passage, the authors expected their models to make up fewer false facts. To test this, they used a method inspired by **FActScore**:

1. Pick 500 Wikipedia topics from the seed set.
2. Ask each model "What do you know about [topic]?"
3. Have another AI (DeepSeek-V3.2) break the answer into small individual claims.
4. Label each claim as **supported**, **contradicted**, or **inconclusive** by the Wikipedia article.

They report two scores. **Precision** is the share of checkable claims that are correct. The **macro** score also counts inconclusive claims against the model.

The results:

- Baguettotron-600M and Baguettotron-MoE got the highest macro scores in the comparison: **41.7%** and **46.3%**. That beats Phi-4-mini (a model 6 times larger) and DeepSeek-MoE-16B (which uses 2.8 times more active parameters and over 40 times more training tokens).
- The 350M model beat similar-sized rivals while seeing **10–140 times fewer tokens**.
- The only comparison model trained on a similar number of tokens, OPT-350M, was the only one that stated more wrong facts than right ones.

The authors conclude their models are the most token-efficient factual learners in every size group, and the only ones above 40% macro.

**But isn't that just because the test topics were in the training seeds?** Partly, yes, and the authors are upfront about it: no model can report facts it never saw. To check that the model isn't just scoring well by saying vague, safe things, they tested 600 Wikipedia topics that were *not* in the seeds:

- On these unfamiliar topics, Baguettotron-MoE **declined to answer 67% of the time**, versus 20% on familiar topics. Its precision on the questions it did attempt fell from 82% to 62%.
- A web-trained MoE of similar active size (OLMoE) declined only 7% of the time and kept 81% precision. The authors say this is expected, since web data covers many more of these topics.

They read this as evidence that their scoring reflects what the model actually knows, and that the model recognizes the limits of its knowledge.

**Do the "certainty" symbols mean anything?** The authors sorted each model's reasoning traces into "mostly confident" or "mostly uncertain" based on which symbols dominated. For the 350M, 600M and MoE models, uncertain-tagged answers were indeed less accurate:

| Model | Accuracy, confident traces | Accuracy, uncertain traces |
|---|---|---|
| Baguettotron-350M | 0.35 | 0.25 |
| Baguettotron-600M | 0.44 | 0.40 |
| Baguettotron-MoE | 0.50 | 0.41 |
| Monad | 0.17 | 0.17 |

The three larger models also made about 15–20% fewer claims when uncertain. Tiny Monad's accuracy showed no difference, but it still said a bit less when uncertain (11.7 → 10.6 claims per topic). The authors suggest that "saying less when unsure" might be learned at smaller sizes than "being more right when sure". They frame this as a suggestion, not a firm result. For models above roughly 300M parameters, they say the markers work as a usable confidence signal.

## Adapting to specialized subjects

The authors argue a big advantage of this approach is that you can aim it at a field where little training material exists. They tried **telecommunications**, which is full of dense jargon. The seeds were about 3 million tokens of telecom Wikipedia plus about 90 million tokens of technical standards documents (3GPP, which defines mobile networks such as 5G). From these they generated about 460 million synthetic tokens and further trained the 600M model.

- A telecom quiz (TeleQnA) improved from **41.6% to 56.7%**.
- Factual accuracy on the standards improved from **21.5% to 38.8%**.
- An AI judge (Claude Opus 4.5) graded 50 telecom questions out of 100. The specialized model scored **67.8**, versus 58.7 for a version trained only on telecom Wikipedia and 27.8 for the original model. The gains were mostly in precise knowledge and in correctly expanding acronyms, where the original model tended to invent plausible but wrong ones.

**Tool calling.** "Tool calling" means the model outputs a correctly formatted command to use an outside program, like a weather lookup. The original model couldn't do this at all. The authors used the same two-helper method to generate 532,000 examples and continued training. On a standard test (BFCL v2), Baguettotron-600M scored **53.1%**. That is 3.9 points above Google's FunctionGemma-270M, a model built for this job, using about 40 times fewer training tokens.

## What the authors think it all means

These are the authors' interpretations, not established facts:

- They argue the famous "20 tokens per parameter" rule was worked out on web text, which isn't well matched to what small models need. In their view, carefully engineered data shifts that curve, so training stays useful far longer.
- They see their results as backing the idea, first suggested by the Phi models, that **data quality** is a lever as important as model size and amount of data, and one that research has under-explored.
- They argue that open sources like Wikipedia make good foundations for this kind of pipeline. In their view this approach can make training data more shareable and transparent than scraped web data.

## Limitations the authors acknowledge

- **Knowledge ceiling.** The model can only know what is in the roughly 58,000 seed articles. That gives precise control but caps what it can know.
- **Cultural bias.** The seeds reflect what English-speaking Wikipedia editors consider important. The other languages were generated from these English-shaped seeds rather than native-language content.
- **Difficulty level.** The exercises are pitched at small models. Larger models would need new pipelines covering coding, multi-step tool use and long tasks.
- **No web data at all.** Leaving web data out kept the experiment clean. But it also meant the team had to deliberately engineer behaviours, like refusing, that might come naturally from real-world text at larger scale.

Finally, the authors have publicly released the Synth dataset and the Baguettotron models under a permissive licence, so others can build on them.