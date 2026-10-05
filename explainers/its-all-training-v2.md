# It's All Training: A Fully Synthetic Single-Stage Recipe for LLMs: explained

*A plain-language explainer of [It's All Training: A Fully Synthetic Single-Stage Recipe for LLMs](https://arxiv.org/abs/2609.37891v1), written by Claude Opus with the rewrite pipeline's writing brief, reader habits and a reference glossary from Wikipedia. Not a rewrite; for full details read the original.*

# Training a Small AI Entirely on Made-Up Practice Material: An Explainer

## The problem: AI is mostly trained on the messy internet

A large language model, or LLM, is a computer program trained on a huge amount of text so that it can write, summarize, translate and answer questions. The chatbots many people use are built this way.

These models usually learn in several stages. First comes **pre-training**: the model reads an enormous pile of text and learns to guess the next word. Most of that pile is copied from the open web. Later stages teach particular behaviors. In **mid-training** and **post-training**, the model learns to follow instructions, hold a conversation and reason step by step. Some of this later teaching uses methods with names like "supervised fine-tuning" and "reinforcement learning."

The authors of this paper, a team led by researchers at the company PleIAs, point to several problems with relying on the web:

- **It's hard to control.** You get whatever happens to be online, good or bad.
- **It's getting harder to collect.** Some website owners now block AI companies from copying their pages.
- **There are legal worries.** Most web pages are protected by copyright, which makes it hard for researchers to share training data openly.
- **It doesn't match later training.** Web text contains very little explicit, step-by-step reasoning, so separate stages are needed afterward to teach those skills.

Many big AI labs have started adding **synthetic data** to the mix. This is text written by existing AI models instead of by people. But these datasets are not public, and nobody fully understands how this kind of data shapes what a model knows and can do, especially for small models.

Synthetic data also has known risks. If one AI simply copies another's answers, any mistakes get copied too. If models keep training on other models' output, they can gradually lose variety and quality, a failure researchers call "model collapse."

Earlier attempts at training mostly on synthetic text, such as Microsoft's Phi models and an open copy called Cosmopedia, leaned heavily on a single recipe: rewriting material as textbooks. The authors note that this left the text too samey on the surface, and later Phi versions went back to mixing in web data.

## The big idea: grow a whole training library from a small, trusted seed

The authors built **Synth**, an openly released collection of almost 80 billion "tokens" of synthetic text in 8 languages. A token is a small chunk of text, often a word or part of a word. A **tokenizer** is the tool that chops text into these chunks.

All of Synth grows from a carefully chosen starting set:

- about 50,000 Wikipedia "vital articles," a list the Wikipedia community keeps of the most important topics;
- 8,698 extra articles on law, medicine and chemistry;
- 3,727 pages from Wikibooks, a free collection of books, mostly about cooking and practical knowledge;
- 130 documents about the model itself, recent events and AI research.

The key technique is a kind of **back-translation**. Normally you would ask an AI a question and keep its answer. Here the authors work backward from a real Wikipedia passage. They generate questions that the passage can answer, then write answers and reasoning that rely on that passage. Because every answer is anchored to a real source, the authors can control which facts the model learns and which languages it sees.

The surprising part is that Synth already contains questions, answers, instructions and reasoning. So the authors could skip the later training stages entirely. Their models get **no supervised fine-tuning and no reinforcement learning**: everything is learned in one single stage. That's where the paper's title, "It's All Training," comes from.

## How Synth is built

The pipeline runs in two steps.

**Step 1: train two helper models.**

- A *query model* writes questions.
- A *reasoning model* writes answers along with reasoning.

Both helpers learn from examples produced by a very strong frontier AI model. A third tool, an off-the-shelf search tool, finds passages with related meaning.

**Step 2: run the helpers at large scale over every seed passage.**

**Keeping the questions varied.** The authors found that asking one big AI to write questions, even with rotating instructions, kept producing the same narrow styles. Instead they fine-tuned a smaller model, Google's Gemma-3-12B, using a technique called LoRA. LoRA adapts an existing model to a new task while training only a small number of extra settings, which saves a lot of computing power.

Each time the query model writes a question, a simple dice-roll system picks requirements along six directions:

- the type of question,
- how hard it is,
- who is asking,
- what kind of answer it should get,
- which language it is in,
- what style it is written in.

This forces the questions to differ in planned ways rather than only in whatever ways come easiest to the model. Every seed passage is used 100 times. The most important articles are split into all their sections, while less important ones contribute only their opening summary.

**Teaching the model to say "I'm not sure."** One design choice stands out. About 20% of the questions are deliberately tricky: they are false, absurd or ambiguous, so the right response is to refuse, correct the question or hedge. Without these, every practice question would have a confident answer, and the model would learn to always answer, even when it shouldn't.

**Reasoning in shorthand.** The reasoning model sees three things:

1. the question,
2. its seed passage,
3. a related passage found by the search tool.

Pairing passages this way builds bridges between topics that no single passage would create.

The reasoning itself is written in a compact, shorthand-like ("stenographic") style using special symbols. There are symbols for logical steps (like "therefore"), symbols for how certain a claim is (from "certain" to "uncertain"), and symbols for checking work. These symbols were added to the tokenizer so that each reasoning step costs only one or two tokens. That lets a small model fit richer reasoning into its limited working space. The authors also added markers at decision points that estimate how uncertain the next choice is. These markers are only labels in the training data and don't control the model's behavior when it is used.

**Other kinds of practice.** Around the core memorization task, Synth adds:

- **Answering with sources:** answering from up to ten retrieved passages while citing them. This is called retrieval-augmented generation, or RAG. It teaches the model to switch between answering from memory and answering from documents it is given.
- **Math problems:** about 3,000 problem templates, each turned into many problems by changing the numbers and re-solving them.
- **Creative writing:** writing under constraints, such as lipograms, which are texts that avoid a particular letter.
- **Editing:** translation, pulling out information, fixing spelling or facts, and rewriting in a new style.
- **Multiple-choice questions.**
- **Practical knowledge:** cooking recipes from Wikibooks.

**Languages and quality.** About 20% of Synth is in languages other than English, mainly a few European languages. The reasoning stays in English even when the question isn't.

An outside data-quality judge called Propella-1 compared Synth with several well-known open training collections. Synth came out on top for quality and value, with the biggest leads in reasoning and educational value. It tied a cleaned Wikipedia collection, FineWiki, for safety. The authors point out that Synth's reasoning score is partly boosted by its built-in reasoning, but its educational-value lead doesn't depend on that. Wikipedia-style data matched Synth on safety but lost on reasoning, while cleaned web data showed the opposite pattern.

## The models they trained

The size of a model is measured in **parameters**, the internal settings it adjusts while learning. More parameters generally means a more capable model. The authors trained four models:

- **Monad**, a tiny 56-million-parameter model, built as a stress test of how far a very small model can be pushed.
- **Baguettotron-350M**, with an unusually deep design, testing the guess that deeper models benefit more from reasoning data.
- **Baguettotron-600M**, a more conventional design and the main reference model.
- **Baguettotron-MoE**, a "Mixture-of-Experts" model. It contains 13.2 billion parameters in total, split among 16 "expert" sections, but only about 1 billion are used for each piece of text. A small routing system picks one expert each time, so the model has lots of stored capacity while staying cheap to run.

The MoE model reached the same training score as the 600M model while seeing only about 32% as many tokens (around 50 billion).

**Trained far longer than the usual rule suggests.** A well-known earlier study, nicknamed "Chinchilla," suggested a model should see about 20 tokens for every parameter. For a 600M model that's about 12 billion tokens. The authors trained theirs on 264 tokens per parameter, far more than that rule suggests. The model was still improving at the end, with no sudden problems during training.

An early comparison also hinted that models trained on Synth can reach a better final level than models trained on the scraped collections FineWiki and FinePDFs-Edu. The authors call this preliminary.

## How well do the models do?

The authors compared their models with four similar-sized open models from other groups: Gemma-3-270M, SmolLM2-360M, LFM2.5-350M and Qwen3-0.6B. The tests included 22 multiple-choice tasks and 8 open-ended tasks. Because these other models differ in design, data and later training, the authors present them as context rather than a perfectly fair contest.

The main point is **efficiency**. The best Baguettotron models trail Qwen3-0.6B, the strongest comparison model, by 4.7 points on multiple-choice and only 0.7 points on open-ended tasks. Yet they were trained on 80 to 700 times fewer tokens.

The tiny Monad, despite having only 56 million parameters, beat Gemma-3-270M and SmolLM2-360M on average in multiple-choice.

Results varied a lot from test to test:

- On a nuclear-science quiz, Baguettotron-600M scored 45%, compared with 53% for Qwen3-0.6B.
- The authors' models tied or beat Qwen on tests of truthfulness and some specialist subjects.
- They lagged clearly on tests that reward broad web knowledge. The authors attribute this to their limited set of seed sources.

## Is it really the data? A fair head-to-head

To check that the gains come from the data, and not from the model's design or training length, the authors trained two more 600M models that were identical to Baguettotron-600M except for what they read:

- one trained on English Wikipedia text (FineWiki), read about 17 times over;
- one trained on educational PDF documents (FinePDFs-Edu).

Without extra training, neither web-trained model could even produce properly formatted multiple-choice answers. So the authors gave both a short extra round of training on conversations and practice quiz questions. This added only about 0.2% to their computing cost.

Even after that extra help, the Synth model led by 16 to 17 points on multiple-choice and 11 to 14 points on open-ended tasks. Both web models stayed at the level of random guessing on MMLU, a well-known general-knowledge quiz. The authors conclude that the advantage isn't only about answer format.

**Does the reasoning matter?** The authors also retrained the 600M model on Synth with the reasoning removed, keeping everything else the same.

- Multiple-choice accuracy barely moved: 41.8% without reasoning versus 42.2% with it.
- Open-ended accuracy dropped from 24.3% to 22.0%.
- The losses were concentrated in truthfulness (down 9.1 points) and nuclear-science reasoning (down 10.0 points).
- Fact recall was essentially unchanged (down 0.5 points on average).
- One test about handling conflicting information actually improved, by 3.9 points.

## Do the models make up fewer facts?

AI models sometimes produce confident-sounding statements that are false. This is often called "hallucination." The psychology term **confabulation** fits even better: producing made-up or distorted memories without meaning to deceive.

Because every Synth answer is anchored to a real Wikipedia passage, the authors expected their models to make up fewer facts. They contrast this with other synthetic-data methods that can introduce claims the source never made, which then get passed on to the student model. (The authors also note that copying text word for word is off-limits for copyright reasons.)

**How they measured it.** For 500 randomly chosen seed topics, they asked each model, "What do you know about [topic]?" Another AI, DeepSeek-V3.2, broke each answer into small single facts. Each fact was then labeled one of three ways:

- **supported** by the Wikipedia article,
- **contradicted** by it,
- **inconclusive**, meaning it couldn't be checked.

They used two scores:

- **Precision:** of the facts that could be checked, what share were correct.
- **Macro score:** correct facts as a share of *all* facts, so unverifiable statements count against the model.

The table also gives a margin of uncertainty for each score, and the authors used a statistical check, called a paired bootstrap, to test whether each other model did significantly worse than theirs.

**What they found.** Baguettotron-600M and Baguettotron-MoE got the highest macro scores of any model tested: 41.7% and 46.3%. They beat Phi-4-mini-instruct, a model six times larger, and DeepSeek-MoE-16B-Chat, which uses 2.8 times as many active parameters and saw more than 40 times as much training text. Baguettotron-350M also beat LFM2.5-350M and SmolLM2-360M-IT despite seeing 10 to 140 times fewer tokens.

The only comparison model trained on a similar amount of text was OPT-350M. It was also the only model that stated more wrong facts than right ones. The authors' conclusion: their models are the most token-efficient fact learners in every size group, and the only ones above 40% on the macro score.

**Is the model just playing it safe?** One worry is that a model could score well by saying only vague, safe things. To test this, the authors used 600 Wikipedia "Good Articles" that were *not* in the seed set. A model can't report facts it never saw; as the authors put it, learning about elephants doesn't teach you about giraffes. The test question, "What do you know about...", also never appears among the 5,000 questions in the released sample of Synth.

On these unfamiliar topics:

- Baguettotron-MoE declined to answer 67% of the time, compared with 20% on familiar topics.
- When it did answer, its precision fell from 82% to 62%.

So the score really does track what the model knows. A web-trained MoE of the same active size, OLMoE, declined only 7% of the time and kept 81% precision. The authors say this is expected, since web data covers many of these topics.

## Can the model tell when it's unsure?

In measurement science, **calibrating** an instrument means checking its readings against a trusted standard. A calibrated AI is one whose confidence matches how often it is actually right. **Epistemic** means "about knowledge," so epistemic markers are the symbols that flag how sure the model is.

The authors sorted each model's reasoning into "confident" or "uncertain," depending on which kind of marker appeared more often. Then they compared the answers in the two groups.

**Accuracy.** For three of the four models, answers with uncertain-dominated reasoning were less factually accurate:

- Baguettotron-350M dropped from 0.35 to 0.25.
- A 600M model dropped from 0.44 to 0.40.
- The MoE dropped from 0.50 to 0.41.

The larger models showed the clearest difference. For tiny Monad the markers carried no signal: it stayed at 0.17 either way.

**How much the model says.** All four models stated fewer facts when uncertain. The bigger ones gave about 15 to 20% fewer facts, and even Monad went from 11.7 to 10.6 facts per topic.

The authors suggest that learning to *say less* when unsure may develop at smaller sizes than learning to *be more right*. For models above roughly 300M parameters, they conclude, the markers work as a usable confidence signal for both. (A small note: the fact-splitting AI sometimes created "facts" about the speaker from refusal sentences, about 6% of all facts. This slightly shrinks the differences reported here.)

## Teaching new specialties

The same method can adapt a model to a specialist field. The authors chose telecommunications, a field with scarce data and dense jargon. As seeds they used telecom articles from Wikipedia plus technical standards from 3GPP, the body that writes mobile-network standards. Together that was about 93 million tokens, which they amplified into about 460 million synthetic tokens.

After extra training on this material, the 600M model improved clearly:

- On a telecom quiz (TeleQnA), accuracy rose from 41.6% to 56.7%.
- Its fact-accuracy score on the 3GPP standards rose from 21.5% to 38.8%.

They also asked Claude Opus 4.5 to grade 50 telecom answers from 0 to 100. The specially trained model scored 67.8, compared with 58.7 for a version trained only on the Wikipedia telecom material and 27.8 for the original model. The gains came mostly from precise knowledge, such as 5G, and from **disambiguation**, working out the right meaning of acronyms. The original model tended to make up plausible but wrong meanings.

**Using tools.** The authors also taught the models "tool calling": producing correctly formatted requests to outside software tools. They used the same two-helper approach to build 532,000 practice examples, then continued training with no reinforcement learning. On a standard tool-calling test, Baguettotron-600M scored 53.1%, 3.9 points above Google's FunctionGemma-270M (7.8 points on a second, averaged accuracy measure), on about 40 times fewer training tokens. The original model made no tool calls at all, so this skill came from the new data. The authors suggest this data could be folded into the main training mix in future.

## What it means

The authors argue that their results change how we should read earlier rules about training size. The Chinchilla rule was worked out using web text, which they describe as poorly matched to what a small model needs to learn. With carefully engineered data, training kept paying off well beyond 20 tokens per parameter. They see this as confirming an idea from the Phi models: data quality and how learnable the data is can be controlled, alongside model size and amount of text, and they believe AI research has under-explored it.

They also stress openness. A capable training setup can be built from about 50,000 Wikipedia articles, which can be shared freely.

## Limits the authors acknowledge

- **Limited knowledge.** The models can only know what is in their roughly 58,000 seed articles. That gives tight control but sets a hard ceiling. Other language versions of Wikipedia could expand it.
- **Cultural slant.** The seeds reflect what English-speaking Wikipedia contributors consider important, even though the data covers several languages. Future work should draw on content written natively in other languages.
- **Difficulty.** The practice tasks suit small models. Larger models would need new pipelines covering coding, multi-step tool use and long tasks.
- **No web data at all.** Leaving out web text isolates Synth's effect, but it means every behavior, including refusing, has to be engineered by hand. Some of those behaviors might come naturally from ordinary text at large scale.