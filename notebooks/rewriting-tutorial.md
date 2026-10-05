---
rat:
  project: ..
  python:
    requires: ">=3.12"
    dependencies:
      - "-e ../../functai/python"
      - "-e ../../lmcc/python"
      - "pandas>=2,<3"
      - "pydantic>=2,<3"
      - "tiktoken==0.12.0"
---

# Can an AI explain a science paper to a 13-year-old?

*A hands-on walkthrough of our first experiment: rewriting a real paper, then checking the rewrites.*

We took one real scientific paper and asked two AI models to rewrite it so a curious 12–14-year-old could understand it — **without getting the science wrong**. Then we asked the same models to grade all the rewrites, including the one we wrote together in conversation.

By the end of this notebook you will know:

1. what the paper and the task look like,
2. how a rewrite is built in five small steps instead of one big one,
3. how the two models' rewrites compare, side by side,
4. whether they caught the mistakes hidden in the original paper,
5. what the AI judges said — and **how far we can trust them**.

**Nothing here costs anything.** Every result was produced once and saved. Each cell below just reads those files. One optional cell (chapter 3) lets you make a single live call if you want to see it happen.

---

## 1. Setup

To keep the cells below short, the repetitive code lives in a small helper file, **`tutorial_kit.py`** (in `rewrite_benchmark/`). It does two jobs:

- **Loading:** it reads the results we saved earlier — the paper, the three rewrites, the log of every model call, and the judges' scores. No model is called.
- **Showing:** it draws the tables, bar charts, colour-shaded tables and side-by-side text you will see, as HTML.

In the notebook we call it `tk`. First, tell Python where to find it:

```python
import sys
sys.path.insert(0, "rewrite_benchmark")  # finds tutorial_kit.py
```

Now load everything we saved:

```python
import pandas as pd
import tutorial_kit as tk

paper = tk.load_paper()
rewrites = tk.load_candidates()
calls = tk.load_calls()
judgments, issues, raw_judgments = tk.load_judgments()

print("Paper:", paper["title"])
print("Rewrites loaded:", [tk.NAMES[k] for k in rewrites])
print("Saved model calls:", len(calls))
```

```output
Paper: Recommendations for increasing yield of the edible Pinus pinea L. pine nuts
Rewrites loaded: ['Conversation', 'Astra', 'Luna']
Saved model calls: 166
```

We have three rewrites to compare:

| Name | Who wrote it | How |
|---|---|---|
| **Conversation** | GPT-6 Astra, with us | In a long chat, section by section, with tools and our feedback |
| **Astra** | GPT-6 Astra | Five automatic steps, no human help |
| **Luna** | GPT-6 Luna | The same five automatic steps |

---

## 2. Meet the paper

The paper is a Chilean study of **stone pine cones**: do heavier cones give more edible pine nuts? ([Loewe-Muñoz et al., PLOS ONE 2024](https://doi.org/10.1371/journal.pone.0300008), CC BY 4.0.)

How long is each part? We count **tokens**, the word-pieces an AI model reads.

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")   # the tokenizer used by GPT-4o

def count_tokens(text):
    return len(enc.encode(text))

# Tokens in each section
sizes = {}
for section in tk.ORDER:
    sizes[section] = count_tokens(paper["sections"][section])

# Tokens in the whole paper, references included
total = count_tokens(paper["source_text"] + paper["references"])

tk.show(tk.bars(sizes, "Tokens in each section of the original paper", unit=" tokens",
                note=f"Whole paper, with references: {total:,} tokens."))
```

<iframe class="rat-output" src="../_assets/generated/26feee4c7980.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

**What we learn:** the methods, results and discussion are the heavy parts. They are also where the numbers live — and numbers are where rewrites go wrong.

Let's read one. Change `SECTION` to any name above and rerun.

```python
SECTION = "results"   # try "abstract", "methods" or "discussion"

text = paper["sections"][SECTION]
tk.show(tk.text_box(f"Original · {SECTION}", text))
```

<iframe class="rat-output" src="../_assets/generated/bb6978af5af2.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

This is hard reading even for adults: "PY", "tertiles", "regression tree", percentages of percentages. That is exactly the kind of text we want to open up.

---

## 3. The task, written as a function

With [FunctAI](https://github.com/MaximeRivest/functai), a prompt is just a typed Python function. The **docstring** is the instruction, the **arguments** are what the model reads, and the **return type** is the shape of the answer.

Here is a small version of what the pipeline does:

```python
from pydantic import BaseModel
from functai import ai

class Rewrite(BaseModel):
    text: str                   # the section, rewritten for the reader
    editorial_notes: list[str]  # problems noticed in the original paper

@ai
def explain_section(section: str, whole_paper: str) -> Rewrite:
    """Rewrite this section of a scientific paper for a curious 12-14-year-old.
    Keep every number, comparison and uncertainty correct. Explain new words.
    Do not summarize. If the paper contradicts itself, say so in editorial_notes."""
    ...

# The same function, set to use GPT-6 Astra.
explain_with_astra = explain_section.using(lm="openai-codex:gpt-6-astra")

# Build the request for the abstract, but do NOT send it.
request = explain_with_astra.render(section=paper["sections"]["abstract"],
                                    whole_paper=paper["source_text"])

size = count_tokens(str(request.system) + str(request.messages))
print("The model would receive", size, "tokens.")
print("Nothing was sent: render() only builds the request.")
```

```output
The model would receive 5615 tokens.
Nothing was sent: render() only builds the request.
```

**Try it live (optional).** Running this cell makes one real model call. It rewrites the abstract only and takes roughly one to five minutes. Skip it if you don't want to wait.

`.stream(...)` makes the same call as `explain_with_astra(...)`, but shows the answer as it is being written, instead of waiting in silence.

```python
abstract = paper["sections"]["abstract"]
stream = explain_with_astra.stream(section=abstract, whole_paper=paper["source_text"])
stream.show()   # prints the rewrite as it arrives
```

```output
{"text":"Pinus pinea, also called stone pine, produces edible pine nuts. An important feature of this crop is its cone-to-pine-nut yield: the total weight of edible pine nuts in a cone, expressed as a percentage of the cone’s weight. The researchers call this PY. This yield is decreasing worldwide, which is worrying because pine nuts from this species are in high demand.\n\nThe researchers followed cone weight, the size and shape of seeds and edible pine nuts, and pine nut yield for 10 years in Chile, where stone pine does not grow naturally. Studying size and shape is called morphometry. Here, “seeds” means pine nuts still inside their shells, while “pine nuts” means the edible kernels after the shells have been removed.\n\nThe study covered seven plantations—areas of planted trees—in different growing environments. The researchers counted, measured and weighed 560 cones and the seeds and pine nuts they contained. They also checked seeds and pine nuts for damage.\n\nThe researchers defined two contrasting groups based on cone weight: heavy cones and light cones. They calculated PY and other features and compared the groups using a mixed linear model. This is a statistical method that can compare groups while accounting for differences linked to factors such as plantation and year. They also used regression trees, a method that repeatedly divides data into groups, to help explain why PY varied.\n\nThe cones in Chile were heavier, on average, than cones from the species’ native range—the places where it grows naturally: 474 g compared with 300 g. The section reports that heavy cones had more pine nuts per cone than light cones, 125 compared with 89, and a higher PY, 4.05% compared with 3.62%. Both differences were statistically significant, meaning the statistical test found evidence of a difference rather than just random variation. However, the paper elsewhere identifies 125 and 89 as the numbers of seeds still in their shells, not edible pine nuts; see the editorial notes.\n\nHeavy cones also had a lower percentage of damaged seeds than light cones: 9.0% compared with 15.9%. The researchers therefore concluded that PY depended on the size and shape of seeds and pine nuts, as well as on seed health. They suggest that fertilization—supplying nutrients—and irrigation—supplying water—could help trees produce more heavy cones and, as a result, increase PY.","editorial_notes":["The section labels 125 versus 89 as pine nuts per cone. The Results and Table 4 identify these as rounded counts of in-shell seeds (124.8 versus 89.1). The reported counts of healthy edible pine nuts are instead 112.9 versus 76.1 for heavy and light cones, respectively.","The section gives an average cone weight of 474 g, consistent with rounding Table 4’s 473.6 g. The Results and Discussion instead give 470 g, apparently using coarser rounding.","The proposed benefits of fertilization and irrigation are possibilities, not effects established by this study’s heavy-versus-light cone comparison. The Discussion explicitly says the observed differences between sites are insufficient to attribute them to irrigation."]}
```

When it finishes, `stream.result` is the finished answer. Show it next to the original:

```python
result = stream.result
tk.show(tk.side_by_side([("Original abstract", abstract),
                         ("Live rewrite", result.text)]),
        tk.bullet_list("Editorial notes", result.editorial_notes))
```

<iframe class="rat-output" src="../_assets/generated/7cb812a9b84f.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

---

## 4. Why five steps instead of one?

A whole paper in one call asks a lot of a model. So we copied what we did in conversation, and split the job into **five calls**:

```
Step 1  opening        title, abstract, first paragraph, conclusion
Step 2  introduction   the rest of the introduction
Step 3  methods
Step 4  results
Step 5  discussion
```

Each step receives **the full original paper** (the source of truth) plus **everything rewritten so far** and a growing **glossary**, so the explanations stay consistent from one section to the next.

That handoff has a cost: each step reads more than the last. Let's see it.

```python
# Keep only the writing calls (the rest are judging calls).
writing = calls[calls["kind"] == "write"]

# One row per step, one column per model. Each cell: tokens that step read.
per_step = writing.pivot(index="step", columns="model", values="input_tokens")
per_step = per_step.reindex(tk.STEPS)   # put the steps in the order they ran

tk.show(tk.table(per_step, "Tokens read by each writing step",
                 note="Each step reads the whole paper plus all earlier rewrites, so the input grows step by step."))
```

<iframe class="rat-output" src="../_assets/generated/23a4bc3a870a.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

And how long did each writer take?

```python
# Add up the five steps for each model.
minutes = writing.groupby("model")["minutes"].sum()

tk.show(tk.bars(minutes.to_dict(), "Total writing time for the whole paper", unit=" min", fmt="{:.0f}",
                colors=tk.MODEL_COLORS,
                note="Both used maximum reasoning effort. Five calls each, run one after another."))
```

<iframe class="rat-output" src="../_assets/generated/911e94d16149.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

**What we learn:** Astra took about 38 minutes, Luna about 16 — more than twice as long. Keep that in mind: if Astra's rewrite turns out better, part of the price is time.

---

## 5. Read the rewrites side by side

This is the heart of it. Pick a section and compare the original with all three rewrites. Try `"methods"` — the hardest one to explain well.

```python
SECTION = "methods"

# One column for the original, then one per rewrite.
columns = [("Original", paper["sections"][SECTION])]
for name, rewrite in rewrites.items():
    columns.append((tk.NAMES[name], rewrite["segments"][SECTION]))

tk.show(tk.side_by_side(columns))
```

Things worth looking for while you read:

- Is every **number** still there, with its unit?
- Does "11.9% higher" stay a *relative* increase, not 11.9 percentage points?
- Are new words (like *standard error* or *regression tree*) explained **before** they are used?
- Does it still sound like a scientist talking, rather than a children's book?

A quick numeric view: how long is each rewrite compared with the original?

```python
# One row per section; one column for the original and one per rewrite.
length = pd.DataFrame(index=pd.Index(tk.ORDER, name="section"))
length["Original"] = [count_tokens(paper["sections"][s]) for s in tk.ORDER]
for name, rewrite in rewrites.items():
    length[tk.NAMES[name]] = [count_tokens(rewrite["segments"][s]) for s in tk.ORDER]

tk.show(tk.table(length, "Length of each section, in tokens",
                 note="Longer is not worse: explaining a term takes words. Much shorter can mean something was dropped."))
```

**What we learn:** the two automatic rewrites are about a quarter **longer** than the original, while our conversation rewrite is about a fifth **shorter**. The biggest growth is in the abstract and conclusion, which Astra more than doubled. Whether that extra length helps or just pads is a question for the judges — and for you.

---

## 6. Did they catch the paper's own mistakes?

The original paper contains a few real errors. A good translator should notice them instead of copying them. We know of four:

1. The abstract calls 125 vs 89 "pine nuts", but the table shows those are **seeds still in their shells**.
2. The formula for damaged seeds is **printed wrongly**.
3. The results text **swaps two numbers** (2.60% and 4.04%) compared with Figure 1.
4. Figure 1 uses **328 cones**, not the 560 collected overall.

**An important catch:** both automatic writers were *given* a notes file listing these errors, and our conversation had discovered them along the way. So this is not a test of spotting errors from scratch. It tests something narrower: **did the writer actually use the evidence it was handed, and say so?**

Each writer lists what it noticed in its **editorial notes**. Let's check which traps each one mentions.

```python
caught = pd.DataFrame(index=pd.Index(list(tk.TRAPS), name="known error"))
for name, rewrite in rewrites.items():
    found = tk.trap_check(rewrite["notes"])   # {error: True or False}
    caught[tk.NAMES[name]] = ["✓ mentioned" if found[error] else "—" for error in tk.TRAPS]

tk.show(tk.table(caught,
                 "Known errors in the paper, and whether each writer flagged them",
                 note="A keyword check on the notes: it shows a mention, not that the fix was right. Read the notes to confirm."))
```

And here are the notes themselves, so you can confirm:

```python
WRITER = "astra"   # or "luna", "conversation_reference"

notes = rewrites[WRITER]["notes"]
tk.show(tk.bullet_list(f"{tk.NAMES[WRITER]} · {len(notes)} editorial notes", notes, numbered=True))
```

**What we learn:** Astra's notes mention all four errors. Luna's miss the abstract's seeds-versus-nuts mix-up, and our conversation never mentioned the 328 cones. A rewrite can read beautifully and still repeat a wrong number from the paper, so this matters. A fairer future test: give the writers **no** error list and see which mistakes they find on their own.

---

## 7. How the rewrites were graded

Reading three rewrites of a whole paper takes an hour. To scale up, we need **AI judges**. Each judge received the original, one anonymous rewrite, and a rubric, then gave a score from **0 to 3** on five aspects:

| Aspect | The question |
|---|---|
| Scientific faithfulness | Are the facts, numbers and uncertainty still correct? |
| Completeness | Did anything important disappear? |
| Accessibility | Could a curious 13-year-old follow it? |
| Document coherence | Does it read as one consistent explanation? |
| Editorial integrity | Were the paper's own errors handled honestly? |

The first three were scored **section by section**, the last two on the **whole document**. Both Astra and Luna judged all three rewrites — including their own.

Every judge also had to **quote** the exact text behind each complaint. If a quote did not exist in the paper or the rewrite, we threw that judgment away as unreliable.

Here is one real judgment, so you can see what a judge produces:

```python
# Valid judgments of Luna's rewrite, on accessibility, that list at least one problem.
examples = [j for j in raw_judgments
            if j["status"] == "ok"
            and j["candidate"] == "luna"
            and j["aspect"] == "accessibility"
            and j["assessment"]["issues"]]

tk.show(tk.judgment_card(examples[0]))
```

---

## 8. The scores — and why they are not enough

First, the headline number people always want: the average score.

```python
valid = judgments[judgments["status"] == "ok"]        # drop the judgments we threw away

average = valid.groupby("candidate")["score"].mean()   # one average per rewrite
average = average.rename(index=tk.NAMES)                # "astra" -> "Astra"

tk.show(tk.bars(average.to_dict(), "Average score (0–3), all aspects and both judges",
                fmt="{:.2f}", colors=tk.NAME_COLORS))
```

They all look close to perfect. **Is that real?** Let's look at how the scores are spread out.

```python
# For each rewrite: what percentage of its judgments got a 0, a 1, a 2 or a 3?
spread = pd.crosstab(valid["candidate"], valid["score"], normalize="index") * 100
spread = spread.reindex(columns=[0, 1, 2, 3], fill_value=0)   # show every score, even unused ones
spread = spread.rename(index=tk.NAMES)

tk.show(tk.heat(spread, "Share of judgments at each score (%)", lo=0, hi=100, fmt="{:.0f}%",
                note="Rows add to 100%. Most judgments are a perfect 3."))
```

**What we learn:** 70–87% of all judgments are a perfect 3. The scale is too coarse: it cannot tell "good" from "excellent". **This is a finding about our rubric, not proof the rewrites are equally good.** We need a sharper measuring tool, and the next chapter shows where the real signal is hiding.

---

## 9. Where the real signal is: the problems judges found

Scores compress everything into one digit. The **issues** each judge listed are far more informative. Let's count them.

```python
# Count the problems: one row per rewrite, one column per severity.
counted = pd.crosstab(issues["candidate"], issues["severity"])
counted = counted.reindex(columns=["critical", "major", "minor"], fill_value=0)
counted = counted.rename(index=tk.NAMES)

tk.show(tk.table(counted, "Problems found in each rewrite (both judges together)"))
```

**What we learn:** no one made a critical error. But Astra's automatic rewrite drew **far fewer complaints** than the other two. This is the clearest difference in the whole experiment.

Which sections do the complaints point to?

```python
# Count the problems again: one row per section, one column per rewrite.
where = pd.crosstab(issues["section"], issues["candidate"])
where = where.reindex(index=tk.ORDER + ["whole_document"], columns=list(rewrites), fill_value=0)
where = where.rename(columns=tk.NAMES)

# lo and hi are swapped on purpose: more problems = redder.
most = where.values.max()
tk.show(tk.heat(where, "Number of problems per section", lo=most, hi=0, fmt="{:.0f}",
                note="Redder = more problems. These are the places to read yourself."))
```

Now read the serious ones. These are the most useful lines in this notebook:

```python
serious = issues[issues["severity"] != "minor"]        # major and critical only
serious = serious.sort_values(["candidate", "section"])

boxes = [tk.issue_box(row) for _, row in serious.iterrows()]
tk.show(tk.title(f"All {len(serious)} major problems, with the quotes behind them"), *boxes)
```

---

## 10. Can we trust the judges?

Two things could make the judges misleading.

**Self-preference.** Does each model grade its own writing more kindly?

```python
per_section = valid[valid["section"] != "whole_document"]   # section judgments only

# One row per judge, one column per rewrite. Each cell: the average score given.
fairness = per_section.pivot_table(index="judge", columns="candidate", values="score", aggfunc="mean")
fairness = fairness[list(rewrites)].rename(columns=tk.NAMES)

tk.show(tk.heat(fairness, "Average section score given, by judge (rows) and rewrite (columns)",
                lo=2.4, hi=3, fmt="{:.2f}",
                note="Read across a row. The Conversation rewrite was also written by Astra."))
```

**What we learn:** each judge gave its own model's rewrite the highest average. The effect is small, but real enough that we should never let a model be the only judge of itself.

**Reliability.** How often did a judge quote text that does not exist?

```python
# Count valid and thrown-away judgments for each judge.
reliability = pd.crosstab(judgments["judge"], judgments["status"])
total_per_judge = reliability.sum(axis=1)
reliability["share thrown away"] = reliability["invalid_judge_output"] / total_per_judge

tk.show(tk.table(reliability, "Valid and invalid judgments, by judge",
                 note="Invalid = quoted text not found in the paper or rewrite, or a score that contradicts its own issues. These are excluded, not counted as zero."))
```

**What we learn:** Luna was a much less reliable judge — about one judgment in four had to be thrown away, against one in twenty-five for Astra. Astra is the better grader here, which is awkward, because it is also one of the writers.

---

## 11. What we can honestly conclude

From **one paper**, **one run**, and **AI judges only**:

- **The five-step pipeline works.** Both models produced complete, readable rewrites of the whole paper with no critical errors.
- **Astra's automatic rewrite looks strongest** — mostly because judges found far fewer problems in it, and its notes used all four known errors it had been told about. It also took more than twice as long as Luna.
- **Our 0–3 scale is too blunt.** Almost everything scored 3, so average scores cannot rank the rewrites. The list of problems is the useful signal.
- **The judges are not neutral.** They slightly favour themselves, and Luna often quoted text that does not exist.

What this does **not** show: that one model is better on other papers, or that real 13-year-olds understand these rewrites. For that we need more papers and real readers.

---

## 12. What to do next

1. **Check the judges yourself.** Read the major problems in chapter 9 and mark each as real or not. Record your scores in `rewrite_benchmark/runs/pine-pilot-v1/replicate-0/human_review.csv`.
2. **Sharpen the measurement.** Replace the 0–3 score with a count of problems, or ask judges *which of two rewrites is better*. Side-by-side choices usually separate close candidates much better.
3. **Plant deliberate mistakes.** Change one number, remove one qualification. A good judge must catch each one. That tells us whether the judge can be trusted.
4. **Add more papers** from other fields before tuning anything, and keep some papers aside as a final test the pipeline never saw.
5. **Then** use the good rewrites to train a smaller, cheaper translator.

The full pipeline that produced these results — with every program, the rubric, and the controls for making new calls — is in [`scientific-rewriting.md`](scientific-rewriting.md). This notebook is the guided tour.
---
rat:
  project: ..
  python:
    requires: ">=3.12"
    dependencies:
      - "-e ../../functai/python"
      - "-e ../../lmcc/python"
      - "pandas>=2,<3"
      - "pydantic>=2,<3"
      - "tiktoken==0.12.0"
---

# Can an AI explain a science paper to a 13-year-old?

*A hands-on walkthrough of our first experiment: rewriting a real paper, then checking the rewrites.*

We took one real scientific paper and asked two AI models to rewrite it so a curious 12–14-year-old could understand it — **without getting the science wrong**. Then we asked the same models to grade all the rewrites, including the one we wrote together in conversation.

By the end of this notebook you will know:

1. what the paper and the task look like,
2. how a rewrite is built in five small steps instead of one big one,
3. how the two models' rewrites compare, side by side,
4. whether they caught the mistakes hidden in the original paper,
5. what the AI judges said — and **how far we can trust them**.

**Nothing here costs anything.** Every result was produced once and saved. Each cell below just reads those files. One optional cell (chapter 3) lets you make a single live call if you want to see it happen.

---

## 1. Setup

To keep the cells below short, the repetitive code lives in a small helper file, **`tutorial_kit.py`** (in `rewrite_benchmark/`). It does two jobs:

- **Loading:** it reads the results we saved earlier — the paper, the three rewrites, the log of every model call, and the judges' scores. No model is called.
- **Showing:** it draws the tables, bar charts, colour-shaded tables and side-by-side text you will see, as HTML.

In the notebook we call it `tk`. First, tell Python where to find it:

```python
import sys
sys.path.insert(0, "rewrite_benchmark")  # finds tutorial_kit.py
```

Now load everything we saved:

```python
import pandas as pd
import tutorial_kit as tk

paper = tk.load_paper()
rewrites = tk.load_candidates()
calls = tk.load_calls()
judgments, issues, raw_judgments = tk.load_judgments()

print("Paper:", paper["title"])
print("Rewrites loaded:", [tk.NAMES[k] for k in rewrites])
print("Saved model calls:", len(calls))
```

```output
Paper: Recommendations for increasing yield of the edible Pinus pinea L. pine nuts
Rewrites loaded: ['Conversation', 'Astra', 'Luna']
Saved model calls: 166
```

We have three rewrites to compare:

| Name | Who wrote it | How |
|---|---|---|
| **Conversation** | GPT-6 Astra, with us | In a long chat, section by section, with tools and our feedback |
| **Astra** | GPT-6 Astra | Five automatic steps, no human help |
| **Luna** | GPT-6 Luna | The same five automatic steps |

---

## 2. Meet the paper

The paper is a Chilean study of **stone pine cones**: do heavier cones give more edible pine nuts? ([Loewe-Muñoz et al., PLOS ONE 2024](https://doi.org/10.1371/journal.pone.0300008), CC BY 4.0.)

How long is each part? We count **tokens**, the word-pieces an AI model reads.

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")   # the tokenizer used by GPT-4o

def count_tokens(text):
    return len(enc.encode(text))

# Tokens in each section
sizes = {}
for section in tk.ORDER:
    sizes[section] = count_tokens(paper["sections"][section])

# Tokens in the whole paper, references included
total = count_tokens(paper["source_text"] + paper["references"])

tk.show(tk.bars(sizes, "Tokens in each section of the original paper", unit=" tokens",
                note=f"Whole paper, with references: {total:,} tokens."))
```

<iframe class="rat-output" src="../_assets/generated/26feee4c7980.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

**What we learn:** the methods, results and discussion are the heavy parts. They are also where the numbers live — and numbers are where rewrites go wrong.

Let's read one. Change `SECTION` to any name above and rerun.

```python
SECTION = "results"   # try "abstract", "methods" or "discussion"

text = paper["sections"][SECTION]
tk.show(tk.text_box(f"Original · {SECTION}", text))
```

<iframe class="rat-output" src="../_assets/generated/bb6978af5af2.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

This is hard reading even for adults: "PY", "tertiles", "regression tree", percentages of percentages. That is exactly the kind of text we want to open up.

---

## 3. The task, written as a function

With [FunctAI](https://github.com/MaximeRivest/functai), a prompt is just a typed Python function. The **docstring** is the instruction, the **arguments** are what the model reads, and the **return type** is the shape of the answer.

Here is a small version of what the pipeline does:

```python
from pydantic import BaseModel
from functai import ai

class Rewrite(BaseModel):
    text: str                   # the section, rewritten for the reader
    editorial_notes: list[str]  # problems noticed in the original paper

@ai
def explain_section(section: str, whole_paper: str) -> Rewrite:
    """Rewrite this section of a scientific paper for a curious 12-14-year-old.
    Keep every number, comparison and uncertainty correct. Explain new words.
    Do not summarize. If the paper contradicts itself, say so in editorial_notes."""
    ...

# The same function, set to use GPT-6 Astra.
explain_with_astra = explain_section.using(lm="openai-codex:gpt-6-astra")

# Build the request for the abstract, but do NOT send it.
request = explain_with_astra.render(section=paper["sections"]["abstract"],
                                    whole_paper=paper["source_text"])

size = count_tokens(str(request.system) + str(request.messages))
print("The model would receive", size, "tokens.")
print("Nothing was sent: render() only builds the request.")
```

```output
The model would receive 5615 tokens.
Nothing was sent: render() only builds the request.
```

**Try it live (optional, one call).** Set `LIVE = True`. It rewrites the abstract only, and takes roughly one to five minutes.

```python
LIVE = False   # set to True to make one real call

if LIVE:
    abstract = paper["sections"]["abstract"]
    result = explain_with_astra(section=abstract, whole_paper=paper["source_text"])

    tk.show(tk.side_by_side([("Original abstract", abstract),
                             ("Live rewrite", result.text)]),
            tk.bullet_list("Editorial notes", result.editorial_notes))
else:
    print("Skipped. Set LIVE = True to make one real call.")
```

---

## 4. Why five steps instead of one?

A whole paper in one call asks a lot of a model. So we copied what we did in conversation, and split the job into **five calls**:

```
Step 1  opening        title, abstract, first paragraph, conclusion
Step 2  introduction   the rest of the introduction
Step 3  methods
Step 4  results
Step 5  discussion
```

Each step receives **the full original paper** (the source of truth) plus **everything rewritten so far** and a growing **glossary**, so the explanations stay consistent from one section to the next.

That handoff has a cost: each step reads more than the last. Let's see it.

```python
# Keep only the writing calls (the rest are judging calls).
writing = calls[calls["kind"] == "write"]

# One row per step, one column per model. Each cell: tokens that step read.
per_step = writing.pivot(index="step", columns="model", values="input_tokens")
per_step = per_step.reindex(tk.STEPS)   # put the steps in the order they ran

tk.show(tk.table(per_step, "Tokens read by each writing step",
                 note="Each step reads the whole paper plus all earlier rewrites, so the input grows step by step."))
```

And how long did each writer take?

```python
# Add up the five steps for each model.
minutes = writing.groupby("model")["minutes"].sum()

tk.show(tk.bars(minutes.to_dict(), "Total writing time for the whole paper", unit=" min", fmt="{:.0f}",
                colors=tk.MODEL_COLORS,
                note="Both used maximum reasoning effort. Five calls each, run one after another."))
```

**What we learn:** Astra took about 38 minutes, Luna about 16 — more than twice as long. Keep that in mind: if Astra's rewrite turns out better, part of the price is time.

---

## 5. Read the rewrites side by side

This is the heart of it. Pick a section and compare the original with all three rewrites. Try `"methods"` — the hardest one to explain well.

```python
SECTION = "methods"

# One column for the original, then one per rewrite.
columns = [("Original", paper["sections"][SECTION])]
for name, rewrite in rewrites.items():
    columns.append((tk.NAMES[name], rewrite["segments"][SECTION]))

tk.show(tk.side_by_side(columns))
```

Things worth looking for while you read:

- Is every **number** still there, with its unit?
- Does "11.9% higher" stay a *relative* increase, not 11.9 percentage points?
- Are new words (like *standard error* or *regression tree*) explained **before** they are used?
- Does it still sound like a scientist talking, rather than a children's book?

A quick numeric view: how long is each rewrite compared with the original?

```python
# One row per section; one column for the original and one per rewrite.
length = pd.DataFrame(index=pd.Index(tk.ORDER, name="section"))
length["Original"] = [count_tokens(paper["sections"][s]) for s in tk.ORDER]
for name, rewrite in rewrites.items():
    length[tk.NAMES[name]] = [count_tokens(rewrite["segments"][s]) for s in tk.ORDER]

tk.show(tk.table(length, "Length of each section, in tokens",
                 note="Longer is not worse: explaining a term takes words. Much shorter can mean something was dropped."))
```

**What we learn:** the two automatic rewrites are about a quarter **longer** than the original, while our conversation rewrite is about a fifth **shorter**. The biggest growth is in the abstract and conclusion, which Astra more than doubled. Whether that extra length helps or just pads is a question for the judges — and for you.

---

## 6. Did they catch the paper's own mistakes?

The original paper contains a few real errors. A good translator should notice them instead of copying them. We know of four:

1. The abstract calls 125 vs 89 "pine nuts", but the table shows those are **seeds still in their shells**.
2. The formula for damaged seeds is **printed wrongly**.
3. The results text **swaps two numbers** (2.60% and 4.04%) compared with Figure 1.
4. Figure 1 uses **328 cones**, not the 560 collected overall.

**An important catch:** both automatic writers were *given* a notes file listing these errors, and our conversation had discovered them along the way. So this is not a test of spotting errors from scratch. It tests something narrower: **did the writer actually use the evidence it was handed, and say so?**

Each writer lists what it noticed in its **editorial notes**. Let's check which traps each one mentions.

```python
caught = pd.DataFrame(index=pd.Index(list(tk.TRAPS), name="known error"))
for name, rewrite in rewrites.items():
    found = tk.trap_check(rewrite["notes"])   # {error: True or False}
    caught[tk.NAMES[name]] = ["✓ mentioned" if found[error] else "—" for error in tk.TRAPS]

tk.show(tk.table(caught,
                 "Known errors in the paper, and whether each writer flagged them",
                 note="A keyword check on the notes: it shows a mention, not that the fix was right. Read the notes to confirm."))
```

And here are the notes themselves, so you can confirm:

```python
WRITER = "astra"   # or "luna", "conversation_reference"

notes = rewrites[WRITER]["notes"]
tk.show(tk.bullet_list(f"{tk.NAMES[WRITER]} · {len(notes)} editorial notes", notes, numbered=True))
```

**What we learn:** Astra's notes mention all four errors. Luna's miss the abstract's seeds-versus-nuts mix-up, and our conversation never mentioned the 328 cones. A rewrite can read beautifully and still repeat a wrong number from the paper, so this matters. A fairer future test: give the writers **no** error list and see which mistakes they find on their own.

---

## 7. How the rewrites were graded

Reading three rewrites of a whole paper takes an hour. To scale up, we need **AI judges**. Each judge received the original, one anonymous rewrite, and a rubric, then gave a score from **0 to 3** on five aspects:

| Aspect | The question |
|---|---|
| Scientific faithfulness | Are the facts, numbers and uncertainty still correct? |
| Completeness | Did anything important disappear? |
| Accessibility | Could a curious 13-year-old follow it? |
| Document coherence | Does it read as one consistent explanation? |
| Editorial integrity | Were the paper's own errors handled honestly? |

The first three were scored **section by section**, the last two on the **whole document**. Both Astra and Luna judged all three rewrites — including their own.

Every judge also had to **quote** the exact text behind each complaint. If a quote did not exist in the paper or the rewrite, we threw that judgment away as unreliable.

Here is one real judgment, so you can see what a judge produces:

```python
# Valid judgments of Luna's rewrite, on accessibility, that list at least one problem.
examples = [j for j in raw_judgments
            if j["status"] == "ok"
            and j["candidate"] == "luna"
            and j["aspect"] == "accessibility"
            and j["assessment"]["issues"]]

tk.show(tk.judgment_card(examples[0]))
```

---

## 8. The scores — and why they are not enough

First, the headline number people always want: the average score.

```python
valid = judgments[judgments["status"] == "ok"]        # drop the judgments we threw away

average = valid.groupby("candidate")["score"].mean()   # one average per rewrite
average = average.rename(index=tk.NAMES)                # "astra" -> "Astra"

tk.show(tk.bars(average.to_dict(), "Average score (0–3), all aspects and both judges",
                fmt="{:.2f}", colors=tk.NAME_COLORS))
```

They all look close to perfect. **Is that real?** Let's look at how the scores are spread out.

```python
# For each rewrite: what percentage of its judgments got a 0, a 1, a 2 or a 3?
spread = pd.crosstab(valid["candidate"], valid["score"], normalize="index") * 100
spread = spread.reindex(columns=[0, 1, 2, 3], fill_value=0)   # show every score, even unused ones
spread = spread.rename(index=tk.NAMES)

tk.show(tk.heat(spread, "Share of judgments at each score (%)", lo=0, hi=100, fmt="{:.0f}%",
                note="Rows add to 100%. Most judgments are a perfect 3."))
```

**What we learn:** 70–87% of all judgments are a perfect 3. The scale is too coarse: it cannot tell "good" from "excellent". **This is a finding about our rubric, not proof the rewrites are equally good.** We need a sharper measuring tool, and the next chapter shows where the real signal is hiding.

---

## 9. Where the real signal is: the problems judges found

Scores compress everything into one digit. The **issues** each judge listed are far more informative. Let's count them.

```python
# Count the problems: one row per rewrite, one column per severity.
counted = pd.crosstab(issues["candidate"], issues["severity"])
counted = counted.reindex(columns=["critical", "major", "minor"], fill_value=0)
counted = counted.rename(index=tk.NAMES)

tk.show(tk.table(counted, "Problems found in each rewrite (both judges together)"))
```

**What we learn:** no one made a critical error. But Astra's automatic rewrite drew **far fewer complaints** than the other two. This is the clearest difference in the whole experiment.

Which sections do the complaints point to?

```python
# Count the problems again: one row per section, one column per rewrite.
where = pd.crosstab(issues["section"], issues["candidate"])
where = where.reindex(index=tk.ORDER + ["whole_document"], columns=list(rewrites), fill_value=0)
where = where.rename(columns=tk.NAMES)

# lo and hi are swapped on purpose: more problems = redder.
most = where.values.max()
tk.show(tk.heat(where, "Number of problems per section", lo=most, hi=0, fmt="{:.0f}",
                note="Redder = more problems. These are the places to read yourself."))
```

Now read the serious ones. These are the most useful lines in this notebook:

```python
serious = issues[issues["severity"] != "minor"]        # major and critical only
serious = serious.sort_values(["candidate", "section"])

boxes = [tk.issue_box(row) for _, row in serious.iterrows()]
tk.show(tk.title(f"All {len(serious)} major problems, with the quotes behind them"), *boxes)
```

---

## 10. Can we trust the judges?

Two things could make the judges misleading.

**Self-preference.** Does each model grade its own writing more kindly?

```python
per_section = valid[valid["section"] != "whole_document"]   # section judgments only

# One row per judge, one column per rewrite. Each cell: the average score given.
fairness = per_section.pivot_table(index="judge", columns="candidate", values="score", aggfunc="mean")
fairness = fairness[list(rewrites)].rename(columns=tk.NAMES)

tk.show(tk.heat(fairness, "Average section score given, by judge (rows) and rewrite (columns)",
                lo=2.4, hi=3, fmt="{:.2f}",
                note="Read across a row. The Conversation rewrite was also written by Astra."))
```

**What we learn:** each judge gave its own model's rewrite the highest average. The effect is small, but real enough that we should never let a model be the only judge of itself.

**Reliability.** How often did a judge quote text that does not exist?

```python
# Count valid and thrown-away judgments for each judge.
reliability = pd.crosstab(judgments["judge"], judgments["status"])
total_per_judge = reliability.sum(axis=1)
reliability["share thrown away"] = reliability["invalid_judge_output"] / total_per_judge

tk.show(tk.table(reliability, "Valid and invalid judgments, by judge",
                 note="Invalid = quoted text not found in the paper or rewrite, or a score that contradicts its own issues. These are excluded, not counted as zero."))
```

**What we learn:** Luna was a much less reliable judge — about one judgment in four had to be thrown away, against one in twenty-five for Astra. Astra is the better grader here, which is awkward, because it is also one of the writers.

---

## 11. What we can honestly conclude

From **one paper**, **one run**, and **AI judges only**:

- **The five-step pipeline works.** Both models produced complete, readable rewrites of the whole paper with no critical errors.
- **Astra's automatic rewrite looks strongest** — mostly because judges found far fewer problems in it, and its notes used all four known errors it had been told about. It also took more than twice as long as Luna.
- **Our 0–3 scale is too blunt.** Almost everything scored 3, so average scores cannot rank the rewrites. The list of problems is the useful signal.
- **The judges are not neutral.** They slightly favour themselves, and Luna often quoted text that does not exist.

What this does **not** show: that one model is better on other papers, or that real 13-year-olds understand these rewrites. For that we need more papers and real readers.

---

## 12. What to do next

1. **Check the judges yourself.** Read the major problems in chapter 9 and mark each as real or not. Record your scores in `rewrite_benchmark/runs/pine-pilot-v1/replicate-0/human_review.csv`.
2. **Sharpen the measurement.** Replace the 0–3 score with a count of problems, or ask judges *which of two rewrites is better*. Side-by-side choices usually separate close candidates much better.
3. **Plant deliberate mistakes.** Change one number, remove one qualification. A good judge must catch each one. That tells us whether the judge can be trusted.
4. **Add more papers** from other fields before tuning anything, and keep some papers aside as a final test the pipeline never saw.
5. **Then** use the good rewrites to train a smaller, cheaper translator.

The full pipeline that produced these results — with every program, the rubric, and the controls for making new calls — is in [`scientific-rewriting.md`](scientific-rewriting.md). This notebook is the guided tour.
---
rat:
  project: ..
  python:
    requires: ">=3.12"
    dependencies:
      - "-e ../../functai/python"
      - "-e ../../lmcc/python"
      - "pandas>=2,<3"
      - "pydantic>=2,<3"
      - "tiktoken==0.12.0"
---

# Can an AI explain a science paper to a 13-year-old?

*A hands-on walkthrough of our first experiment: rewriting a real paper, then checking the rewrites.*

We took one real scientific paper and asked two AI models to rewrite it so a curious 12–14-year-old could understand it — **without getting the science wrong**. Then we asked the same models to grade all the rewrites, including the one we wrote together in conversation.

By the end of this notebook you will know:

1. what the paper and the task look like,
2. how a rewrite is built in five small steps instead of one big one,
3. how the two models' rewrites compare, side by side,
4. whether they caught the mistakes hidden in the original paper,
5. what the AI judges said — and **how far we can trust them**.

**Nothing here costs anything.** Every result was produced once and saved. Each cell below just reads those files. One optional cell (chapter 3) lets you make a single live call if you want to see it happen.

---

## 1. Setup

To keep the cells below short, the repetitive code lives in a small helper file, **`tutorial_kit.py`** (in `rewrite_benchmark/`). It does two jobs:

- **Loading:** it reads the results we saved earlier — the paper, the three rewrites, the log of every model call, and the judges' scores. No model is called.
- **Showing:** it draws the tables, bar charts, colour-shaded tables and side-by-side text you will see, as HTML.

In the notebook we call it `tk`. First, tell Python where to find it:

```python
import sys
sys.path.insert(0, "rewrite_benchmark")  # finds tutorial_kit.py
```

Now load everything we saved:

```python
import pandas as pd
import tutorial_kit as tk

paper = tk.load_paper()
rewrites = tk.load_candidates()
calls = tk.load_calls()
judgments, issues, raw_judgments = tk.load_judgments()

print("Paper:", paper["title"])
print("Rewrites loaded:", [tk.NAMES[k] for k in rewrites])
print("Saved model calls:", len(calls))
```

```output
Paper: Recommendations for increasing yield of the edible Pinus pinea L. pine nuts
Rewrites loaded: ['Conversation', 'Astra', 'Luna']
Saved model calls: 166
```

We have three rewrites to compare:

| Name | Who wrote it | How |
|---|---|---|
| **Conversation** | GPT-6 Astra, with us | In a long chat, section by section, with tools and our feedback |
| **Astra** | GPT-6 Astra | Five automatic steps, no human help |
| **Luna** | GPT-6 Luna | The same five automatic steps |

---

## 2. Meet the paper

The paper is a Chilean study of **stone pine cones**: do heavier cones give more edible pine nuts? ([Loewe-Muñoz et al., PLOS ONE 2024](https://doi.org/10.1371/journal.pone.0300008), CC BY 4.0.)

How long is each part? We count **tokens**, the word-pieces an AI model reads.

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")   # the tokenizer used by GPT-4o

def count_tokens(text):
    return len(enc.encode(text))

# Tokens in each section
sizes = {}
for section in tk.ORDER:
    sizes[section] = count_tokens(paper["sections"][section])

# Tokens in the whole paper, references included
total = count_tokens(paper["source_text"] + paper["references"])

tk.show(tk.bars(sizes, "Tokens in each section of the original paper", unit=" tokens",
                note=f"Whole paper, with references: {total:,} tokens."))
```

<iframe class="rat-output" src="../_assets/generated/26feee4c7980.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

**What we learn:** the methods, results and discussion are the heavy parts. They are also where the numbers live — and numbers are where rewrites go wrong.

Let's read one. Change `SECTION` to any name above and rerun.

```python
SECTION = "results"   # try "abstract", "methods" or "discussion"

text = paper["sections"][SECTION]
tk.show(tk.text_box(f"Original · {SECTION}", text))
```

<iframe class="rat-output" src="../_assets/generated/bb6978af5af2.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

This is hard reading even for adults: "PY", "tertiles", "regression tree", percentages of percentages. That is exactly the kind of text we want to open up.

---

## 3. The task, written as a function

With [FunctAI](https://github.com/MaximeRivest/functai), a prompt is just a typed Python function. The **docstring** is the instruction, the **arguments** are what the model reads, and the **return type** is the shape of the answer.

Here is a small version of what the pipeline does:

```python
from pydantic import BaseModel
from functai import ai

class Rewrite(BaseModel):
    text: str                   # the section, rewritten for the reader
    editorial_notes: list[str]  # problems noticed in the original paper

@ai
def explain_section(section: str, whole_paper: str) -> Rewrite:
    """Rewrite this section of a scientific paper for a curious 12-14-year-old.
    Keep every number, comparison and uncertainty correct. Explain new words.
    Do not summarize. If the paper contradicts itself, say so in editorial_notes."""
    ...

# The same function, set to use GPT-6 Astra.
explain_with_astra = explain_section.using(lm="openai-codex:gpt-6-astra")

# Build the request for the abstract, but do NOT send it.
request = explain_with_astra.render(section=paper["sections"]["abstract"],
                                    whole_paper=paper["source_text"])

size = count_tokens(str(request.system) + str(request.messages))
print("The model would receive", size, "tokens.")
print("Nothing was sent: render() only builds the request.")
```

```output
The model would receive 5615 tokens.
Nothing was sent: render() only builds the request.
```

**Try it live (optional).** Running this cell makes one real model call. It rewrites the abstract only and takes roughly one to five minutes. Skip it if you don't want to wait.

`.stream(...)` makes the same call as `explain_with_astra(...)`, but shows the answer as it is being written, instead of waiting in silence.

```python
abstract = paper["sections"]["abstract"]
stream = explain_with_astra.stream(section=abstract, whole_paper=paper["source_text"])
stream.show()   # prints the rewrite as it arrives
```

```output
{"text":"Pinus pinea, also called stone pine, produces edible pine nuts. An important feature of this crop is its cone-to-pine-nut yield: the total weight of edible pine nuts in a cone, expressed as a percentage of the cone’s weight. The researchers call this PY. This yield is decreasing worldwide, which is worrying because pine nuts from this species are in high demand.\n\nThe researchers followed cone weight, the size and shape of seeds and edible pine nuts, and pine nut yield for 10 years in Chile, where stone pine does not grow naturally. Studying size and shape is called morphometry. Here, “seeds” means pine nuts still inside their shells, while “pine nuts” means the edible kernels after the shells have been removed.\n\nThe study covered seven plantations—areas of planted trees—in different growing environments. The researchers counted, measured and weighed 560 cones and the seeds and pine nuts they contained. They also checked seeds and pine nuts for damage.\n\nThe researchers defined two contrasting groups based on cone weight: heavy cones and light cones. They calculated PY and other features and compared the groups using a mixed linear model. This is a statistical method that can compare groups while accounting for differences linked to factors such as plantation and year. They also used regression trees, a method that repeatedly divides data into groups, to help explain why PY varied.\n\nThe cones in Chile were heavier, on average, than cones from the species’ native range—the places where it grows naturally: 474 g compared with 300 g. The section reports that heavy cones had more pine nuts per cone than light cones, 125 compared with 89, and a higher PY, 4.05% compared with 3.62%. Both differences were statistically significant, meaning the statistical test found evidence of a difference rather than just random variation. However, the paper elsewhere identifies 125 and 89 as the numbers of seeds still in their shells, not edible pine nuts; see the editorial notes.\n\nHeavy cones also had a lower percentage of damaged seeds than light cones: 9.0% compared with 15.9%. The researchers therefore concluded that PY depended on the size and shape of seeds and pine nuts, as well as on seed health. They suggest that fertilization—supplying nutrients—and irrigation—supplying water—could help trees produce more heavy cones and, as a result, increase PY.","editorial_notes":["The section labels 125 versus 89 as pine nuts per cone. The Results and Table 4 identify these as rounded counts of in-shell seeds (124.8 versus 89.1). The reported counts of healthy edible pine nuts are instead 112.9 versus 76.1 for heavy and light cones, respectively.","The section gives an average cone weight of 474 g, consistent with rounding Table 4’s 473.6 g. The Results and Discussion instead give 470 g, apparently using coarser rounding.","The proposed benefits of fertilization and irrigation are possibilities, not effects established by this study’s heavy-versus-light cone comparison. The Discussion explicitly says the observed differences between sites are insufficient to attribute them to irrigation."]}
```

When it finishes, `stream.result` is the finished answer. Show it next to the original:

```python
result = stream.result
tk.show(tk.side_by_side([("Original abstract", abstract),
                         ("Live rewrite", result.text)]),
        tk.bullet_list("Editorial notes", result.editorial_notes))
```

<iframe class="rat-output" src="../_assets/generated/7cb812a9b84f.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

---

## 4. Why five steps instead of one?

A whole paper in one call asks a lot of a model. So we copied what we did in conversation, and split the job into **five calls**:

```
Step 1  opening        title, abstract, first paragraph, conclusion
Step 2  introduction   the rest of the introduction
Step 3  methods
Step 4  results
Step 5  discussion
```

Each step receives **the full original paper** (the source of truth) plus **everything rewritten so far** and a growing **glossary**, so the explanations stay consistent from one section to the next.

That handoff has a cost: each step reads more than the last. Let's see it.

```python
# Keep only the writing calls (the rest are judging calls).
writing = calls[calls["kind"] == "write"]

# One row per step, one column per model. Each cell: tokens that step read.
per_step = writing.pivot(index="step", columns="model", values="input_tokens")
per_step = per_step.reindex(tk.STEPS)   # put the steps in the order they ran

tk.show(tk.table(per_step, "Tokens read by each writing step",
                 note="Each step reads the whole paper plus all earlier rewrites, so the input grows step by step."))
```

<iframe class="rat-output" src="../_assets/generated/23a4bc3a870a.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

And how long did each writer take?

```python
# Add up the five steps for each model.
minutes = writing.groupby("model")["minutes"].sum()

tk.show(tk.bars(minutes.to_dict(), "Total writing time for the whole paper", unit=" min", fmt="{:.0f}",
                colors=tk.MODEL_COLORS,
                note="Both used maximum reasoning effort. Five calls each, run one after another."))
```

<iframe class="rat-output" src="../_assets/generated/911e94d16149.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

**What we learn:** Astra took about 38 minutes, Luna about 16 — more than twice as long. Keep that in mind: if Astra's rewrite turns out better, part of the price is time.

---

## 5. Read the rewrites side by side

This is the heart of it. Pick a section and compare the original with all three rewrites. Try `"methods"` — the hardest one to explain well.

```python
SECTION = "methods"

# One column for the original, then one per rewrite.
columns = [("Original", paper["sections"][SECTION])]
for name, rewrite in rewrites.items():
    columns.append((tk.NAMES[name], rewrite["segments"][SECTION]))

tk.show(tk.side_by_side(columns))
```

Things worth looking for while you read:

- Is every **number** still there, with its unit?
- Does "11.9% higher" stay a *relative* increase, not 11.9 percentage points?
- Are new words (like *standard error* or *regression tree*) explained **before** they are used?
- Does it still sound like a scientist talking, rather than a children's book?

A quick numeric view: how long is each rewrite compared with the original?

```python
# One row per section; one column for the original and one per rewrite.
length = pd.DataFrame(index=pd.Index(tk.ORDER, name="section"))
length["Original"] = [count_tokens(paper["sections"][s]) for s in tk.ORDER]
for name, rewrite in rewrites.items():
    length[tk.NAMES[name]] = [count_tokens(rewrite["segments"][s]) for s in tk.ORDER]

tk.show(tk.table(length, "Length of each section, in tokens",
                 note="Longer is not worse: explaining a term takes words. Much shorter can mean something was dropped."))
```

**What we learn:** the two automatic rewrites are about a quarter **longer** than the original, while our conversation rewrite is about a fifth **shorter**. The biggest growth is in the abstract and conclusion, which Astra more than doubled. Whether that extra length helps or just pads is a question for the judges — and for you.

---

## 6. Did they catch the paper's own mistakes?

The original paper contains a few real errors. A good translator should notice them instead of copying them. We know of four:

1. The abstract calls 125 vs 89 "pine nuts", but the table shows those are **seeds still in their shells**.
2. The formula for damaged seeds is **printed wrongly**.
3. The results text **swaps two numbers** (2.60% and 4.04%) compared with Figure 1.
4. Figure 1 uses **328 cones**, not the 560 collected overall.

**An important catch:** both automatic writers were *given* a notes file listing these errors, and our conversation had discovered them along the way. So this is not a test of spotting errors from scratch. It tests something narrower: **did the writer actually use the evidence it was handed, and say so?**

Each writer lists what it noticed in its **editorial notes**. Let's check which traps each one mentions.

```python
caught = pd.DataFrame(index=pd.Index(list(tk.TRAPS), name="known error"))
for name, rewrite in rewrites.items():
    found = tk.trap_check(rewrite["notes"])   # {error: True or False}
    caught[tk.NAMES[name]] = ["✓ mentioned" if found[error] else "—" for error in tk.TRAPS]

tk.show(tk.table(caught,
                 "Known errors in the paper, and whether each writer flagged them",
                 note="A keyword check on the notes: it shows a mention, not that the fix was right. Read the notes to confirm."))
```

And here are the notes themselves, so you can confirm:

```python
WRITER = "astra"   # or "luna", "conversation_reference"

notes = rewrites[WRITER]["notes"]
tk.show(tk.bullet_list(f"{tk.NAMES[WRITER]} · {len(notes)} editorial notes", notes, numbered=True))
```

**What we learn:** Astra's notes mention all four errors. Luna's miss the abstract's seeds-versus-nuts mix-up, and our conversation never mentioned the 328 cones. A rewrite can read beautifully and still repeat a wrong number from the paper, so this matters. A fairer future test: give the writers **no** error list and see which mistakes they find on their own.

---

## 7. How the rewrites were graded

Reading three rewrites of a whole paper takes an hour. To scale up, we need **AI judges**. Each judge received the original, one anonymous rewrite, and a rubric, then gave a score from **0 to 3** on five aspects:

| Aspect | The question |
|---|---|
| Scientific faithfulness | Are the facts, numbers and uncertainty still correct? |
| Completeness | Did anything important disappear? |
| Accessibility | Could a curious 13-year-old follow it? |
| Document coherence | Does it read as one consistent explanation? |
| Editorial integrity | Were the paper's own errors handled honestly? |

The first three were scored **section by section**, the last two on the **whole document**. Both Astra and Luna judged all three rewrites — including their own.

Every judge also had to **quote** the exact text behind each complaint. If a quote did not exist in the paper or the rewrite, we threw that judgment away as unreliable.

Here is one real judgment, so you can see what a judge produces:

```python
# Valid judgments of Luna's rewrite, on accessibility, that list at least one problem.
examples = [j for j in raw_judgments
            if j["status"] == "ok"
            and j["candidate"] == "luna"
            and j["aspect"] == "accessibility"
            and j["assessment"]["issues"]]

tk.show(tk.judgment_card(examples[0]))
```

---

## 8. The scores — and why they are not enough

First, the headline number people always want: the average score.

```python
valid = judgments[judgments["status"] == "ok"]        # drop the judgments we threw away

average = valid.groupby("candidate")["score"].mean()   # one average per rewrite
average = average.rename(index=tk.NAMES)                # "astra" -> "Astra"

tk.show(tk.bars(average.to_dict(), "Average score (0–3), all aspects and both judges",
                fmt="{:.2f}", colors=tk.NAME_COLORS))
```

They all look close to perfect. **Is that real?** Let's look at how the scores are spread out.

```python
# For each rewrite: what percentage of its judgments got a 0, a 1, a 2 or a 3?
spread = pd.crosstab(valid["candidate"], valid["score"], normalize="index") * 100
spread = spread.reindex(columns=[0, 1, 2, 3], fill_value=0)   # show every score, even unused ones
spread = spread.rename(index=tk.NAMES)

tk.show(tk.heat(spread, "Share of judgments at each score (%)", lo=0, hi=100, fmt="{:.0f}%",
                note="Rows add to 100%. Most judgments are a perfect 3."))
```

**What we learn:** 70–87% of all judgments are a perfect 3. The scale is too coarse: it cannot tell "good" from "excellent". **This is a finding about our rubric, not proof the rewrites are equally good.** We need a sharper measuring tool, and the next chapter shows where the real signal is hiding.

---

## 9. Where the real signal is: the problems judges found

Scores compress everything into one digit. The **issues** each judge listed are far more informative. Let's count them.

```python
# Count the problems: one row per rewrite, one column per severity.
counted = pd.crosstab(issues["candidate"], issues["severity"])
counted = counted.reindex(columns=["critical", "major", "minor"], fill_value=0)
counted = counted.rename(index=tk.NAMES)

tk.show(tk.table(counted, "Problems found in each rewrite (both judges together)"))
```

**What we learn:** no one made a critical error. But Astra's automatic rewrite drew **far fewer complaints** than the other two. This is the clearest difference in the whole experiment.

Which sections do the complaints point to?

```python
# Count the problems again: one row per section, one column per rewrite.
where = pd.crosstab(issues["section"], issues["candidate"])
where = where.reindex(index=tk.ORDER + ["whole_document"], columns=list(rewrites), fill_value=0)
where = where.rename(columns=tk.NAMES)

# lo and hi are swapped on purpose: more problems = redder.
most = where.values.max()
tk.show(tk.heat(where, "Number of problems per section", lo=most, hi=0, fmt="{:.0f}",
                note="Redder = more problems. These are the places to read yourself."))
```

Now read the serious ones. These are the most useful lines in this notebook:

```python
serious = issues[issues["severity"] != "minor"]        # major and critical only
serious = serious.sort_values(["candidate", "section"])

boxes = [tk.issue_box(row) for _, row in serious.iterrows()]
tk.show(tk.title(f"All {len(serious)} major problems, with the quotes behind them"), *boxes)
```

---

## 10. Can we trust the judges?

Two things could make the judges misleading.

**Self-preference.** Does each model grade its own writing more kindly?

```python
per_section = valid[valid["section"] != "whole_document"]   # section judgments only

# One row per judge, one column per rewrite. Each cell: the average score given.
fairness = per_section.pivot_table(index="judge", columns="candidate", values="score", aggfunc="mean")
fairness = fairness[list(rewrites)].rename(columns=tk.NAMES)

tk.show(tk.heat(fairness, "Average section score given, by judge (rows) and rewrite (columns)",
                lo=2.4, hi=3, fmt="{:.2f}",
                note="Read across a row. The Conversation rewrite was also written by Astra."))
```

**What we learn:** each judge gave its own model's rewrite the highest average. The effect is small, but real enough that we should never let a model be the only judge of itself.

**Reliability.** How often did a judge quote text that does not exist?

```python
# Count valid and thrown-away judgments for each judge.
reliability = pd.crosstab(judgments["judge"], judgments["status"])
total_per_judge = reliability.sum(axis=1)
reliability["share thrown away"] = reliability["invalid_judge_output"] / total_per_judge

tk.show(tk.table(reliability, "Valid and invalid judgments, by judge",
                 note="Invalid = quoted text not found in the paper or rewrite, or a score that contradicts its own issues. These are excluded, not counted as zero."))
```

**What we learn:** Luna was a much less reliable judge — about one judgment in four had to be thrown away, against one in twenty-five for Astra. Astra is the better grader here, which is awkward, because it is also one of the writers.

---

## 11. What we can honestly conclude

From **one paper**, **one run**, and **AI judges only**:

- **The five-step pipeline works.** Both models produced complete, readable rewrites of the whole paper with no critical errors.
- **Astra's automatic rewrite looks strongest** — mostly because judges found far fewer problems in it, and its notes used all four known errors it had been told about. It also took more than twice as long as Luna.
- **Our 0–3 scale is too blunt.** Almost everything scored 3, so average scores cannot rank the rewrites. The list of problems is the useful signal.
- **The judges are not neutral.** They slightly favour themselves, and Luna often quoted text that does not exist.

What this does **not** show: that one model is better on other papers, or that real 13-year-olds understand these rewrites. For that we need more papers and real readers.

---

## 12. What to do next

1. **Check the judges yourself.** Read the major problems in chapter 9 and mark each as real or not. Record your scores in `rewrite_benchmark/runs/pine-pilot-v1/replicate-0/human_review.csv`.
2. **Sharpen the measurement.** Replace the 0–3 score with a count of problems, or ask judges *which of two rewrites is better*. Side-by-side choices usually separate close candidates much better.
3. **Plant deliberate mistakes.** Change one number, remove one qualification. A good judge must catch each one. That tells us whether the judge can be trusted.
4. **Add more papers** from other fields before tuning anything, and keep some papers aside as a final test the pipeline never saw.
5. **Then** use the good rewrites to train a smaller, cheaper translator.

The full pipeline that produced these results — with every program, the rubric, and the controls for making new calls — is in [`scientific-rewriting.md`](scientific-rewriting.md). This notebook is the guided tour.
---
rat:
  project: ..
  python:
    requires: ">=3.12"
    dependencies:
      - "-e ../../functai/python"
      - "-e ../../lmcc/python"
      - "pandas>=2,<3"
      - "pydantic>=2,<3"
      - "tiktoken==0.12.0"
---

# Can an AI explain a science paper to a 13-year-old?

*A hands-on walkthrough of our first experiment: rewriting a real paper, then checking the rewrites.*

We took one real scientific paper and asked two AI models to rewrite it so a curious 12–14-year-old could understand it — **without getting the science wrong**. Then we asked the same models to grade all the rewrites, including the one we wrote together in conversation.

By the end of this notebook you will know:

1. what the paper and the task look like,
2. how a rewrite is built in five small steps instead of one big one,
3. how the two models' rewrites compare, side by side,
4. whether they caught the mistakes hidden in the original paper,
5. what the AI judges said — and **how far we can trust them**.

**Nothing here costs anything.** Every result was produced once and saved. Each cell below just reads those files. One optional cell (chapter 3) lets you make a single live call if you want to see it happen.

---

## 1. Setup

To keep the cells below short, the repetitive code lives in a small helper file, **`tutorial_kit.py`** (in `rewrite_benchmark/`). It does two jobs:

- **Loading:** it reads the results we saved earlier — the paper, the three rewrites, the log of every model call, and the judges' scores. No model is called.
- **Showing:** it draws the tables, bar charts, colour-shaded tables and side-by-side text you will see, as HTML.

In the notebook we call it `tk`. First, tell Python where to find it:

```python
import sys
sys.path.insert(0, "rewrite_benchmark")  # finds tutorial_kit.py
```

Now load everything we saved:

```python
import pandas as pd
import tutorial_kit as tk

paper = tk.load_paper()
rewrites = tk.load_candidates()
calls = tk.load_calls()
judgments, issues, raw_judgments = tk.load_judgments()

print("Paper:", paper["title"])
print("Rewrites loaded:", [tk.NAMES[k] for k in rewrites])
print("Saved model calls:", len(calls))
```

```output
Paper: Recommendations for increasing yield of the edible Pinus pinea L. pine nuts
Rewrites loaded: ['Conversation', 'Astra', 'Luna']
Saved model calls: 166
```

We have three rewrites to compare:

| Name | Who wrote it | How |
|---|---|---|
| **Conversation** | GPT-6 Astra, with us | In a long chat, section by section, with tools and our feedback |
| **Astra** | GPT-6 Astra | Five automatic steps, no human help |
| **Luna** | GPT-6 Luna | The same five automatic steps |

---

## 2. Meet the paper

The paper is a Chilean study of **stone pine cones**: do heavier cones give more edible pine nuts? ([Loewe-Muñoz et al., PLOS ONE 2024](https://doi.org/10.1371/journal.pone.0300008), CC BY 4.0.)

How long is each part? We count **tokens**, the word-pieces an AI model reads.

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")   # the tokenizer used by GPT-4o

def count_tokens(text):
    return len(enc.encode(text))

# Tokens in each section
sizes = {}
for section in tk.ORDER:
    sizes[section] = count_tokens(paper["sections"][section])

# Tokens in the whole paper, references included
total = count_tokens(paper["source_text"] + paper["references"])

tk.show(tk.bars(sizes, "Tokens in each section of the original paper", unit=" tokens",
                note=f"Whole paper, with references: {total:,} tokens."))
```

<iframe class="rat-output" src="../_assets/generated/26feee4c7980.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

**What we learn:** the methods, results and discussion are the heavy parts. They are also where the numbers live — and numbers are where rewrites go wrong.

Let's read one. Change `SECTION` to any name above and rerun.

```python
SECTION = "results"   # try "abstract", "methods" or "discussion"

text = paper["sections"][SECTION]
tk.show(tk.text_box(f"Original · {SECTION}", text))
```

<iframe class="rat-output" src="../_assets/generated/bb6978af5af2.html" sandbox="allow-scripts" loading="lazy" style="width:100%;height:440px;border:0"></iframe>

This is hard reading even for adults: "PY", "tertiles", "regression tree", percentages of percentages. That is exactly the kind of text we want to open up.

---

## 3. The task, written as a function

With [FunctAI](https://github.com/MaximeRivest/functai), a prompt is just a typed Python function. The **docstring** is the instruction, the **arguments** are what the model reads, and the **return type** is the shape of the answer.

Here is a small version of what the pipeline does:

```python
from pydantic import BaseModel
from functai import ai

class Rewrite(BaseModel):
    text: str                   # the section, rewritten for the reader
    editorial_notes: list[str]  # problems noticed in the original paper

@ai
def explain_section(section: str, whole_paper: str) -> Rewrite:
    """Rewrite this section of a scientific paper for a curious 12-14-year-old.
    Keep every number, comparison and uncertainty correct. Explain new words.
    Do not summarize. If the paper contradicts itself, say so in editorial_notes."""
    ...

# The same function, set to use GPT-6 Astra.
explain_with_astra = explain_section.using(lm="openai-codex:gpt-6-astra")

# Build the request for the abstract, but do NOT send it.
request = explain_with_astra.render(section=paper["sections"]["abstract"],
                                    whole_paper=paper["source_text"])

size = count_tokens(str(request.system) + str(request.messages))
print("The model would receive", size, "tokens.")
print("Nothing was sent: render() only builds the request.")
```

```output
The model would receive 5615 tokens.
Nothing was sent: render() only builds the request.
```

**Try it live (optional, one call).** Set `LIVE = True`. It rewrites the abstract only, and takes roughly one to five minutes.

```python
LIVE = False   # set to True to make one real call

if LIVE:
    abstract = paper["sections"]["abstract"]
    result = explain_with_astra(section=abstract, whole_paper=paper["source_text"])

    tk.show(tk.side_by_side([("Original abstract", abstract),
                             ("Live rewrite", result.text)]),
            tk.bullet_list("Editorial notes", result.editorial_notes))
else:
    print("Skipped. Set LIVE = True to make one real call.")
```

---

## 4. Why five steps instead of one?

A whole paper in one call asks a lot of a model. So we copied what we did in conversation, and split the job into **five calls**:

```
Step 1  opening        title, abstract, first paragraph, conclusion
Step 2  introduction   the rest of the introduction
Step 3  methods
Step 4  results
Step 5  discussion
```

Each step receives **the full original paper** (the source of truth) plus **everything rewritten so far** and a growing **glossary**, so the explanations stay consistent from one section to the next.

That handoff has a cost: each step reads more than the last. Let's see it.

```python
# Keep only the writing calls (the rest are judging calls).
writing = calls[calls["kind"] == "write"]

# One row per step, one column per model. Each cell: tokens that step read.
per_step = writing.pivot(index="step", columns="model", values="input_tokens")
per_step = per_step.reindex(tk.STEPS)   # put the steps in the order they ran

tk.show(tk.table(per_step, "Tokens read by each writing step",
                 note="Each step reads the whole paper plus all earlier rewrites, so the input grows step by step."))
```

And how long did each writer take?

```python
# Add up the five steps for each model.
minutes = writing.groupby("model")["minutes"].sum()

tk.show(tk.bars(minutes.to_dict(), "Total writing time for the whole paper", unit=" min", fmt="{:.0f}",
                colors=tk.MODEL_COLORS,
                note="Both used maximum reasoning effort. Five calls each, run one after another."))
```

**What we learn:** Astra took about 38 minutes, Luna about 16 — more than twice as long. Keep that in mind: if Astra's rewrite turns out better, part of the price is time.

---

## 5. Read the rewrites side by side

This is the heart of it. Pick a section and compare the original with all three rewrites. Try `"methods"` — the hardest one to explain well.

```python
SECTION = "methods"

# One column for the original, then one per rewrite.
columns = [("Original", paper["sections"][SECTION])]
for name, rewrite in rewrites.items():
    columns.append((tk.NAMES[name], rewrite["segments"][SECTION]))

tk.show(tk.side_by_side(columns))
```

Things worth looking for while you read:

- Is every **number** still there, with its unit?
- Does "11.9% higher" stay a *relative* increase, not 11.9 percentage points?
- Are new words (like *standard error* or *regression tree*) explained **before** they are used?
- Does it still sound like a scientist talking, rather than a children's book?

A quick numeric view: how long is each rewrite compared with the original?

```python
# One row per section; one column for the original and one per rewrite.
length = pd.DataFrame(index=pd.Index(tk.ORDER, name="section"))
length["Original"] = [count_tokens(paper["sections"][s]) for s in tk.ORDER]
for name, rewrite in rewrites.items():
    length[tk.NAMES[name]] = [count_tokens(rewrite["segments"][s]) for s in tk.ORDER]

tk.show(tk.table(length, "Length of each section, in tokens",
                 note="Longer is not worse: explaining a term takes words. Much shorter can mean something was dropped."))
```

**What we learn:** the two automatic rewrites are about a quarter **longer** than the original, while our conversation rewrite is about a fifth **shorter**. The biggest growth is in the abstract and conclusion, which Astra more than doubled. Whether that extra length helps or just pads is a question for the judges — and for you.

---

## 6. Did they catch the paper's own mistakes?

The original paper contains a few real errors. A good translator should notice them instead of copying them. We know of four:

1. The abstract calls 125 vs 89 "pine nuts", but the table shows those are **seeds still in their shells**.
2. The formula for damaged seeds is **printed wrongly**.
3. The results text **swaps two numbers** (2.60% and 4.04%) compared with Figure 1.
4. Figure 1 uses **328 cones**, not the 560 collected overall.

**An important catch:** both automatic writers were *given* a notes file listing these errors, and our conversation had discovered them along the way. So this is not a test of spotting errors from scratch. It tests something narrower: **did the writer actually use the evidence it was handed, and say so?**

Each writer lists what it noticed in its **editorial notes**. Let's check which traps each one mentions.

```python
caught = pd.DataFrame(index=pd.Index(list(tk.TRAPS), name="known error"))
for name, rewrite in rewrites.items():
    found = tk.trap_check(rewrite["notes"])   # {error: True or False}
    caught[tk.NAMES[name]] = ["✓ mentioned" if found[error] else "—" for error in tk.TRAPS]

tk.show(tk.table(caught,
                 "Known errors in the paper, and whether each writer flagged them",
                 note="A keyword check on the notes: it shows a mention, not that the fix was right. Read the notes to confirm."))
```

And here are the notes themselves, so you can confirm:

```python
WRITER = "astra"   # or "luna", "conversation_reference"

notes = rewrites[WRITER]["notes"]
tk.show(tk.bullet_list(f"{tk.NAMES[WRITER]} · {len(notes)} editorial notes", notes, numbered=True))
```

**What we learn:** Astra's notes mention all four errors. Luna's miss the abstract's seeds-versus-nuts mix-up, and our conversation never mentioned the 328 cones. A rewrite can read beautifully and still repeat a wrong number from the paper, so this matters. A fairer future test: give the writers **no** error list and see which mistakes they find on their own.

---

## 7. How the rewrites were graded

Reading three rewrites of a whole paper takes an hour. To scale up, we need **AI judges**. Each judge received the original, one anonymous rewrite, and a rubric, then gave a score from **0 to 3** on five aspects:

| Aspect | The question |
|---|---|
| Scientific faithfulness | Are the facts, numbers and uncertainty still correct? |
| Completeness | Did anything important disappear? |
| Accessibility | Could a curious 13-year-old follow it? |
| Document coherence | Does it read as one consistent explanation? |
| Editorial integrity | Were the paper's own errors handled honestly? |

The first three were scored **section by section**, the last two on the **whole document**. Both Astra and Luna judged all three rewrites — including their own.

Every judge also had to **quote** the exact text behind each complaint. If a quote did not exist in the paper or the rewrite, we threw that judgment away as unreliable.

Here is one real judgment, so you can see what a judge produces:

```python
# Valid judgments of Luna's rewrite, on accessibility, that list at least one problem.
examples = [j for j in raw_judgments
            if j["status"] == "ok"
            and j["candidate"] == "luna"
            and j["aspect"] == "accessibility"
            and j["assessment"]["issues"]]

tk.show(tk.judgment_card(examples[0]))
```

---

## 8. The scores — and why they are not enough

First, the headline number people always want: the average score.

```python
valid = judgments[judgments["status"] == "ok"]        # drop the judgments we threw away

average = valid.groupby("candidate")["score"].mean()   # one average per rewrite
average = average.rename(index=tk.NAMES)                # "astra" -> "Astra"

tk.show(tk.bars(average.to_dict(), "Average score (0–3), all aspects and both judges",
                fmt="{:.2f}", colors=tk.NAME_COLORS))
```

They all look close to perfect. **Is that real?** Let's look at how the scores are spread out.

```python
# For each rewrite: what percentage of its judgments got a 0, a 1, a 2 or a 3?
spread = pd.crosstab(valid["candidate"], valid["score"], normalize="index") * 100
spread = spread.reindex(columns=[0, 1, 2, 3], fill_value=0)   # show every score, even unused ones
spread = spread.rename(index=tk.NAMES)

tk.show(tk.heat(spread, "Share of judgments at each score (%)", lo=0, hi=100, fmt="{:.0f}%",
                note="Rows add to 100%. Most judgments are a perfect 3."))
```

**What we learn:** 70–87% of all judgments are a perfect 3. The scale is too coarse: it cannot tell "good" from "excellent". **This is a finding about our rubric, not proof the rewrites are equally good.** We need a sharper measuring tool, and the next chapter shows where the real signal is hiding.

---

## 9. Where the real signal is: the problems judges found

Scores compress everything into one digit. The **issues** each judge listed are far more informative. Let's count them.

```python
# Count the problems: one row per rewrite, one column per severity.
counted = pd.crosstab(issues["candidate"], issues["severity"])
counted = counted.reindex(columns=["critical", "major", "minor"], fill_value=0)
counted = counted.rename(index=tk.NAMES)

tk.show(tk.table(counted, "Problems found in each rewrite (both judges together)"))
```

**What we learn:** no one made a critical error. But Astra's automatic rewrite drew **far fewer complaints** than the other two. This is the clearest difference in the whole experiment.

Which sections do the complaints point to?

```python
# Count the problems again: one row per section, one column per rewrite.
where = pd.crosstab(issues["section"], issues["candidate"])
where = where.reindex(index=tk.ORDER + ["whole_document"], columns=list(rewrites), fill_value=0)
where = where.rename(columns=tk.NAMES)

# lo and hi are swapped on purpose: more problems = redder.
most = where.values.max()
tk.show(tk.heat(where, "Number of problems per section", lo=most, hi=0, fmt="{:.0f}",
                note="Redder = more problems. These are the places to read yourself."))
```

Now read the serious ones. These are the most useful lines in this notebook:

```python
serious = issues[issues["severity"] != "minor"]        # major and critical only
serious = serious.sort_values(["candidate", "section"])

boxes = [tk.issue_box(row) for _, row in serious.iterrows()]
tk.show(tk.title(f"All {len(serious)} major problems, with the quotes behind them"), *boxes)
```

---

## 10. Can we trust the judges?

Two things could make the judges misleading.

**Self-preference.** Does each model grade its own writing more kindly?

```python
per_section = valid[valid["section"] != "whole_document"]   # section judgments only

# One row per judge, one column per rewrite. Each cell: the average score given.
fairness = per_section.pivot_table(index="judge", columns="candidate", values="score", aggfunc="mean")
fairness = fairness[list(rewrites)].rename(columns=tk.NAMES)

tk.show(tk.heat(fairness, "Average section score given, by judge (rows) and rewrite (columns)",
                lo=2.4, hi=3, fmt="{:.2f}",
                note="Read across a row. The Conversation rewrite was also written by Astra."))
```

**What we learn:** each judge gave its own model's rewrite the highest average. The effect is small, but real enough that we should never let a model be the only judge of itself.

**Reliability.** How often did a judge quote text that does not exist?

```python
# Count valid and thrown-away judgments for each judge.
reliability = pd.crosstab(judgments["judge"], judgments["status"])
total_per_judge = reliability.sum(axis=1)
reliability["share thrown away"] = reliability["invalid_judge_output"] / total_per_judge

tk.show(tk.table(reliability, "Valid and invalid judgments, by judge",
                 note="Invalid = quoted text not found in the paper or rewrite, or a score that contradicts its own issues. These are excluded, not counted as zero."))
```

**What we learn:** Luna was a much less reliable judge — about one judgment in four had to be thrown away, against one in twenty-five for Astra. Astra is the better grader here, which is awkward, because it is also one of the writers.

---

## 11. What we can honestly conclude

From **one paper**, **one run**, and **AI judges only**:

- **The five-step pipeline works.** Both models produced complete, readable rewrites of the whole paper with no critical errors.
- **Astra's automatic rewrite looks strongest** — mostly because judges found far fewer problems in it, and its notes used all four known errors it had been told about. It also took more than twice as long as Luna.
- **Our 0–3 scale is too blunt.** Almost everything scored 3, so average scores cannot rank the rewrites. The list of problems is the useful signal.
- **The judges are not neutral.** They slightly favour themselves, and Luna often quoted text that does not exist.

What this does **not** show: that one model is better on other papers, or that real 13-year-olds understand these rewrites. For that we need more papers and real readers.

---

## 12. What to do next

1. **Check the judges yourself.** Read the major problems in chapter 9 and mark each as real or not. Record your scores in `rewrite_benchmark/runs/pine-pilot-v1/replicate-0/human_review.csv`.
2. **Sharpen the measurement.** Replace the 0–3 score with a count of problems, or ask judges *which of two rewrites is better*. Side-by-side choices usually separate close candidates much better.
3. **Plant deliberate mistakes.** Change one number, remove one qualification. A good judge must catch each one. That tells us whether the judge can be trusted.
4. **Add more papers** from other fields before tuning anything, and keep some papers aside as a final test the pipeline never saw.
5. **Then** use the good rewrites to train a smaller, cheaper translator.

The full pipeline that produced these results — with every program, the rubric, and the controls for making new calls — is in [`scientific-rewriting.md`](scientific-rewriting.md). This notebook is the guided tour.
