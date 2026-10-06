# Writing from notes

**The question.** Does a writer need to know the science to write a paper well for a curious
14-year-old, or is it enough to be a very good writer who is *given* the knowledge? If the
second is true, a small model (0.8B) only needs writing skill and everyday knowledge, and the
science can come in with the input.

**The analogy.** A Scientific American writer covers fields they do not know. They work from
reporting notes: the facts, what the hard words mean, and what matters (the scientist tells
them). Their own contribution is the writing: order, links, words, pace, warmth.

**The experiment.** The writer never sees the paper. Opus, playing the scientist, turns each
part of a paper into notes; a writer turns the notes into prose. Here the writer is also Opus:
this measures the *ceiling* and checks that the method holds before any small model is trained
on notes.

```
paper part ──► 1. notes (scientist) ──► 2. check notes vs paper ──► fix ──► check again
                                                                            │
                     ┌──────────────────────────────────────────────────────┘
                     ▼
              3. write from notes alone, at two levels of help
                     │
                     ├──► 4. trace: every fact of the notes in the text? anything added?
                     └──► 5. fair blind judge: from-notes text vs the old Opus answer
```

## The notes

For each part (the opening, then the introduction, methods, results, discussion):

| field | what it holds |
|---|---|
| headings | every heading, in order |
| facts | every piece of information, one per line, with its numbers, names, citations and hedges; importance: main / support / detail |
| paragraphs | the plan: under each heading, the paragraphs in order, each with a purpose and its facts |
| definitions | what each hard word means, in general (never a side fact); *central* ideas get an everyday picture |

## Two levels of help

The point is to find where a writer breaks: writing sentences, or planning.

| level | the writer gets | the writer must do |
|---|---|---|
| 1 | facts, importance, paragraph plan, definitions with pictures | write the sentences |
| 3 | facts shuffled under each heading, definitions only | also group, order, link, stress |

(Level 2, only a section plan, and level 4, the paper itself, are left for later. Level 4 is
what the current models do.)

## The checks

- **Notes vs paper** (`check_notes`): did the notes lose, change or add anything? Wrong
  definitions? If yes, Opus fixes the notes and the checker looks again.
- **Trace** (`trace`): every fact of the notes is kept, changed, weaker/stronger, or missing in
  the text, and every outside fact the text added is listed. This makes faithfulness a count,
  not an opinion.
- **Fair blind judge** (`compare`): which version would the reader prefer, given it must be
  faithful to the paper? Asked in both orders; a win must hold both ways. Unlike the v3 pilot's
  judge, it sees what each reader already read (so a term explained one section earlier is not
  "unexplained"), and a short correct definition does not count as an added claim.
- **Mechanical**: readability (sentence length, grade) and share of the paper's numbers kept.

The comparison is against the **old Opus answer**, written from the paper with full knowledge.
v3 vs old is judged again too, to see what the fair judge alone changes.

## Files

```
README.md     this page
prompts.py    rounds 1-3: everything the models are told (the AI functions and the briefs)
render.py     rounds 1-3: how the notes are shown: full (checkers), level 1, level 3
pilot.py      rounds 1-3: runs it on 5 papers from the v3 pilot, one folder per round
viewer.py     rounds 1-3: builds out/round<N>/index.html to read everything side by side
out/round<N>/ one JSON per paper, summary.json, run.log, index.html

bundle/       round 4: the dense bundle
  sources.py    title, abstract, captions and tables, word for word from the paper's XML
  prompts.py    everything the models are told (front, part, check, fix, write, trace, judge)
  render.py     the bundle as the writer reads it; the views the checkers read
  pilot.py      runs it (same 5 papers), one folder per try: out/4a, out/4b, out/4c;
                also the code checks of the output's form (placeholders, citations, prose only)
  rejudge.py    judges a finished try again with the current judge, without rewriting
  produce.py    the 300-paper training-data run (Opus and Astra, 150 each), every call saved:
                select / run opus / run astra / status; data in out/data300/ (not in git)
  structure.py  4d: headings, paragraphs, citations [n] and placeholder places, by code from the XML
  checks.py     4d: code checks of each paragraph's facts (numbers, citations, hedge words)
  cache.py      4d: the shared glossary cache (glossary_cache.json)
  pilot_4d.py   4d: runs it (code builds the structure, the model writes the facts)
  viewer.py     builds out/<try>/index.html: the bundle, verdicts, trace, texts side by side
```

Run: `.venv/bin/python training/notes/pilot.py [--papers 1]`, then
`.venv/bin/python training/notes/viewer.py [round]`.

The 5 papers (different subfields, 2 with invented species names):
PMC12291832 (shrimp, water science), PMC12010427 (environmental chemistry), PMC7616100
(environmental engineering), PMC12466960 (conservation), PMC11227166 (ecological modelling).

## Rounds

Every change to the prompts is a new round with its own folder, so rounds can be compared.

**Round 1** (1 paper). The writer stuck to the notes almost perfectly (217 of 218 facts kept,
2 small additions), but lost all 5 parts to the old answer at both levels. The judge's reasons
pointed at the notes, not the writer: the scientist had added outside facts (where the shrimp
comes from, what a measure "was based on"), definitions carried side facts, notes contained
facts about the document itself ("the abstract is a Simple Summary"), the title was copied,
and the text explained every word it was given, so it read heavier.

**Round 2** (same paper). Notes may add nothing, not even in definitions; no layout facts;
definitions marked central; statistics sheet given; a fix step corrects the notes; the writer
explains only what is needed, may use plainer names, writes a real title. Result: level 1 won
2 of 5 parts (was 0), level 3 tied 1. Remaining: jargon kept where the old answer uses a plain
word ("poikilothermic" vs "cold-blooded"), definition-heavy methods, notes still giving
study-specific definitions the paper does not state.

**Round 3** (5 papers, 25 parts, 336 Opus calls, about 1% of the weekly allowance). The writer
prefers plain words and keeps a technical name only when needed; at most one new explanation
per sentence; definitions give only the general meaning; the notes checker sees the rest of
the paper.

| fair blind judge, 25 parts | new wins | no clear winner | old wins |
|---|---|---|---|
| level 1 (plan) vs old | 3 | 9 | 13 |
| level 3 (shuffled) vs old | 2 | 2 | 21 |
| v3 vs old | 6 | 7 | 12 |
| v3 vs old, the pilot's judge, same parts | 3 | 7 | 15 |

- **The notes hold the paper.** 993 facts. Before the fix: 1 major and 48 minor problems,
  mostly loose definitions. After the fix: 0 major, 25 minor.
- **The writer holds the notes.** Level 1 kept 981 of 990 facts with 13 outside additions in
  about 34,000 words; level 3 about the same, with 20 additions and 5 wrong explanations.
- **Planning matters a lot.** Same facts and definitions, but without the plan level 3 loses
  21 of 25 (level 1: 13). The judge's reasons: it reorders the paper's flow and stresses the
  wrong things. For a small model this is the key finding: either the input carries the plan,
  or planning must be learned.
- **Level 1 is close to the old answer, not above it.** Its openings are as good or better
  (2 wins, 3 ties). It loses mostly in results and discussion. The reasons: denser (paragraphs
  101 words vs 87), keeps more technical terms where the old answer keeps one plain name
  throughout ("seed origin", "formula"), and a few additions. Some "additions" are false
  alarms: "Netatmo" stations are named in the methods, but the judge only sees the results.
- **The fair judge alone helps v3** (3 to 6 wins, 15 to 12 losses on the same parts), but
  does not turn it around.

**Round 4: the dense bundle** (folder `bundle/`, 5 papers). Instead of sentence-like notes per
part (1.8 times the paper's length), one dense, structured bundle per paper: title, abstract,
every figure caption and table copied word for word from the paper's XML; the paper's *things*
(names defined once); a short glossary and the reference sheets; and an outline of every part
with dense facts (symbols, short names, claims with their hedge words and evidence) placed in a
paragraph plan. The writer gets the whole bundle and writes the paper one part at a time. The
judge now also sees the rest of the paper and, from 4b's re-judging on, its captions and tables.

- **4a** (1 paper): the bundle came out longer than the paper (a 2,175-word glossary), and the
  trace, seeing only the part's facts, counted what the writer took from the glossary or a
  table as "added".
- **4b** (5 papers, about 400 Opus calls with the re-judging): word budgets, trace sees the
  whole bundle.

| fair judge, 25 parts (sees whole paper, captions, tables) | first wins | no clear winner | second wins |
|---|---|---|---|
| bundle vs old | 6 | 6 | 13 |
| bundle vs round 3 level 1 | 12 | 3 | 10 |
| round 3 level 1 vs old | 5 | 7 | 13 |

- **Density.** Dense outlines are 65-93% of the words of the text they come from (the
  shrimp paper 67%, the table-heavy conservation paper 93%); the glossary is about 600 words;
  the whole bundle without reference sheets is about the paper's length, because the abstract,
  captions and tables are verbatim (and the corpus text does not always count the tables).
- **Faithful.** 520 of 528 items kept, 8 changed or with changed certainty, 17 small outside
  additions and 4 wrong explanations in 30,000 words. Unpacking dense notes causes a few new
  slips ("until 1945" written as "by 1945"; a list of three places read as two).
- **Same quality as round 3 from half the input,** but still behind the old answer. The judge's
  reasons, in order: paragraphs too long (104 words on average vs 87); whole tables
  reproduced in the text (a 27-row appendix table inline), which the old answers never had
  because the corpus text lacked them; a few slips from unpacking.
- **The judge is noisy.** Asked again on the same texts with only the captions added, it gave
  the same verdict 56 times out of 75. With 25 parts, a difference of 2 or 3 wins is noise.

- **4c** (5 papers, about 250 Opus calls): **the writer writes text only.** Figures and tables
  are never rewritten: the writer puts a placeholder on its own line (`[F2]`, `[T1]`) where the
  outline places it, and the app shows the original. Citations are `[n]` or `[n,m]`, numbers
  into the paper's reference list (built by code from the XML as "Lodders 2021"), whatever the
  paper's own style was. No lists, tables or bold: headings and prose only. So the model learns
  one way of doing each thing, and presentation (figures, tables, citations, glossary pop-ups)
  stays in code. Also: planned paragraphs of at most about 4 facts, written paragraphs of at
  most about 90 words. The judge compares the prose only.

| fair judge, 25 parts, prose only | first wins | no clear winner | second wins |
|---|---|---|---|
| 4c vs old | **10** | 6 | 9 |
| 4c vs 4b | 10 | 8 | 7 |

  - **Form, checked by code:** every placeholder exactly once and in the right part (0 errors
    in 25 parts); no author-year citations, lists, tables or bold; 4 citation numbers not in
    the part's notes (after counting the notes' ranges, which the writer correctly expanded).
  - **Faithful:** 482 of 492 facts kept, 10 changed, 12 small outside additions, 2 wrong
    explanations. 94% of the notes' numbers appear in the text.
  - **Shorter and lighter:** 25,000 words against 36,000 for the old answers (no captions or
    tables written any more, and less padding), paragraphs 67 words (old 87), grade 9.0.
  - **Still behind the old answer in the opening** (2 wins, 3 losses); the judge's reasons:
    an overstated claim ("do not overlap at all" for an overlap of 0.07), a detail dropped
    from the abstract, a fact moved from "shrimp ponds in general" to "our tanks".

- **4d** (5 papers, about 180 Opus calls): **code builds the bundle's structure.** From the
  paper's XML (`structure.py`): headings, paragraphs in order, citations turned into `[n]` from
  the XML's own links (bracketed, superscript ranges and author-year alike), and each
  placeholder placed after the paragraph that first mentions it. Labels are now the same in
  every paper (`## introduction_rest` was sometimes a label, sometimes a real heading). The
  model only turns each numbered paragraph into dense facts, and writes plain meanings only for
  words the shared glossary cache (`cache.py`) does not have. Code checks every paragraph's
  numbers, citations and hedge words against its facts (`checks.py`); the model check also ran
  everywhere, to measure what code misses.

  | | 4c (model does the structure) | 4d (code does the structure) |
  |---|---|---|
  | Opus calls per paper, to make data | about 30 | about 19 (26 with the model check on every part) |
  | judge, 4d vs 4c, prose only | | 8 wins, 8 no clear winner, 9 losses: the same quality |
  | form errors (placeholders, citations, styles) | 4 | 2 |
  | facts kept by the writer | 482 of 492 | 647 of 656 |

  - **Code checks catch counting problems, not meaning.** They flagged 14 of 25 parts (18
    hedge words and 9 numbers lost), all fixed. The model check found a major problem in 5
    parts; 2 of these were in parts code had passed, both meaning errors no count can see
    (the notes said the shrimp's feeding sounds "clearly decreased" where the paper says the
    nitrogen compounds barely changed them).
  - **The glossary cache barely helps yet:** 19 of 194 meanings reused, because 5 papers from 5
    fields share few words. It should grow with 1,000 papers; to be measured.
  - **Notes a little less dense:** paragraph by paragraph gives more facts (648 vs 486).

- **4e** (5 papers, about 190 Opus calls; same pipeline file, `pilot_4d.py`): three fixes to
  the scientist's prompts. (1) Worked examples for both steps (`EXAMPLE_FRONT`,
  `EXAMPLE_PARAGRAPHS` in `prompts.py`), taken from a paper outside the test set (sorghum under
  drought): two paragraphs and their facts, and good and bad glossary entries, including
  offline Wikipedia's wrong senses ("culm: waste coal", "accession: a king taking the throne").
  (2) A picture may never be a number, a typical value or a fact. (3) A new step,
  `check_glossary`, checks things and glossary on their own before new meanings enter the
  shared cache. A first run of paper 1 lost the "15 to 40 words" glossary limit and doubled
  the glossary; it is kept in `out/4e/discarded/`, the limit was restored and all 5 papers rerun
  from an empty cache.

  | | 4d | 4e |
  |---|---|---|
  | judge, 4e vs 4d (prose only) | | 10 wins, 6 no clear winner, 9 losses: the same |
  | model check: major problems | 5 | 3 |
  | model check: facts not in the paper | 8 | **0** |
  | pictures carrying facts ("Q10 is usually about 2") | some | none seen |
  | glossary check corrections | (no check) | 7 (2 entries removed, e.g. "heatmap", a word the paper never uses) |
  | dense facts, words vs the paper's paragraphs | 79% | 79% |

  - **The examples made the notes cleaner, not shorter.** No outside facts slipped into the
    notes, but the "about half the words" in the example was ignored like the budget before.
  - **The code check's hedge list was too eager.** "Potential energy surface", "potential
    habitats" and "we could not identify" were flagged as lost hedges. Without "potential",
    "possible" and "could not", the flagged hedge words drop from 22 to 6 in 4e, and the parts
    sent to a fix by code from 14 to 12 (4d: 14 to 9). `checks.py` now does this.

## Things to know when reading the results

- **Old answers on renamed papers leak the real species.** The old answers were written for the
  real paper and renamed afterwards, so in PMC12291832 the old answer still says "whiteleg
  shrimp" (the real common name of *Penaeus vannamei*) five times, while the paper only says
  *Tobrabut grofel*. The fair judge counts it against the old answer.
- **The judge is Opus, and so is every writer.** It cannot prefer one model's style over
  another here, but it is one judge; its reasons are opinions, which is why both orders must
  agree and why the trace counts exist.
- **Numbers kept** counts every number of the original, including section numbers ("5.
  Conclusions"); dropping those is intended, so 93% here can mean nothing was lost.

## The 300-paper training-data run (started 2026-10-06)

`bundle/produce.py` makes training data with the 4e recipe, plus a check-and-revise loop on
the written text: code form check and trace; if anything is found, the teacher revises the
part once, and both checks run again. 300 papers (3,000-7,000 words, all 11 subfields, none
of the validation, test, rounds 1-4 or prompt-example papers; 25% with invented species
names), 150 written entirely by Opus and 150 entirely by Astra, so the two teachers can be
compared and mixed.

Everything is kept in `bundle/out/data300/<teacher>/<paper_id>/` (see the docstring of
`produce.py`): the selection record, the paper's structure, every model call with its full
inputs and output (`calls.jsonl`), the bundle at every stage, every draft, check and
revision of the text. That is enough to train the writer (bundle -> text), a bundle-maker
(paragraph -> facts; paper -> things and glossary) and checkers, without calling a model again.

Limits: Opus stops starting papers at 97% of the weekly Claude allowance and waits when the
5-hour window is full; Astra stops at 97% of the weekly Codex allowance or the moment the
credit balance drops, so it never spends credits. `produce.py status` shows progress, limits
and the stop reason (also in `out/data300/STATUS-<teacher>.txt`). Rerunning `produce.py run
<teacher>` after a reset continues where it stopped.
