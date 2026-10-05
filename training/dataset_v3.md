# Training set v3: a writer that knows nothing a 12-year-old doesn't know

Status: design, not built (2026-10-04).

## Goal

A small student model should spend its weights on *writing*: understanding the paper,
composing plain, warm, well-ordered text, and reasoning the way a kid does ("a pika is a
mammal, so it breathes air, has fur, feeds its young milk"). It should never draw on
specialist memory. Every fact it writes must come from one of three sources:

| source | what |
|---|---|
| **P** the paper | the section being rewritten (and the rewritten opening) |
| **G** the glossary | reference entries given in the input |
| **K** kid knowledge | words usually learned before 12, school-level science, and general traits of a category ("animals eat", "acids are sour") |

When an explanation would need anything else, the model writes a **marker** instead of
explaining from memory, so the term can be explained afterwards (a lookup, a bigger
model, or a tap-to-explain in the reader). And the training data must make it impossible
to succeed by remembering.

## What v1 and v2 teach today (measured 2026-10-04)

| | v1 `examples.parquet` | v2 `examples-glossary.parquet` |
|---|---|---|
| papers / training conversations | 3,159 / 14,894 | 4,416 / 20,874 |
| answers by | Opus 7,897 · Astra 6,997 | Opus 13,877 · Astra 6,997 |
| glossary in the input | none | median 20 entries per conversation |
| answers written with the glossary? | — | **no**: same Opus/Astra answers as before |

Why the models still explain from memory:
1. **The answers were written without the glossary.** Their explanations come from
   Opus's own knowledge; the training score rewards reproducing them, so reading the
   glossary earns nothing.
2. **The glossary deliberately leaves common terms to memory.** Terms used in more than
   1% of corpus papers, and common English words, were excluded: *nitrate* (12% of
   papers), *ecosystem* (35%), *biomass* (27%), *sediment* (13%), *isotope* (5%),
   *hectare* (3%), *herbivore* (2%). That was right for "use the glossary for rare
   terms", and is the opposite of the new goal.
3. **No example ever lacks an entry the answer needs**, so the model never learns what to
   do then: it fills the gap from memory (the "pika is a rodent" error).
4. **Entities are real and recur across papers**, so the model can memorise them.
5. **A third of the answers are Astra's,** a different style that was never measured
   with the current benchmark.

## v3: the changes

### 1. A kid-knowledge boundary
- **Kid lexicon:** words usually learned before 12 (Kuperman et al. 2012), plus a
  curated list of ~300 school-science concepts (cell, atom, gravity, food chain, …).
- **Content terms vs. academic wording:** terms that name a *thing* (species, chemical,
  unit, place, method, measured quantity, scientific concept) need facts; general
  academic words (*robust, implementation, paradigm*) only need rewording and never get
  a glossary entry or a marker.

### 2. A full-coverage glossary
- Every content term outside the kid lexicon gets an entry, with **no corpus-frequency
  or common-word exemption**. Abbreviations still take their meaning from the paper.
- Each entry gets a short **category** ("a small mammal", "a chemical compound", "a
  statistical method", "a unit of area"), from the first sentence of the reference page
  (Wikidata "instance of" later). The category is what licenses kid-style reasoning.

### 3. Answers grounded in P + G + K (one Opus call per conversation)
A **grounding edit** of the existing Opus answer, not a rewrite:
- Opus gets the original section, the current answer, the glossary for this example, and
  the list of terms whose entry is *withheld* in this example (see 4).
- It may edit only sentences that mention or explain a content term, or that add facts
  from outside P + G + K; every other sentence must come back unchanged (checked by
  diff).
- **Terms with an entry:** explained at first use, woven into the sentence, based on the
  entry and K only.
- **Terms without an entry:** written as a **marker** at first use, with no explanation:
  `⟦Oxytropis⟧`.
- **Outside facts** (background knowledge Opus added that is not in P, G or K): removed
  or turned into a marker.
- It returns the answer plus a list: for each term, its first use, how it was handled
  (explained / marker / already known), and the source of each explanation (P, G or K).
  That list drives the training weights (step 7) and the checks (step 8).

### 4. Glossary dropout (teaches the marker)
Per example, a random share of entries (0–40%, chosen before the edit) is withheld from
the input, and the edit writes markers for exactly those terms. About 10% of examples get
**no glossary at all**: every outside term becomes a marker. Because Opus writes the
marker version itself, the prose stays natural in both cases.

### 5. Memory-proof entities
In ~25% of papers, specific entities are **renamed consistently** to invented,
pronounceable names in the paper, the glossary and the answer (species, chemicals,
drugs, named methods; countries and well-known places stay real). The glossary entry
keeps the true category and the paper's facts: "*tarnek*: a small mammal of high
grasslands, related to rabbits". Memory of the real thing cannot help; only the glossary
can. Invented names are checked to exist neither in English nor as Wikipedia titles.

A small share (~5%) of renamed entries get an **unusual but harmless property** that the
answer then relies on, so "trust the glossary over expectations" is rewarded. Since the
names are invented, no false statement about a real thing is ever trained.

### 6. One teacher
Opus answers only (13,877 conversations), edited as above. Astra's answers are left out:
another style, never measured, and they would need the same edit anyway. The `writer`
input stays (always "opus") so the student functions keep their interface.

### 7. Training
- Same next-token training, plus **extra weight (×3) on the explanation phrases and the
  markers** (from the edit's term list); every other token is scored as before, so the
  style learned so far is not pushed around.
- 4B: continue from the current adapter, gentle learning rate. 0.8B: from the original
  weights (as the glossary runs showed, a fresh start learns a new input habit better).
- Same 40 validation papers, transformed the same way.

### 8. How we will know it works
Extend `term_check.py` and add two test sets:
- **Grounding:** for every explanation, its source (P / G / K / outside). Target:
  *outside* ≈ 0.
- **Marker accuracy:** with entries withheld, the share of those terms that get a marker
  rather than an explanation (recall), and markers placed where an entry *was* given
  (false alarms).
- **Renamed-entity test:** papers with invented names; any fact about an invented entity
  not in the glossary is a confabulation by definition.
- **Weaving:** with full glossaries, the share of outside terms explained at first use.
  Target: Opus's level or better (78% today).
- **Style must hold:** the judge's "pleasant" and "structure" scores, words per sentence,
  word-age distribution.

## Build plan
1. Kid lexicon + content-term detector + glossary v3 with categories (local, minutes).
2. Renaming script for 25% of papers (local).
3. **Pilot:** 50 papers end to end (~250 Opus calls), read 10 by hand, measure with 8.
4. Fix what the pilot shows, then scale: ~14,000 Opus calls in batches over 2–3 weekly
   allowances, or a 1,000-paper set first (~5,000 calls).
5. `prepare.py --v3` writes the conversations with loss-weight masks; `train.py` reads them.

## Open choices
- Marker form: `⟦term⟧` (proposed; rare characters, easy to find and replace).
- Astra out (proposed) or edited too.
- Rename share 25%, and whether minor places are renamed.
- Size of the first build: 1,000 papers or all 4,416.
# Training set v3: a writer that knows nothing a 12-year-old doesn't know

Status: design, not built (2026-10-04).

## Goal

A small student model should spend its weights on *writing*: understanding the paper,
composing plain, warm, well-ordered text, and reasoning the way a kid does ("a pika is a
mammal, so it breathes air, has fur, feeds its young milk"). It should never draw on
specialist memory. Every fact it writes must come from one of three sources:

| source | what |
|---|---|
| **P** the paper | the section being rewritten (and the rewritten opening) |
| **G** the glossary | reference entries given in the input |
| **K** kid knowledge | words usually learned before 12, school-level science, and general traits of a category ("animals eat", "acids are sour") |

When an explanation would need anything else, the model writes a **marker** instead of
explaining from memory, so the term can be explained afterwards (a lookup, a bigger
model, or a tap-to-explain in the reader). And the training data must make it impossible
to succeed by remembering.

## What v1 and v2 teach today (measured 2026-10-04)

| | v1 `examples.parquet` | v2 `examples-glossary.parquet` |
|---|---|---|
| papers / training conversations | 3,159 / 14,894 | 4,416 / 20,874 |
| answers by | Opus 7,897 · Astra 6,997 | Opus 13,877 · Astra 6,997 |
| glossary in the input | none | median 20 entries per conversation |
| answers written with the glossary? | — | **no**: same Opus/Astra answers as before |

Why the models still explain from memory:
1. **The answers were written without the glossary.** Their explanations come from
   Opus's own knowledge; the training score rewards reproducing them, so reading the
   glossary earns nothing.
2. **The glossary deliberately leaves common terms to memory.** Terms used in more than
   1% of corpus papers, and common English words, were excluded: *nitrate* (12% of
   papers), *ecosystem* (35%), *biomass* (27%), *sediment* (13%), *isotope* (5%),
   *hectare* (3%), *herbivore* (2%). That was right for "use the glossary for rare
   terms", and is the opposite of the new goal.
3. **No example ever lacks an entry the answer needs**, so the model never learns what to
   do then: it fills the gap from memory (the "pika is a rodent" error).
4. **Entities are real and recur across papers**, so the model can memorise them.
5. **A third of the answers are Astra's,** a different style that was never measured
   with the current benchmark.

## v3: the changes

### 1. A kid-knowledge boundary
- **Kid lexicon:** words usually learned before 12 (Kuperman et al. 2012), plus a
  curated list of ~300 school-science concepts (cell, atom, gravity, food chain, …).
- **Content terms vs. academic wording:** terms that name a *thing* (species, chemical,
  unit, place, method, measured quantity, scientific concept) need facts; general
  academic words (*robust, implementation, paradigm*) only need rewording and never get
  a glossary entry or a marker.

### 2. A full-coverage glossary
- Every content term outside the kid lexicon gets an entry, with **no corpus-frequency
  or common-word exemption**. Abbreviations still take their meaning from the paper.
- Each entry gets a short **category** ("a small mammal", "a chemical compound", "a
  statistical method", "a unit of area"), from the first sentence of the reference page
  (Wikidata "instance of" later). The category is what licenses kid-style reasoning.

### 3. Answers grounded in P + G + K (one Opus call per conversation)
A **grounding edit** of the existing Opus answer, not a rewrite:
- Opus gets the original section, the current answer, the glossary for this example, and
  the list of terms whose entry is *withheld* in this example (see 4).
- It may edit only sentences that mention or explain a content term, or that add facts
  from outside P + G + K; every other sentence must come back unchanged (checked by
  diff).
- **Terms with an entry:** explained at first use, woven into the sentence, based on the
  entry and K only.
- **Terms without an entry:** written as a **marker** at first use, with no explanation:
  `⟦Oxytropis⟧`.
- **Outside facts** (background knowledge Opus added that is not in P, G or K): removed
  or turned into a marker.
- It returns the answer plus a list: for each term, its first use, how it was handled
  (explained / marker / already known), and the source of each explanation (P, G or K).
  That list drives the training weights (step 7) and the checks (step 8).

### 4. Glossary dropout (teaches the marker)
Per example, a random share of entries (0–40%, chosen before the edit) is withheld from
the input, and the edit writes markers for exactly those terms. About 10% of examples get
**no glossary at all**: every outside term becomes a marker. Because Opus writes the
marker version itself, the prose stays natural in both cases.

### 5. Memory-proof entities
In ~25% of papers, specific entities are **renamed consistently** to invented,
pronounceable names in the paper, the glossary and the answer (species, chemicals,
drugs, named methods; countries and well-known places stay real). The glossary entry
keeps the true category and the paper's facts: "*tarnek*: a small mammal of high
grasslands, related to rabbits". Memory of the real thing cannot help; only the glossary
can. Invented names are checked to exist neither in English nor as Wikipedia titles.

A small share (~5%) of renamed entries get an **unusual but harmless property** that the
answer then relies on, so "trust the glossary over expectations" is rewarded. Since the
names are invented, no false statement about a real thing is ever trained.

### 6. One teacher
Opus answers only (13,877 conversations), edited as above. Astra's answers are left out:
another style, never measured, and they would need the same edit anyway. The `writer`
input stays (always "opus") so the student functions keep their interface.

### 7. Training
- Same next-token training, plus **extra weight (×3) on the explanation phrases and the
  markers** (from the edit's term list); every other token is scored as before, so the
  style learned so far is not pushed around.
- 4B: continue from the current adapter, gentle learning rate. 0.8B: from the original
  weights (as the glossary runs showed, a fresh start learns a new input habit better).
- Same 40 validation papers, transformed the same way.

### 8. How we will know it works
Extend `term_check.py` and add two test sets:
- **Grounding:** for every explanation, its source (P / G / K / outside). Target:
  *outside* ≈ 0.
- **Marker accuracy:** with entries withheld, the share of those terms that get a marker
  rather than an explanation (recall), and markers placed where an entry *was* given
  (false alarms).
- **Renamed-entity test:** papers with invented names; any fact about an invented entity
  not in the glossary is a confabulation by definition.
- **Weaving:** with full glossaries, the share of outside terms explained at first use.
  Target: Opus's level or better (78% today).
- **Style must hold:** the judge's "pleasant" and "structure" scores, words per sentence,
  word-age distribution.

## Build plan
1. Kid lexicon + content-term detector + glossary v3 with categories (local, minutes).
2. Renaming script for 25% of papers (local).
3. **Pilot:** 50 papers end to end (~250 Opus calls), read 10 by hand, measure with 8.
4. Fix what the pilot shows, then scale: ~14,000 Opus calls in batches over 2–3 weekly
   allowances, or a 1,000-paper set first (~5,000 calls).
5. `prepare.py --v3` writes the conversations with loss-weight masks; `train.py` reads them.

## Open choices
- Marker form: `⟦term⟧` (proposed; rare characters, easy to find and replace).
- Astra out (proposed) or edited too.
- Rename share 25%, and whether minor places are renamed.
- Size of the first build: 1,000 papers or all 4,416.
# Training set v3: a writer that knows nothing a 12-year-old doesn't know

Status: design, not built (2026-10-04).

## Goal

A small student model should spend its weights on *writing*: understanding the paper,
composing plain, warm, well-ordered text, and reasoning the way a kid does ("a pika is a
mammal, so it breathes air, has fur, feeds its young milk"). It should never draw on
specialist memory. Every fact it writes must come from one of three sources:

| source | what |
|---|---|
| **P** the paper | the section being rewritten (and the rewritten opening) |
| **G** the glossary | reference entries given in the input |
| **K** kid knowledge | words usually learned before 12, school-level science, and general traits of a category ("animals eat", "acids are sour") |

When an explanation would need anything else, the model writes a **marker** instead of
explaining from memory, so the term can be explained afterwards (a lookup, a bigger
model, or a tap-to-explain in the reader). And the training data must make it impossible
to succeed by remembering.

## What v1 and v2 teach today (measured 2026-10-04)

| | v1 `examples.parquet` | v2 `examples-glossary.parquet` |
|---|---|---|
| papers / training conversations | 3,159 / 14,894 | 4,416 / 20,874 |
| answers by | Opus 7,897 · Astra 6,997 | Opus 13,877 · Astra 6,997 |
| glossary in the input | none | median 20 entries per conversation |
| answers written with the glossary? | — | **no**: same Opus/Astra answers as before |

Why the models still explain from memory:
1. **The answers were written without the glossary.** Their explanations come from
   Opus's own knowledge; the training score rewards reproducing them, so reading the
   glossary earns nothing.
2. **The glossary deliberately leaves common terms to memory.** Terms used in more than
   1% of corpus papers, and common English words, were excluded: *nitrate* (12% of
   papers), *ecosystem* (35%), *biomass* (27%), *sediment* (13%), *isotope* (5%),
   *hectare* (3%), *herbivore* (2%). That was right for "use the glossary for rare
   terms", and is the opposite of the new goal.
3. **No example ever lacks an entry the answer needs**, so the model never learns what to
   do then: it fills the gap from memory (the "pika is a rodent" error).
4. **Entities are real and recur across papers**, so the model can memorise them.
5. **A third of the answers are Astra's,** a different style that was never measured
   with the current benchmark.

## v3: the changes

### 1. A kid-knowledge boundary
- **Kid lexicon:** words usually learned before 12 (Kuperman et al. 2012), plus a
  curated list of ~300 school-science concepts (cell, atom, gravity, food chain, …).
- **Content terms vs. academic wording:** terms that name a *thing* (species, chemical,
  unit, place, method, measured quantity, scientific concept) need facts; general
  academic words (*robust, implementation, paradigm*) only need rewording and never get
  a glossary entry or a marker.

### 2. A full-coverage glossary
- Every content term outside the kid lexicon gets an entry, with **no corpus-frequency
  or common-word exemption**. Abbreviations still take their meaning from the paper.
- Each entry gets a short **category** ("a small mammal", "a chemical compound", "a
  statistical method", "a unit of area"), from the first sentence of the reference page
  (Wikidata "instance of" later). The category is what licenses kid-style reasoning.

### 3. Answers grounded in P + G + K (one Opus call per conversation)
A **grounding edit** of the existing Opus answer, not a rewrite:
- Opus gets the original section, the current answer, the glossary for this example, and
  the list of terms whose entry is *withheld* in this example (see 4).
- It may edit only sentences that mention or explain a content term, or that add facts
  from outside P + G + K; every other sentence must come back unchanged (checked by
  diff).
- **Terms with an entry:** explained at first use, woven into the sentence, based on the
  entry and K only.
- **Terms without an entry:** written as a **marker** at first use, with no explanation:
  `⟦Oxytropis⟧`.
- **Outside facts** (background knowledge Opus added that is not in P, G or K): removed
  or turned into a marker.
- It returns the answer plus a list: for each term, its first use, how it was handled
  (explained / marker / already known), and the source of each explanation (P, G or K).
  That list drives the training weights (step 7) and the checks (step 8).

### 4. Glossary dropout (teaches the marker)
Per example, a random share of entries (0–40%, chosen before the edit) is withheld from
the input, and the edit writes markers for exactly those terms. About 10% of examples get
**no glossary at all**: every outside term becomes a marker. Because Opus writes the
marker version itself, the prose stays natural in both cases.

### 5. Memory-proof entities
In ~25% of papers, specific entities are **renamed consistently** to invented,
pronounceable names in the paper, the glossary and the answer (species, chemicals,
drugs, named methods; countries and well-known places stay real). The glossary entry
keeps the true category and the paper's facts: "*tarnek*: a small mammal of high
grasslands, related to rabbits". Memory of the real thing cannot help; only the glossary
can. Invented names are checked to exist neither in English nor as Wikipedia titles.

A small share (~5%) of renamed entries get an **unusual but harmless property** that the
answer then relies on, so "trust the glossary over expectations" is rewarded. Since the
names are invented, no false statement about a real thing is ever trained.

### 6. One teacher
Opus answers only (13,877 conversations), edited as above. Astra's answers are left out:
another style, never measured, and they would need the same edit anyway. The `writer`
input stays (always "opus") so the student functions keep their interface.

### 7. Training
- Same next-token training, plus **extra weight (×3) on the explanation phrases and the
  markers** (from the edit's term list); every other token is scored as before, so the
  style learned so far is not pushed around.
- 4B: continue from the current adapter, gentle learning rate. 0.8B: from the original
  weights (as the glossary runs showed, a fresh start learns a new input habit better).
- Same 40 validation papers, transformed the same way.

### 8. How we will know it works
Extend `term_check.py` and add two test sets:
- **Grounding:** for every explanation, its source (P / G / K / outside). Target:
  *outside* ≈ 0.
- **Marker accuracy:** with entries withheld, the share of those terms that get a marker
  rather than an explanation (recall), and markers placed where an entry *was* given
  (false alarms).
- **Renamed-entity test:** papers with invented names; any fact about an invented entity
  not in the glossary is a confabulation by definition.
- **Weaving:** with full glossaries, the share of outside terms explained at first use.
  Target: Opus's level or better (78% today).
- **Style must hold:** the judge's "pleasant" and "structure" scores, words per sentence,
  word-age distribution.

## Build plan
1. Kid lexicon + content-term detector + glossary v3 with categories (local, minutes).
2. Renaming script for 25% of papers (local).
3. **Pilot:** 50 papers end to end (~250 Opus calls), read 10 by hand, measure with 8.
4. Fix what the pilot shows, then scale: ~14,000 Opus calls in batches over 2–3 weekly
   allowances, or a 1,000-paper set first (~5,000 calls).
5. `prepare.py --v3` writes the conversations with loss-weight masks; `train.py` reads them.

## Open choices
- Marker form: `⟦term⟧` (proposed; rare characters, easy to find and replace).
- Astra out (proposed) or edited too.
- Rename share 25%, and whether minor places are renamed.
- Size of the first build: 1,000 papers or all 4,416.
# Training set v3: a writer that knows nothing a 12-year-old doesn't know

Status: design, not built (2026-10-04).

## Goal

A small student model should spend its weights on *writing*: understanding the paper,
composing plain, warm, well-ordered text, and reasoning the way a kid does ("a pika is a
mammal, so it breathes air, has fur, feeds its young milk"). It should never draw on
specialist memory. Every fact it writes must come from one of three sources:

| source | what |
|---|---|
| **P** the paper | the section being rewritten (and the rewritten opening) |
| **G** the glossary | reference entries given in the input |
| **K** kid knowledge | words usually learned before 12, school-level science, and general traits of a category ("animals eat", "acids are sour") |

When an explanation would need anything else, the model writes a **marker** instead of
explaining from memory, so the term can be explained afterwards (a lookup, a bigger
model, or a tap-to-explain in the reader). And the training data must make it impossible
to succeed by remembering.

## What v1 and v2 teach today (measured 2026-10-04)

| | v1 `examples.parquet` | v2 `examples-glossary.parquet` |
|---|---|---|
| papers / training conversations | 3,159 / 14,894 | 4,416 / 20,874 |
| answers by | Opus 7,897 · Astra 6,997 | Opus 13,877 · Astra 6,997 |
| glossary in the input | none | median 20 entries per conversation |
| answers written with the glossary? | — | **no**: same Opus/Astra answers as before |

Why the models still explain from memory:
1. **The answers were written without the glossary.** Their explanations come from
   Opus's own knowledge; the training score rewards reproducing them, so reading the
   glossary earns nothing.
2. **The glossary deliberately leaves common terms to memory.** Terms used in more than
   1% of corpus papers, and common English words, were excluded: *nitrate* (12% of
   papers), *ecosystem* (35%), *biomass* (27%), *sediment* (13%), *isotope* (5%),
   *hectare* (3%), *herbivore* (2%). That was right for "use the glossary for rare
   terms", and is the opposite of the new goal.
3. **No example ever lacks an entry the answer needs**, so the model never learns what to
   do then: it fills the gap from memory (the "pika is a rodent" error).
4. **Entities are real and recur across papers**, so the model can memorise them.
5. **A third of the answers are Astra's,** a different style that was never measured
   with the current benchmark.

## v3: the changes

### 1. A kid-knowledge boundary
- **Kid lexicon:** words usually learned before 12 (Kuperman et al. 2012), plus a
  curated list of ~300 school-science concepts (cell, atom, gravity, food chain, …).
- **Content terms vs. academic wording:** terms that name a *thing* (species, chemical,
  unit, place, method, measured quantity, scientific concept) need facts; general
  academic words (*robust, implementation, paradigm*) only need rewording and never get
  a glossary entry or a marker.

### 2. A full-coverage glossary
- Every content term outside the kid lexicon gets an entry, with **no corpus-frequency
  or common-word exemption**. Abbreviations still take their meaning from the paper.
- Each entry gets a short **category** ("a small mammal", "a chemical compound", "a
  statistical method", "a unit of area"), from the first sentence of the reference page
  (Wikidata "instance of" later). The category is what licenses kid-style reasoning.

### 3. Answers grounded in P + G + K (one Opus call per conversation)
A **grounding edit** of the existing Opus answer, not a rewrite:
- Opus gets the original section, the current answer, the glossary for this example, and
  the list of terms whose entry is *withheld* in this example (see 4).
- It may edit only sentences that mention or explain a content term, or that add facts
  from outside P + G + K; every other sentence must come back unchanged (checked by
  diff).
- **Terms with an entry:** explained at first use, woven into the sentence, based on the
  entry and K only.
- **Terms without an entry:** written as a **marker** at first use, with no explanation:
  `⟦Oxytropis⟧`.
- **Outside facts** (background knowledge Opus added that is not in P, G or K): removed
  or turned into a marker.
- It returns the answer plus a list: for each term, its first use, how it was handled
  (explained / marker / already known), and the source of each explanation (P, G or K).
  That list drives the training weights (step 7) and the checks (step 8).

### 4. Glossary dropout (teaches the marker)
Per example, a random share of entries (0–40%, chosen before the edit) is withheld from
the input, and the edit writes markers for exactly those terms. About 10% of examples get
**no glossary at all**: every outside term becomes a marker. Because Opus writes the
marker version itself, the prose stays natural in both cases.

### 5. Memory-proof entities
In ~25% of papers, specific entities are **renamed consistently** to invented,
pronounceable names in the paper, the glossary and the answer (species, chemicals,
drugs, named methods; countries and well-known places stay real). The glossary entry
keeps the true category and the paper's facts: "*tarnek*: a small mammal of high
grasslands, related to rabbits". Memory of the real thing cannot help; only the glossary
can. Invented names are checked to exist neither in English nor as Wikipedia titles.

A small share (~5%) of renamed entries get an **unusual but harmless property** that the
answer then relies on, so "trust the glossary over expectations" is rewarded. Since the
names are invented, no false statement about a real thing is ever trained.

### 6. One teacher
Opus answers only (13,877 conversations), edited as above. Astra's answers are left out:
another style, never measured, and they would need the same edit anyway. The `writer`
input stays (always "opus") so the student functions keep their interface.

### 7. Training
- Same next-token training, plus **extra weight (×3) on the explanation phrases and the
  markers** (from the edit's term list); every other token is scored as before, so the
  style learned so far is not pushed around.
- 4B: continue from the current adapter, gentle learning rate. 0.8B: from the original
  weights (as the glossary runs showed, a fresh start learns a new input habit better).
- Same 40 validation papers, transformed the same way.

### 8. How we will know it works
Extend `term_check.py` and add two test sets:
- **Grounding:** for every explanation, its source (P / G / K / outside). Target:
  *outside* ≈ 0.
- **Marker accuracy:** with entries withheld, the share of those terms that get a marker
  rather than an explanation (recall), and markers placed where an entry *was* given
  (false alarms).
- **Renamed-entity test:** papers with invented names; any fact about an invented entity
  not in the glossary is a confabulation by definition.
- **Weaving:** with full glossaries, the share of outside terms explained at first use.
  Target: Opus's level or better (78% today).
- **Style must hold:** the judge's "pleasant" and "structure" scores, words per sentence,
  word-age distribution.

## Build plan
1. Kid lexicon + content-term detector + glossary v3 with categories (local, minutes).
2. Renaming script for 25% of papers (local).
3. **Pilot:** 50 papers end to end (~250 Opus calls), read 10 by hand, measure with 8.
4. Fix what the pilot shows, then scale: ~14,000 Opus calls in batches over 2–3 weekly
   allowances, or a 1,000-paper set first (~5,000 calls).
5. `prepare.py --v3` writes the conversations with loss-weight masks; `train.py` reads them.

## Open choices
- Marker form: `⟦term⟧` (proposed; rare characters, easy to find and replace).
- Astra out (proposed) or edited too.
- Rename share 25%, and whether minor places are renamed.
- Size of the first build: 1,000 papers or all 4,416.
