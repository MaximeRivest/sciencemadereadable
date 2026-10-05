"""What the reader knows: one definition, shared by the training-data edit
(training/v3/build.py), the term checker (rewrite_benchmark/term_check.py)
and the readability metrics (rewrite_benchmark/readability.py).

The reader: a curious 14-year-old (the top of our 12-14 target), no science background.
- Ordinary wording (robust, comprehensive, implementation) counts as known when it is
  usually learned before READER_AGE (Kuperman et al. 2012 age-of-acquisition ratings).
  Those words never need an explanation; plainer words are still welcome.
- Science terms (species, chemicals, units, methods, statistics terms, measured
  quantities) are judged by what they are, not by their age rating: they need an
  explanation or a marker unless they are in SCHOOL_SCIENCE.
"""
READER_AGE = 14

SCHOOL_SCIENCE = sorted(set("""
atom atoms molecule molecules cell cells bacteria bacterium germ germs microbe microbes microorganism
microorganisms virus viruses gene genes dna protein proteins enzyme? nutrient nutrients energy force
gravity temperature oxygen carbon nitrogen hydrogen water climate weather species habitat ecosystem
ecosystems food chain food web plant plants animal animals mammal mammals bird birds reptile reptiles
amphibian amphibians fish insect insects predator predators prey photosynthesis fossil fossils soil rock
rocks mineral minerals volcano earthquake planet orbit acid acids electricity magnet microscope
experiment experiments data graph graphs average percent percentage measure measurement sample
temperature pollution pollutant evolution extinct extinction population organism organisms
""".split()) - {"enzyme?"})

READER = ("a curious 14-year-old who reads English well and has school-level science (they know: "
          + ", ".join(SCHOOL_SCIENCE) + "), but no specialist background. Ordinary words a 14-year-old "
          "knows need no explanation. Science terms (species, chemicals, units, methods, statistics terms, "
          "measured quantities) need an explanation unless they are in that list.")
