# Scientific article token measurements

Measured: 2026-09-28T09:43:30.628737+00:00. Tokenizer: **o200k_base** (GPT-4o), tiktoken 0.12.0.

## Results

| Article | Without bibliography | With bibliography |
|---|---:|---:|
| [Audit and feedback to change diagnostic image ordering practices: A systematic review and meta-analysis](https://doi.org/10.1371/journal.pone.0300001) | 11,327 | 14,767 |
| [Validity, reliability, and readability of single-item and short physical activity questionnaires for use in surveillance: A systematic review](https://doi.org/10.1371/journal.pone.0300003) | 7,803 | 11,777 |
| [Coping strategies of psychiatrists and psychiatry trainees following patient suicide and suicide attempt: A national cross-sectional study in Saudi Arabia](https://doi.org/10.1371/journal.pone.0300004) | 6,552 | 9,695 |
| [Development and validation of the Michigan Chronic Disease Simulation Model (MICROSIM)](https://doi.org/10.1371/journal.pone.0300005) | 12,134 | 16,239 |
| [LEDT and Idebenone treatment modulate autophagy and improve regenerative capacity in the dystrophic muscle through an AMPK-pathway](https://doi.org/10.1371/journal.pone.0300006) | 12,281 | 16,395 |
| [Beyond the classroom walls: Stakeholder experiences with remote instruction in Post RN baccalaureate nursing program during the COVID-19 pandemic: A qualitative inquiry](https://doi.org/10.1371/journal.pone.0300007) | 10,961 | 15,006 |
| [Recommendations for increasing yield of the edible Pinus pinea L. pine nuts](https://doi.org/10.1371/journal.pone.0300008) | 4,727 | 9,364 |
| [Antidiabetic drug administration prevents bone mineral density loss: Evidence from a two-sample Mendelian randomization study](https://doi.org/10.1371/journal.pone.0300009) | 6,546 | 8,662 |
| [Combination prediction method of students’ performance based on ant colony algorithm](https://doi.org/10.1371/journal.pone.0300010) | 8,300 | 10,804 |
| [HEBE project: Healthy aging versus inflamm-aging: The role of physical exercise in modulating the biomarkers of age-associated and environmentally determined chronic diseases, study protocol](https://doi.org/10.1371/journal.pone.0300011) | 7,020 | 9,839 |

## Summary

- With bibliography: mean **12,255**, median **11,290**, range **8,662–16,395** tokens.
- Without bibliography: mean **8,765**, median **8,052**, range **4,727–12,281** tokens.
- All ten together, with bibliography: **122,548** tokens.

## Method and limits

First ten eligible research-article records in ascending PLOS ONE DOI order starting at 0300001; convenience sample, not random or representative. Reviews may be classified as research-article by the publisher.

**Included:** Title, abstract, body, headings, textual figure/table captions, table cell text, in-text citation markers, and back matter such as acknowledgments. Full version also includes bibliography.

**Excluded:** XML markup, author/affiliation metadata, publisher/license metadata, image pixels, linked supplementary file contents, and any text/equations only available as images. Alternative equation representations are counted once, preferring TeX.

**Normalization:** Structural blocks separated by blank lines; whitespace collapsed within paragraphs; UTF-8 with one trailing newline. Table alternatives prefer actual table text over images. MathML is flattened to its text, so mathematical layout is not fully preserved; these are text-only counts, not multimodal counts. No chat framing or prompt overhead.

These are exact token counts for the saved extracted text—not estimates from word counts. PDF extraction, another tokenizer, or adding images/supplements will change the result. This one-journal sample does not estimate the average length of all scientific articles.

## Reproduce

From the project root:

```sh
uv run article-tokens/measure.py
```

The script reuses saved publisher XML in `xml/`. Exact counted inputs are in `text/`. Detailed provenance and checksums are in `results.json`, with a spreadsheet-friendly copy in `results.csv`.
