---
name: science-search
description: Search ~200 million scientific papers (OpenAlex titles and abstracts) by meaning or by exact words, look papers up by DOI / OpenAlex id / PMCID, find similar papers and walk citations, through the free ScienceMadeReadable search API (no key). Use when a task needs scientific literature - finding studies or evidence on a question, checking what research says about a claim, a literature review, citing papers, resolving a DOI to its title, authors and abstract, building a reading list, or writing code that calls this API.
---

# Science search (ScienceMadeReadable)

Read-only JSON over HTTPS, free, no key. Base: `https://api.sciencemadereadable.com/api/data`

Full guide, with today's coverage, every parameter, field meanings and limits: `GET <base>/llms.txt`.
Read it before the first search in a conversation. Interactive reference: `<base>/docs`; spec: `<base>/openapi.json`.

## The main calls

```sh
B=https://api.sciencemadereadable.com/api/data
curl -sG "$B/search" --data-urlencode "q=effect of intermittent fasting on weight loss in adults" -d k=20 -d abstracts=false
curl -sG "$B/search" --data-urlencode 'q="gut microbiome" AND (autism OR ASD) NOT mice' -d k=20 -d abstracts=false
curl -s  "$B/works/10.1126/science.aan8048"           # DOI, W123..., PMC123..., or a doi.org / openalex.org URL
curl -s  "$B/similar/W2782456798?k=10&abstracts=false"
curl -s  "$B/walk?id=10.1126/science.aan8048"          # 'answers' = what similar papers cite and this one doesn't
```

Results: `id` (OpenAlex W-number), `doi`, `title`, `year`, `type`, `authors` (first 12, " | "), `venue`,
`citations`, `abstract`, `score`, `junk` (0-1, likelihood of spam or not scholarship), `openalex`, `pmcid`.
Filters: `year_from`, `year_to`, `type=review|article|preprint|...`, `open_access=true`, `license=cc-by`.

## Doing it well

- Questions and descriptions: search by meaning (the default). Exact names, genes, drugs, phrases, "everything
  mentioning X": quotes / AND / OR / NOT (switches to exact words by itself). For a review, run both and merge by `id`.
- Rewrite casual questions into research language first ("how to lose weight" -> "effectiveness of dietary and
  exercise interventions for weight loss in adults"). It is the biggest quality gain.
- Screen with `abstracts=false`, then fetch `/works/{id}` only for the papers you keep. Snowball with `/similar`
  and `/walk`.
- English titles and abstracts only, no full texts. Search by meaning may not reach every year; `/llms.txt` says
  how far. Exact words and `/works` cover the whole corpus.
- `topic` / `field` labels are often wrong: judge a paper from its title and abstract. A low `junk` score does
  not mean a paper is right, and citation counts favour older papers.
- Cite the DOI when there is one, else the `openalex` URL. Never invent details a record lacks.

## Limits

- 60 requests a minute per address; HTTP 429 means wait a minute. Ask for up to `k=100` at once instead of
  looping, and do not run searches in parallel.
- 502/503 for a minute or two: a data update or a restart. Retry later.
- Requests, including search words, appear in the server's request log, which does not record the caller's
  address.

To point a person to a paper in a readable page: `https://sciencemadereadable.com/?work=<DOI>`.
