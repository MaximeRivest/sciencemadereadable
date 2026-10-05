# Science made readable · the app

**Live: https://sciencemadereadable.com**, served by GitHub Pages from the `gh-pages` branch of this
repository (DNS at GoDaddy: apex A/AAAA → GitHub Pages, www → maximerivest.github.io). The queue's
public door is Tailscale Funnel on lambda, https://lambda.tail69222b.ts.net:10000 → `server.py`'s public
port 8799 (queue, saved rewrites, examples, counts only). Publish a new version of the page (from the
repository root; `SITE` is a checkout of the `gh-pages` branch):

```
(cd app && node tools/build.mjs) && .venv/bin/python app/tools/export_site.py https://lambda.tail69222b.ts.net:10000 sciencemadereadable.com
rsync -a --delete --exclude .git app/site/ "$SITE"/ && git -C "$SITE" add -A && git -C "$SITE" commit -m update && git -C "$SITE" push
```

## Where the models run

Since launch day (2026-10-05), on one rented H100 (Nebius, about $4.50/hour, $108/day):

```
app/rent/h100_up.sh      # rent it, start the 9B (Hugging Face, fixed version) and the 0.8B, its worker on lambda,
                         # then free lambda's GPUs (home_gpus.sh off). About 10 minutes.
app/rent/h100_down.sh    # delete the machine and its disk: billing stops. The site then says "GPU offline".
app/rent/home_gpus.sh on|off|status   # our models on lambda's 2 x RTX 3090 instead, as before launch
```

Lambda keeps the queue, the paper service and the public door; only the writing moves to the H100.
The worker runs on lambda and reaches the H100 through an SSH tunnel (ports 8016 for the 9B, 8017 for
the 0.8B); the rented machine opens no port to the internet.

Search open research papers and read them rewritten in plain words, next to the original, with the
paper's own figures and tables. Our small models run on our GPU (above); six commercial models run with the
reader's own API key.

On lambda (tailnet only): https://lambda.tail69222b.ts.net:18795/

## Parts

| | where | what |
|---|---|---|
| the page (`web/`, built to `web/app.js`) | static: GitHub Pages, or `server.py` | search (Europe PMC), the paper (Europe PMC XML, split by `web/src/jats.ts`), figures (PubMed Central's open copy on AWS), the reader, big models with the reader's key |
| the queue (`server.py`) | must be reachable by the page | jobs for our models, saved rewrites, examples |
| the paper service (`server.py /api/paper`) | lambda, local only | sections, reference glossary (offline Wikipedia), reply limits, for the worker |
| the worker (`worker/worker.ts`) | lambda (it can run anywhere) | takes jobs, runs our model (vLLM) with the scored requests, streams progress back |

```
cd app && npm install && node tools/build.mjs         # postinstall links lm15 to functai's own copy
.venv/bin/python app/server.py                        # tmux "smr-server" on lambda (port 8795)
cd app && npm run worker                               # tmux "smr-worker" (home GPUs) or "smr-worker-h100";
                                                       # settings: worker/config.json (examples: config*.example.json)
.venv/bin/python app/tools/export_site.py https://QUEUE   # static site for GitHub Pages → app/site/
```

## Same input as the scored runs, everywhere

- `tools/check_prompts.ts`: the TypeScript programs send byte for byte what the Python ones sent (our
  models: the prompt they were trained on). 16/16.
- `tools/check_jats.ts`: the browser's section splitter and licence check give exactly the Python
  corpus code's result. 500/500 corpus papers.
- The worker gets sections and glossary from the Python paper service (corpus papers: the saved
  training glossary; new papers: built the same way from the offline copy, 99% identical).

## Reading view

The rewrite is matched to the original paragraph by paragraph (`web/src/align.ts`, from
`training/side_by_side.py`). Figures and tables are found in the XML and put back where the
paper has them (`jats.ts layout`). Tables are always the publisher's: the models' rewritten tables
are never shown. Figure and table captions are shown rewritten. A section still being written
streams as one block, then snaps to paragraph alignment when it is finished.

## Several readers at once

The worker takes up to `parallel` papers at once (worker/config.json, default 3); vLLM writes them
side by side. On the RTX 3090 the 9B's working memory holds about one paper's text, so sharing
mostly spreads the same speed: measured, 3 papers at once took 150-200 s each (75-110 s alone),
about 1.35 times more papers per hour. Everyone sees text within seconds instead of waiting in
line. More capacity needs more GPU memory: another worker anywhere (it only pulls jobs).

## Saved rewrites

Examples (the 3 benchmark papers, all nine models, the scored rewrites): `tools/seed_examples.py`,
copied into the static site too so they work offline. Our models' rewrites made through the queue
are saved by the queue (only the worker can write them). Rewrites made with a reader's key stay in
their browser.

## Usage counts

`stats.py`, on our own server: no cookies, no third party. A visitor is an anonymous code made
each day from address and browser with a random key that is thrown away the next day (unique
visitors per day, no following anyone across days, addresses never stored). Search words are not
stored. Do Not Track / Global Privacy Control: nothing sent. Events in `stats/events-YYYY-MM.jsonl`;
the dashboard is `/stats`, shown only to tailnet devices (Tailscale adds who is asking; requests
from the internet don't carry it).

## Security

No API key reaches any server of ours. Model text is rendered as Markdown with raw HTML escaped and
sanitized (DOMPurify). Content Security Policy in the page (GitHub Pages can't send headers): own
scripts only; images only from PubMed Central's open copy. Queue: at most 2 jobs per address, 30 in
line; the worker authenticates with `worker_token` (not in the repository).

## functai / lm15 notes (to report upstream)

- functai TS doesn't bundle for the browser: `accounts.ts` imports `TerminalUI`; `tools/build.mjs` swaps in a stand-in.
- lm15 sends `anthropic-dangerous-direct-browser-access` only for Claude Code logins; `pipeline.ts` adds it.
- OpenAI's 401 on `/v1/responses` has no CORS header (looks like a network error); the page checks the key on `/v1/models` first.
