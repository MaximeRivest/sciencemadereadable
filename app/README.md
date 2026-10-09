# Science made readable · the app

**Live: https://sciencemadereadable.com**, served by GitHub Pages from the `gh-pages` branch of this
repository (DNS at GoDaddy: apex A/AAAA → GitHub Pages, www → maximerivest.github.io). Its back end is this
machine, reached at **https://api.sciencemadereadable.com**: DNS → the OVH relay (`encrypted-link-relay`,
144.217.95.30, also Chattering Anywhere's), whose Caddy (`/etc/caddy/conf.d/sciencemadereadable.caddy`, Let's
Encrypt, no access log) proxies to 127.0.0.1:18799 there, the end of an outbound SSH tunnel lambda keeps open
(user service `smr-public-tunnel`, key `~/.ssh/smr-tunnel_ed25519`; on the relay the account `smr-tunnel` may only
listen on that one address, `sshd_config.d/10-smr-tunnel.conf`). The relay cannot open connections into lambda
(tailnet policy); lambda reaches out. `server.py`'s public port 8799 answers: queue, saved rewrites, examples,
counts, and `/api/data/*` (search, map, works; srl_search on :8810). The old Funnel door
(https://lambda.tail69222b.ts.net:10000) still serves the rented-GPU worker scripts. Publish a new version of the
page (from the repository root; `SITE` is a checkout of the `gh-pages` branch):

```
(cd app && node tools/build.mjs) && .venv/bin/python app/tools/export_site.py https://api.sciencemadereadable.com sciencemadereadable.com
rsync -a --delete --exclude .git app/site/ "$SITE"/ && git -C "$SITE" add -A && git -C "$SITE" commit -m update && git -C "$SITE" push
```

## Where the models run

Since 2026-10-05 evening, **our 9B on lambda's GPU 0**, in InkType's place, declared in
`~/Projects/os/machines` so it survives reboots: `model-vllm-smr-9b` (system service, the pinned Hugging
Face version, vLLM 0.29.0, localhost :8015), and the user services `smr-server` (the queue) and
`smr-worker-home` (3 papers at once). `app/rent/home_gpus.sh status|on|off` switches them for now; to give
GPU 0 back for good, set `inktypePaused = false` in `hosts/lambda/ai-services.nix` and remove `smr-9b`.
The model router never parks it; the big on-demand models refuse to start while it runs.

For launch days, a rented H100 as a second worker (Nebius, about $4.50/hour), with the 0.8B too:

```
app/rent/h100_up.sh      # rent it, start the 9B and the 0.8B there, its worker on lambda. About 10 minutes.
app/rent/h100_down.sh    # delete the machine and its disk: billing stops. The home 9B keeps going.
```

Each worker offers only the models that answer (checked every 15 s), and the queue gives a paper only to
a worker that has its model: a restarting model makes papers wait in line, not fail.

## Parts

| | where | what |
|---|---|---|
| the page (`web/`, built to `web/app.js`) | static: GitHub Pages, or `server.py` | search (Europe PMC), the paper (Europe PMC XML, split by `web/src/jats.ts`), figures (PubMed Central's open copy on AWS), the reader, big models with the reader's key |
| the queue (`server.py`) | must be reachable by the page | jobs for our models, saved rewrites, examples |
| the paper service (`server.py /api/paper`) | lambda, local only | sections, reference glossary (offline Wikipedia), reply limits, for the worker |
| the worker (`worker/worker.ts`) | lambda (it can run anywhere) | takes jobs, runs our model (vLLM) with the scored requests, streams progress back |

```
cd app && npm install && node tools/build.mjs         # postinstall links lm15 to functai's own copy
.venv/bin/python app/server.py                        # service smr-server on lambda (port 8795)
cd app && npm run worker                               # service smr-worker-home, or tmux "smr-worker-h100";
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

## Support

The support window (♥ Support; shareable as `?support`): amounts, what the GPU costs a day, and the
support of the last 24 hours. Settings in `support.json` (not published; `support.example.json`).

- **Stripe** (card, Apple Pay, Google Pay, no account): one payment link per amount, $3 a month, any
  amount, and "sponsor a day" ($110, asks the name to show and the day). Made by
  `STRIPE_SETUP_KEY=rk_... .venv/bin/python app/tools/stripe_setup.py` (a restricted key with write
  access to Products, Prices, Payment Links; not saved; delete it afterwards). Test links (`rk_test_`)
  appear only on the private port; the public door and the GitHub Pages copy show live links only.
  After paying, Stripe sends the supporter to `/?thanks`.
- **Totals:** the queue reads Stripe every 5 minutes with a read-only restricted key in
  `stripe_read_key` (Checkout Sessions: read; Subscriptions: read), and GitHub Sponsors with this
  machine's `gh` login. Only totals leave lambda (plus GitHub sponsors who chose to be public).
- **Sponsor of the day:** a paid day waits in `sponsors.json` until approved:
  `.venv/bin/python app/tools/sponsor_day.py` (list), `... approve cs_... [DAY]`, `... refuse cs_...`.
  Approved names go into `support.json`'s `sponsors_by_day` and show that day only.
- `daily_cost` in `support.json`: what a rented H100 day costs (108), the unit the window speaks in.

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
