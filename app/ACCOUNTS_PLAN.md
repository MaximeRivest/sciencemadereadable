# Accounts, API keys and the paid plan

Built 2026-10-16. Code: `accounts.py` (accounts, keys, plans, Stripe, read-only), `server.py` (the door: limits per
key, `/api/account/*`), `web/src/account.ts` (the page, `?account`), `web/terms.html`, `web/privacy.html`,
`tools/stripe_plan_setup.py`, `tools/accounts_admin.py`. Tests: `test_accounts.py` (through the real door, fake
GitHub/Google/Stripe).

## What it does

| | How | Limit | Price |
|---|---|---|---|
| No key | nothing changes | 30/min per address | free |
| Free account | sign in (GitHub or Google), make a key | 60/min per account, from any address | free |
| Paid | Stripe subscription | 300/min, 100,000/month; past that 60/min until the month ends | $10/month (+ tax) |

- Keys: `smr_` + 32 characters, shown once, stored as SHA-256 only, up to 10 per account. Sent as
  `Authorization: Bearer smr_...` or `?key=smr_...` (MCP address for apps that can't set headers). Stripped before
  the search service and from log lines. A wrong or deleted key is refused (401; for MCP a tool error the model
  reads), never silently treated as no key. Other `Authorization` values are ignored, as before.
- Paid traffic: at most 3 paid requests at once (about half of what the search service does: ~11/s one at a time,
  16-19/s at most, measured 2026-10-16); the rest wait up to 20 s, then 503 "retry in 5 s". The free door is never
  held up by paid traffic.
- Sign-in: OAuth with PKCE, state in a 15-minute cookie bound to the browser. Session: 30-day cookie
  (`HttpOnly; Secure; SameSite=Lax; Path=/api/account`) on api.sciencemadereadable.com; the account routes answer
  the site's own origins only, and changes must be JSON POSTs with the site as Origin. Same verified email through
  GitHub and Google = one account.
- Stripe, read-only as for field maps: the page sends the payer to the payment link with
  `client_reference_id=<account id>`; on return (`?account&subscribed=cs_...`) and every minute the door reads
  checkouts; every 5 minutes the plan's subscriptions (cancelled/unpaid: back to free). Customers cancel and change
  cards in Stripe's customer portal. Deleting an account is refused while a Stripe plan is active.
- The 429 messages point to `?account` (free key / paid plan) once a sign-in provider is configured.
- Dev site: `/__dev/login` signs in as a "dev" account (real GitHub/Google sign-in always returns to the public site).

## Launch steps

Done by Maxime (the accounts and money are his):

1. **GitHub sign-in.** https://github.com/settings/applications/new (an *OAuth App*):
   name `ScienceMadeReadable`, homepage `https://sciencemadereadable.com`, callback
   `https://api.sciencemadereadable.com/api/account/callback/github`, no device flow. Generate a client secret.
2. **Google sign-in.** https://console.cloud.google.com → new project → Google Auth Platform:
   - Branding: name `ScienceMadeReadable`, support email, home page `https://sciencemadereadable.com`,
     privacy `https://sciencemadereadable.com/privacy.html`, terms `https://sciencemadereadable.com/terms.html`,
     authorised domain `sciencemadereadable.com`.
   - Audience: External, then *Publish app* (only openid/email/profile: no review needed).
   - Clients → Create → Web application → redirect URI `https://api.sciencemadereadable.com/api/account/callback/google`.
3. Put both in `app/oauth.json` (`chmod 600`; git-ignored; read on each request, no restart):
   `{"github": {"client_id": "...", "client_secret": "..."}, "google": {"client_id": "...", "client_secret": "..."}}`
4. **Stripe.**
   - Finish the business review.
   - Add **Subscriptions: Read** to the read-only restricted key (`app/stripe_read_key`).
   - Stripe Tax: Settings → Tax: origin address, your registrations. (Recommended; or run step 4 without `--tax`.)
   - A temporary restricted key (write: Products, Prices, Payment Links):
     `STRIPE_SETUP_KEY=rk_live_... .venv/bin/python app/tools/stripe_plan_setup.py --tax`, then delete that key.
   - Settings → Billing → Customer portal: allow cancelling and updating the card; activate the login link;
     `.venv/bin/python app/tools/stripe_plan_setup.py --portal https://billing.stripe.com/p/login/...`
5. Check `web/terms.html` and `web/privacy.html` (contact email, "in Canada", refund rule) are what you want.

Then (an agent can do these):

6. Restart the door if not done yet (`app/tools/publish.sh queue`, only while no rewrite runs); check
   `curl https://api.sciencemadereadable.com/api/account` lists both providers.
7. Sign in on the real site, make a key, buy the plan once with your own card, check the page says Paid and a key
   allows 300/min; refund yourself in the Stripe dashboard (within a few minutes the plan goes back to free).
8. Publish the page (`app/tools/publish.sh`).
9. Tell programs and agents: the search guide (`search/web/llms.txt`), the MCP server's instructions
   (`search/srl_search/mcp_server.py`), the skill (`search/skill/SKILL.md`, `app/web/agents/science-search/SKILL.md`),
   the door's `MCP_HELP`/`ASK_HELP`: "optional key for more: https://sciencemadereadable.com/?account".
   Through `search-dev` and `search/dev.sh promote`.

## Looking after it

`app/tools/accounts_admin.py list | show | grant | ungrant | delete | sync | check` (`check`: payments with no
account, two plans at once). A friend or a lab helping out: `grant EMAIL 2027-06-30`.

## Trade-offs taken

- No email/password sign-in: nobody without GitHub or Google can sign up (email links can be added later; they
  need a mail provider).
- A key in the MCP address (`?key=`) is less private than a header (it may sit in an app's settings or logs), but it
  is the only way for apps that can't set headers. The page says so.
- Usage counts are written every 15 s: a restart loses at most 15 s of counting (fair-use counts, not billing).
- Plan changes reach the door within 20 s (cache); a cancellation within 5 minutes (polling, no webhook: no
  public endpoint or Stripe signing secret to keep).
- Stripe Tax computes tax; registering and filing stay with Maxime. A merchant of record (Paddle, Lemon Squeezy)
  would take that on for ~5% a sale; reconsider if sales grow.
- Terms and privacy are written in plain language, not by a lawyer.
