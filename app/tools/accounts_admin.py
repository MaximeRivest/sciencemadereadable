"""Look after the API accounts (app/accounts.py, app/accounts.db) by hand.

    .venv/bin/python app/tools/accounts_admin.py list                 every account: plan, keys, requests this month
    .venv/bin/python app/tools/accounts_admin.py show EMAIL|ID        one account in full (never a key itself)
    .venv/bin/python app/tools/accounts_admin.py grant EMAIL|ID 2027-01-31 [--note "why"]
                                                                       the paid plan for free until that day
    .venv/bin/python app/tools/accounts_admin.py ungrant EMAIL|ID
    .venv/bin/python app/tools/accounts_admin.py delete EMAIL|ID      (refused while a Stripe plan is active)
    .venv/bin/python app/tools/accounts_admin.py sync [--days 30]     read the plan's checkouts and subscriptions now
    .venv/bin/python app/tools/accounts_admin.py check                problems: payments with no account, two plans

Safe while the door runs (SQLite, WAL); the door sees a change within 20 seconds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accounts as A


def find(who: str) -> str:
    row = A.q1("SELECT id FROM users WHERE id=? OR lower(email)=lower(?)", (who, who)) \
        or A.q1("SELECT user_id id FROM identities WHERE lower(email)=lower(?) OR lower(login)=lower(?)", (who, who))
    if not row:
        sys.exit(f"no account {who!r}")
    return row["id"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for c in ("show", "ungrant", "delete"):
        sub.add_parser(c).add_argument("who")
    g = sub.add_parser("grant")
    g.add_argument("who")
    g.add_argument("until", help="YYYY-MM-DD")
    g.add_argument("--note", default="")
    s = sub.add_parser("sync")
    s.add_argument("--days", type=int, default=30)
    sub.add_parser("check")
    a = ap.parse_args()

    if a.cmd == "list":
        rows = A.q("SELECT u.id, u.email, u.name, u.created, (SELECT COUNT(*) FROM keys k WHERE k.user_id=u.id AND "
                   "revoked IS NULL) keys, (SELECT GROUP_CONCAT(provider) FROM identities i WHERE i.user_id=u.id) via "
                   "FROM users u ORDER BY u.created")
        for r in rows:
            p = A.plan_of(r["id"])
            print(f"{r['id']}  {(r['email'] or '-'):34} {p['plan']:4} {p['source'] or '':6} keys {r['keys']:2}  "
                  f"month {A.month_used(r['id']):>7,}  {r['via']:13} since {r['created'][:10]}")
        print(f"{len(rows)} accounts")
    elif a.cmd == "show":
        uid = find(a.who)
        d = A.me(uid, private=True)
        d.pop("plans", None), d.pop("offer", None)
        for k, v in d.items():
            print(f"{k:9} {v}")
        for s in A.q("SELECT * FROM subscriptions WHERE user_id=?", (uid,)):
            print("stripe   ", dict(s))
    elif a.cmd == "grant":
        uid = find(a.who)
        A.q("UPDATE users SET grant_until=?, grant_note=? WHERE id=?", (a.until, a.note or None, uid))
        print(f"{uid}: paid plan until {a.until} (end of that day, UTC)")
    elif a.cmd == "ungrant":
        uid = find(a.who)
        A.q("UPDATE users SET grant_until=NULL, grant_note=NULL WHERE id=?", (uid,))
        print(f"{uid}: grant removed")
    elif a.cmd == "delete":
        uid = find(a.who)
        try:
            A.delete_user(uid)
        except A.AccountError as e:
            sys.exit(e.message)
        print(f"{uid}: deleted")
    elif a.cmd == "sync":
        print(f"checkouts read: {A.sync_checkouts(a.days)}, subscriptions updated: {A.refresh_subscriptions()}")
    elif a.cmd == "check":
        bad = 0
        for s in A.q("SELECT * FROM subscriptions WHERE user_id IS NULL AND status IN ('active','trialing','past_due')"):
            bad += 1
            print(f"paying, but no account: {s['id']} (customer {s['customer']}, checkout {s['session']}): find who "
                  f"in Stripe, then UPDATE subscriptions SET user_id=... or refund")
        for r in A.q("SELECT user_id, COUNT(*) n FROM subscriptions WHERE status IN ('active','trialing','past_due') "
                     "AND user_id IS NOT NULL GROUP BY user_id HAVING n > 1"):
            bad += 1
            print(f"two plans at once: {r['user_id']} ({r['n']}): cancel and refund one in Stripe")
        print("all good" if not bad else f"{bad} to look at")


main()
