"""Payments for whole-field maps (app/fieldpay.py): who paid, names waiting to be shown, money owed back.

    .venv/bin/python app/tools/field_payments.py                 # list
    .venv/bin/python app/tools/field_payments.py approve KEY     # show the payer's name on the map
    .venv/bin/python app/tools/field_payments.py refuse KEY      # never show it
    .venv/bin/python app/tools/field_payments.py refunded FILE   # a refund made by hand in Stripe's dashboard
    .venv/bin/python app/tools/field_payments.py remove KEY "why"   # take a topic out of the line (its text is public)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fieldpay as P  # noqa: E402


def main(a):
    files = sorted(P.PAID.glob("*.json"))
    if not a:
        owed = {p for p, _ in P.owed()}
        for p in files:
            m = json.loads(p.read_text())
            job = P._read(P.FIELDS / (m.get("key") or "-") / "job.json") or {}
            flag = ("OWED BACK" if p in owed else f"refund {m['refund']}" if m.get("refund") else "")
            name = m.get("name") and f"name {m['name']!r} {'shown' if m.get('name_ok') else 'WAITING'}"
            print(f"{p.name:46} ${(m.get('amount') or 0) / 100:5.2f} {m.get('paid_at', '')[:16]}  {m.get('topic') or ''!r:40.40} "
                  f"map {job.get('status', '-'):8} {name or ''} {flag} {m.get('why') or ''}")
        if not P.REFUND_KEY.exists() and owed:
            print(f"\n{len(owed)} to refund by hand (Stripe dashboard → Payments), then: field_payments.py refunded FILE")
        return
    cmd, arg = a[0], a[1]
    if cmd == "remove":
        sys.path.insert(0, str(Path.home() / "Projects/scholarsreadinglist/search"))
        from srl_search.field import FieldStore
        FieldStore(P.FIELDS).remove(arg, a[2] if len(a) > 2 else "removed from the line")
        return print("removed")
    p = P.PAID / (arg if arg.endswith(".json") else f"{arg}.json")
    m = json.loads(p.read_text())
    if cmd in ("approve", "refuse"):
        m["name_ok"] = cmd == "approve"
        if cmd == "refuse":
            m["name"] = None
    elif cmd == "refunded":
        m["refund"] = "done"
    else:
        sys.exit(__doc__)
    P._write(p, m)
    print(f"{p.name}: {cmd}")


main(sys.argv[1:])
