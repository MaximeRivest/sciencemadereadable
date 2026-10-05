"""How much of the Claude subscription is used right now.

Reads the same numbers as Claude Code's /usage, with the login saved by the
`claude` CLI. The access token is never printed.

    .venv/bin/python paper_corpus/usage.py
"""
import json
import urllib.request
from pathlib import Path

CREDENTIALS = Path.home() / ".claude" / ".credentials.json"


def claude_usage() -> dict:
    """{'five_hour': %, 'five_hour_resets': time, 'seven_day': %, 'seven_day_resets': time}"""
    token = json.loads(CREDENTIALS.read_text())["claudeAiOauth"]["accessToken"]
    request = urllib.request.Request("https://api.anthropic.com/api/oauth/usage", headers={
        "Authorization": f"Bearer {token}", "anthropic-beta": "oauth-2025-04-20",
        "User-Agent": "claude-cli/2.1.284"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    return {"five_hour": data["five_hour"]["utilization"], "five_hour_resets": data["five_hour"]["resets_at"],
            "seven_day": data["seven_day"]["utilization"], "seven_day_resets": data["seven_day"]["resets_at"]}


def codex_usage() -> dict:
    """ChatGPT/Codex subscription (used by Astra and Luna): {'weekly': %, 'weekly_resets': time, 'plan': ...}.
    Same numbers as Codex's /status. The access token is never printed."""
    from datetime import datetime, timezone
    tokens = json.loads((Path.home() / ".codex" / "auth.json").read_text())["tokens"]
    request = urllib.request.Request("https://chatgpt.com/backend-api/wham/usage", headers={
        "Authorization": f"Bearer {tokens['access_token']}", "chatgpt-account-id": tokens.get("account_id") or "",
        "User-Agent": "codex_cli_rs", "originator": "codex_cli_rs"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    window = data["rate_limit"]["primary_window"]
    return {"weekly": window["used_percent"], "limit_reached": data["rate_limit"]["limit_reached"],
            "weekly_resets": datetime.fromtimestamp(window["reset_at"], timezone.utc).isoformat(),
            "window_days": window["limit_window_seconds"] / 86400, "plan": data.get("plan_type")}


if __name__ == "__main__":
    u = claude_usage()
    print("Claude (Opus)")
    print(f"  5-hour window: {u['five_hour']:.0f}% used (resets {u['five_hour_resets'][:16]} UTC)")
    print(f"  Weekly:        {u['seven_day']:.0f}% used (resets {u['seven_day_resets'][:16]} UTC)")
    c = codex_usage()
    print(f"ChatGPT {c['plan']} (Astra, Luna)")
    print(f"  Weekly:        {c['weekly']:.0f}% used (resets {c['weekly_resets'][:16]} UTC)")
