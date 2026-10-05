"""Live view of the long rewrite run. Ctrl-C to quit (the run keeps going).

    paper_corpus/dashboard.sh
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = {"Opus": HERE / "rewrites", "Astra": HERE / "rewrites_astra"}

GREEN, YELLOW, RED, DIM, BOLD, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def bar(fraction, width=40, color=GREEN):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def ago(iso):
    if not iso:
        return "never"
    seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    return f"{seconds:.0f}s ago" if seconds < 120 else f"{seconds / 60:.0f} min ago"


def draw():
    width = shutil.get_terminal_size((100, 40)).columns
    lines = [f"{BOLD}Paper rewriting — opening-only recipe{RESET}   {DIM}{time.strftime('%H:%M:%S')}{RESET}", ""]
    done = {f.stem for out in RUNS.values() for f in (out / "rewrites").glob("*.parquet")}
    total = 4500
    for out in RUNS.values():
        if (out / "status.json").exists():
            total = json.loads((out / "status.json").read_text()).get("total_training") or total
    lines.append(f"{BOLD}All writers{RESET} {bar(len(done) / total)}  {len(done):,} / {total:,}  ({len(done) / total:.0%})")
    lines.append("")
    for name, out in RUNS.items():
        if (out / "status.json").exists():
            lines += run_section(name, out, width)
    lines.append(f"{DIM}start.sh [astra] · stop.sh [astra] (graceful) · tmux attach -t paper-rewrite[-astra] · Ctrl-C quits this view{RESET}")
    return lines


def run_section(name, out, width):
    STATUS, LOG, SAVED = out / "status.json", out / "run.log", out / "rewrites"
    lines = [f"{BOLD}── {name} ──{RESET}"]
    s = json.loads(STATUS.read_text())
    running = alive(s.get("pid"))
    st = s["state"] if running or s["state"] == "finished" else "not running"
    color = {"running": GREEN, "paused": YELLOW, "stopping": YELLOW, "finished": DIM}.get(st, RED)
    lines.append(f"State: {color}{BOLD}{st.upper()}{RESET}  {s.get('reason', '')}")
    if s.get("pause_until") and st == "paused":
        resume = datetime.fromtimestamp(s["pause_until"]).strftime("%H:%M")
        lines.append(f"       paused until {resume} (local time)")
    lines.append(f"{DIM}status written {ago(s.get('written_at'))}{RESET}")
    lines.append("")

    total = s["total_training"] or 1
    done_files = len(list(SAVED.glob("*.parquet")))
    this_run = s["done_this_run"]
    lines.append(f"Papers   {done_files:,} written by {name}")
    started = datetime.fromisoformat(s["started_at"])
    hours = max((datetime.now(timezone.utc) - started).total_seconds() / 3600, 1e-6)
    rate = this_run / hours
    left = total - done_files
    eta = f"{left / rate:.0f} h of running" if rate > 0 else "—"
    lines.append(f"This run {this_run:,} done · {s['failed_this_run']} failed · {s['rate_limits']} rate limits · "
                 f"{rate:.0f} papers/h · {left:,} left ≈ {eta}")
    mins = s.get("minutes_per_paper") or []
    if mins:
        lines.append(f"         {sum(mins) / len(mins):.1f} min per paper · "
                     f"{s['input_tokens'] / 1e6:.1f}M tokens in, {s['output_tokens'] / 1e6:.1f}M out this run")
    lines.append("")

    u = s.get("usage")
    if u:
        windows = [("Weekly ", "seven_day", "seven_day_resets", 95 if name == "Opus" else 100)]
        if not u.get("no_five_hour"):
            windows.insert(0, ("5-hour ", "five_hour", "five_hour_resets", 90))
        for label, key, resets, limit in windows:
            pct = u[key]
            c = GREEN if pct < 70 else YELLOW if pct < limit else RED
            lines.append(f"{label}  {bar(pct / 100, color=c)}  {pct:.0f}%  "
                         f"{DIM}(run stops/pauses at {limit}% · resets {(u[resets] or '—')[5:16].replace('T', ' ')} UTC){RESET}")
        lines.append(f"{DIM}usage checked {ago(s.get('usage_checked_at'))}{RESET}")
        lines.append("")

    working = s.get("in_progress") or {}
    lines.append(f"Working on {len(working)} paper(s):")
    for pid, since in sorted(working.items(), key=lambda kv: kv[1]):
        minutes = (time.time() - since) / 60
        flag = f"  {RED}slow{RESET}" if minutes > 20 else ""
        lines.append(f"  {pid:<14} {minutes:4.1f} min{flag}")
    lines.append("")

    errors = s.get("last_errors") or []
    if errors:
        lines.append(f"{RED}Last errors:{RESET}")
        for e in errors[-3:]:
            lines.append(f"  {e['at'][11:19]} {e['paper_id']}: {e['error'][:width - 30]}")
        lines.append("")

    if LOG.exists():
        lines.append(f"{DIM}Log (run.log):{RESET}")
        for line in LOG.read_text().splitlines()[-6:]:
            lines.append(f"  {DIM}{line[:width - 4]}{RESET}")
    lines.append("")
    return lines


def main():
    once = "--once" in sys.argv
    try:
        while True:
            out = "\n".join(draw())
            if once:
                print(out)
                return
            sys.stdout.write("\033[H\033[2J" + out + "\n")
            sys.stdout.flush()
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
