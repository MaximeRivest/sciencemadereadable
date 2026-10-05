"""The long run: rewrite every training paper with Opus, until the subscription runs out.

Start, watch, stop (from any terminal on lambda):

    paper_corpus/start.sh        # starts in the background (tmux session "paper-rewrite")
    paper_corpus/dashboard.sh    # live progress
    paper_corpus/stop.sh         # finish the papers in progress, then stop

Safe to start again at any time: finished papers are skipped.

How it respects the limits
- Every 2 minutes it reads the Claude subscription usage.
  5-hour window at 90% or more: it pauses until the window resets, then continues.
  Weekly allowance at 95% or more: it finishes the papers in progress and stops.
- A rate-limit error pauses all workers (15 minutes, or what the provider asks),
  and the paper is put back in the queue; it does not count as a failure.
- 8 failures in a row (a login problem, a broken setting...) stop the run.
- A paper that has failed 3 times is skipped from then on (listed in failures.jsonl).
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import dpyr
import lm15

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "rewrite_benchmark"))
sys.path.insert(0, str(HERE))
import translator
from usage import claude_usage, codex_usage

# Which writer: "opus" (default) or "astra" (python run.py astra).
WRITER = sys.argv[1] if len(sys.argv) > 1 else "opus"
translator.use_writer(WRITER)
OTHER_WRITER = "astra" if WRITER == "opus" else "opus"
OUT = HERE / ("rewrites" if WRITER == "opus" else f"rewrites_{WRITER}")
OTHER_SAVED = HERE / ("rewrites" if OTHER_WRITER == "opus" else f"rewrites_{OTHER_WRITER}") / "rewrites"
SAVED = OUT / "rewrites"
FAILURES = OUT / "failures.jsonl"
STATUS = OUT / "status.json"
LOG = OUT / "run.log"
STOP_FILE = OUT / "STOP"
TEST_IDS = HERE / "test_ids.txt"

WORKERS = 10                 # papers at the same time
FIVE_HOUR_PAUSE = 90         # % of the 5-hour window: pause until it resets
# % of the weekly allowance: stop for good. Astra's ChatGPT plan also powers the
# assistant in Chattering, and going over would spend paid credits: stop earlier.
WEEKLY_STOP = 95 if WRITER == "opus" else 100   # Astra: use the whole week (resets are available)
RATE_LIMIT_PAUSE = 15 * 60   # seconds, when the provider gives no retry time
MAX_FAILURES_IN_A_ROW = 8
MAX_ATTEMPTS_PER_PAPER = 3
SLOW_MINUTES = 20            # flagged on the dashboard

SAVED.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc)


def log(message):
    line = f"{now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- shared state

lock = threading.Lock()
state = {
    "state": "starting", "reason": "", "started_at": now().isoformat(), "pid": None,
    "done_before_run": 0, "done_this_run": 0, "failed_this_run": 0, "rate_limits": 0,
    "failures_in_a_row": 0, "pause_until": None, "in_progress": {}, "last_errors": [],
    "usage": None, "usage_checked_at": None, "input_tokens": 0, "output_tokens": 0,
    "remaining": 0, "total_training": 0, "minutes_per_paper": [],
}
stopping = threading.Event()


def write_status():
    with lock:
        snapshot = json.loads(json.dumps(state, default=str))
    snapshot["minutes_per_paper"] = snapshot["minutes_per_paper"][-200:]
    snapshot["written_at"] = now().isoformat()
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot, indent=1))
    tmp.replace(STATUS)


def stop(reason):
    if not stopping.is_set():
        log(f"STOPPING: {reason}")
        with lock:
            state["reason"] = reason
            state["state"] = "stopping"
        stopping.set()


def pause(seconds, reason):
    until = time.time() + seconds
    with lock:
        if state["pause_until"] is None or until > state["pause_until"]:
            state["pause_until"] = until
            state["reason"] = reason
    log(f"PAUSE {seconds / 60:.0f} min: {reason}")


# ---------------------------------------------------------------- work

def attempts_so_far():
    if not FAILURES.exists():
        return Counter()
    return Counter(json.loads(l)["paper_id"] for l in FAILURES.read_text().splitlines() if l.strip())


def save_paper(paper, rows):
    final = SAVED / translator.file_name(paper["paper_id"])
    tmp = final.with_suffix(".tmp")
    dpyr.read(rows).write_parquet(tmp)
    tmp.replace(final)


def worker(todo: queue.Queue):
    while not stopping.is_set():
        with lock:
            until = state["pause_until"]
        if until and time.time() < until:
            time.sleep(5)
            continue
        try:
            paper = todo.get_nowait()
        except queue.Empty:
            return
        pid = paper["paper_id"]
        if (OTHER_SAVED / translator.file_name(pid)).exists():
            continue   # the other writer already did this one (the two runs meet in the middle)
        with lock:
            state["in_progress"][pid] = time.time()
        try:
            rows = translator.translate_paper(paper)
            save_paper(paper, rows)
            with lock:
                state["done_this_run"] += 1
                state["remaining"] -= 1
                state["failures_in_a_row"] = 0
                state["input_tokens"] += sum(r["input_tokens"] for r in rows)
                state["output_tokens"] += sum(r["output_tokens"] for r in rows)
                state["minutes_per_paper"].append(rows[0]["paper_minutes"])
        except lm15.RateLimitError as error:
            todo.put(paper)   # not a failure: try again after the pause
            with lock:
                state["rate_limits"] += 1
            pause(getattr(error, "retry_after", None) or RATE_LIMIT_PAUSE, f"rate limit ({str(error)[:120]})")
        except Exception as error:
            message = f"{type(error).__name__}: {str(error)[:300]}"
            with FAILURES.open("a") as f:
                f.write(json.dumps({"paper_id": pid, "error": message, "at": now().isoformat()}) + "\n")
            with lock:
                state["failed_this_run"] += 1
                state["failures_in_a_row"] += 1
                state["last_errors"] = (state["last_errors"] + [{"paper_id": pid, "error": message,
                                                                  "at": now().isoformat()}])[-5:]
                in_a_row = state["failures_in_a_row"]
            log(f"failed {pid}: {message[:160]}")
            if in_a_row >= MAX_FAILURES_IN_A_ROW:
                stop(f"{in_a_row} failures in a row; last: {message[:160]}")
        finally:
            with lock:
                state["in_progress"].pop(pid, None)


def monitor():
    """Usage checks, the STOP file, and the status file, until the run ends."""
    last_usage = 0.0
    while True:
        if STOP_FILE.exists():
            stop("STOP file found (stop.sh)")
        if time.time() - last_usage > 120:
            last_usage = time.time()
            try:
                if WRITER == "opus":
                    u = claude_usage()
                else:   # ChatGPT has one weekly window; shown in the same fields
                    c = codex_usage()
                    u = {"five_hour": 0, "five_hour_resets": c["weekly_resets"], "seven_day": c["weekly"],
                         "seven_day_resets": c["weekly_resets"], "no_five_hour": True}
                    if c["limit_reached"]:
                        stop("ChatGPT limit reached")
                with lock:
                    state["usage"], state["usage_checked_at"] = u, now().isoformat()
                if u["seven_day"] >= WEEKLY_STOP:
                    stop(f"weekly allowance at {u['seven_day']:.0f}% (resets {u['seven_day_resets'][:16]} UTC)")
                elif u["five_hour"] >= FIVE_HOUR_PAUSE:
                    reset = datetime.fromisoformat(u["five_hour_resets"]).timestamp() if u["five_hour_resets"] else time.time() + 1800
                    pause(max(60, reset - time.time() + 60), f"5-hour window at {u['five_hour']:.0f}%")
            except Exception as error:
                log(f"usage check failed ({type(error).__name__}); continuing")
        with lock:
            until = state["pause_until"]
            if until and time.time() >= until:
                state["pause_until"] = None
                state["reason"] = ""
            if state["state"] not in ("stopping", "finished"):
                state["state"] = "paused" if state["pause_until"] else "running"
        write_status()
        if state["state"] == "finished":
            return
        time.sleep(10)


def main():
    import os
    STOP_FILE.unlink(missing_ok=True)
    test = set(TEST_IDS.read_text().split())
    papers = [p for p in dpyr.read_parquet(HERE / "papers.parquet").to_dicts() if p["paper_id"] not in test]
    done = {f.stem for f in SAVED.glob("*.parquet")} | {f.stem for f in OTHER_SAVED.glob("*.parquet")}
    tries = attempts_so_far()
    todo_list = [p for p in papers if translator.file_name(p["paper_id"])[:-8] not in done]
    skipped = [p for p in todo_list if tries[p["paper_id"]] >= MAX_ATTEMPTS_PER_PAPER]
    todo_list = [p for p in todo_list if tries[p["paper_id"]] < MAX_ATTEMPTS_PER_PAPER]
    if WRITER != "opus":
        todo_list.reverse()   # start from the other end of the list than Opus
    # Papers that already failed go last: a handful of hard papers at the front of
    # the queue would otherwise look like "8 failures in a row" and stop the run.
    todo_list.sort(key=lambda p: tries[p["paper_id"]])

    with lock:
        state.update(pid=os.getpid(), writer=WRITER, model=translator.MODEL, done_before_run=len(papers) - len(todo_list) - len(skipped),
                     remaining=len(todo_list), total_training=len(papers))
    log(f"START: {len(papers)} training papers, {state['done_before_run']} already done, "
        f"{len(todo_list)} to do, {len(skipped)} skipped after {MAX_ATTEMPTS_PER_PAPER} failures")

    todo = queue.Queue()
    for p in todo_list:
        todo.put(p)
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    workers = [threading.Thread(target=worker, args=(todo,)) for _ in range(WORKERS)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    with lock:
        if not stopping.is_set():
            state["reason"] = "all papers done"
        state["state"] = "finished"
    write_status()
    log(f"END ({state['reason']}): {state['done_this_run']} papers this run, "
        f"{state['done_before_run'] + state['done_this_run']} of {len(papers)} in total")


if __name__ == "__main__":
    main()
