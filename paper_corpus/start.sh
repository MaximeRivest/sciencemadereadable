#!/usr/bin/env bash
# Start (or resume) a rewrite run in the background. Safe to run twice.
#   start.sh          Opus
#   start.sh astra    GPT-6 Astra (works from the other end of the list)
cd "$(dirname "$0")/.."
WRITER="${1:-opus}"
SESSION="paper-rewrite"; [ "$WRITER" != "opus" ] && SESSION="paper-rewrite-$WRITER"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "$WRITER already running. Watch: paper_corpus/dashboard.sh"
  exit 0
fi
tmux new-session -d -s "$SESSION" ".venv/bin/python paper_corpus/run.py $WRITER; echo; echo 'Run ended. Press Enter to close.'; read"
echo "Started $WRITER. Watch: paper_corpus/dashboard.sh   Stop: paper_corpus/stop.sh $1"
