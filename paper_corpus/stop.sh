#!/usr/bin/env bash
# Ask a run to stop: papers in progress finish and are saved, no new ones start.
#   stop.sh          Opus
#   stop.sh astra    Astra
cd "$(dirname "$0")"
WRITER="${1:-opus}"
DIR="rewrites"; [ "$WRITER" != "opus" ] && DIR="rewrites_$WRITER"
mkdir -p "$DIR" && touch "$DIR/STOP"
echo "Stop requested for $WRITER. Papers in progress finish first (a few minutes)."
