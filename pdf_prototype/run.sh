#!/usr/bin/env bash
# Standalone pipeline; never calls the site's queue or publishes a rewrite.
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
  echo 'Usage: ./run.sh INPUT.pdf RUN_DIRECTORY [REVIEW.json]' >&2
  echo 'Extracted text is sent to TypeSafe. Set TYPESAFE_API_KEY first.' >&2
  exit 2
fi
pdf=$(realpath "$1")
run=$(realpath -m "$2")
review=()
if (( $# == 3 )); then review=(--overrides "$(realpath "$3")"); fi
cd "$(dirname "$0")"
uv sync --frozen
.venv/bin/python extract.py "$pdf" "$run"
.venv/bin/python label.py "$run" "${review[@]}"
../.venv/bin/python prepare_input.py "$run"
node --conditions=functai-source rewrite.ts "$run"
printf '\nSaved rewrite: %s/rewrite.md\n' "$run"
