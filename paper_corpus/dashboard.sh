#!/usr/bin/env bash
cd "$(dirname "$0")/.."
exec .venv/bin/python paper_corpus/dashboard.py "$@"
