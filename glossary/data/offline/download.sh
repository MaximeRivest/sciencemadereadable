#!/usr/bin/env bash
# Offline reference sources for the glossary (snapshot 2026-09-27 of Wikipedia).
# Wikimedia asks for at most 2 parallel downloads per address: we use 2.
cd "$(dirname "$0")"
B=https://dumps.wikimedia.org/other/cirrus_search_index/20260927
get() { [ -s "$2" ] && [ ! -e "$2.part" ] && return; curl -sS -L -C - --retry 10 -A "ScholarsReadingList-glossary/0.1" -o "$2.part" "$1" && mv "$2.part" "$2" && echo "$(date +%T) got $2"; }
export -f get
get "$B/index_name%3Dsimplewiki_content/simplewiki_content-20260927-00000.json.bz2" simplewiki-00000.json.bz2
get https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz kaikki-English.jsonl.gz &
seq -w 0 65 | sed 's/^/000/; s/.*\(.....\)$/\1/' | xargs -P 2 -I{} bash -c "get '$B/index_name%3Denwiki_content/enwiki_content-20260927-{}.json.bz2' enwiki-{}.json.bz2"
wait
echo "$(date +%T) all downloaded"
