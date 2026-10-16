#!/usr/bin/env bash
# PDF -> reviewed rewrite -> reading editions, entirely outside the public site.
set -euo pipefail
usage() {
  echo 'Usage: ./make_readable.sh INPUT.pdf RUN_DIRECTORY --out READING_DIRECTORY [--audio] [--pdf] [--review REVIEW.json] [--reference-xml SOURCE.xml] [--visuals REVIEWED_VISUALS.json]' >&2
  echo 'Web + EPUB are always made. --audio adds Kokoro MP3; --pdf adds the large-print PDF.' >&2
  echo 'Extracted PDF text is sent to TypeSafe; OCR, rewriting and speech use local services.' >&2
  exit 2
}
(( $# >= 2 )) || usage
input=$(realpath "$1"); run=$(realpath -m "$2"); shift 2
out=''; xml=''; review=(); formats=()
while (( $# )); do
  case "$1" in
    --out) (( $# >= 2 )) || usage; out=$(realpath -m "$2"); shift 2 ;;
    --review) (( $# >= 2 )) || usage; review=("$(realpath "$2")"); shift 2 ;;
    --reference-xml) (( $# >= 2 )) || usage; xml=$(realpath "$2"); shift 2 ;;
    --visuals) (( $# >= 2 )) || usage; formats+=(--visuals "$(realpath "$2")"); shift 2 ;;
    --audio|--pdf) formats+=("$1"); shift ;;
    *) usage ;;
  esac
done
[[ -n "$out" ]] || usage
cd "$(dirname "$0")"
./run.sh "$input" "$run" "${review[@]}"
if [[ -n "$xml" ]]; then ../.venv/bin/python compare_xml.py "$run" "$xml"; fi
.venv/bin/python editions.py "$run" --out "$out" "${formats[@]}"
