"""Build the usual glossary and reply limits from PDF-derived sections only.

Run with ../.venv/bin/python: reuses the site's installed glossary dependencies
and read-only reference database without importing or calling its queue/server.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "glossary"))
import glossary as G
from inputs import glossary_lines
import tiktoken

OPENING = ["title", "abstract", "introduction_first", "conclusion"]
OTHER = ["introduction_rest", "methods", "results", "discussion"]


def prepare(run):
    start = time.monotonic()
    labelled = json.loads((run / "labelled.json").read_text())
    if not labelled["ready"]:
        raise ValueError(
            "Section labelling needs review. Inspect labelled.json; do not bypass the gate."
        )
    sections = labelled["sections"]
    signature = hashlib.sha256(
        json.dumps(sections, sort_keys=True).encode()
    ).hexdigest()
    dest = run / "paper.json"
    if (
        dest.exists()
        and json.loads(dest.read_text()).get("sections_sha256") == signature
    ):
        print("Glossary cache hit")
        return
    if not G.OFFLINE:
        raise RuntimeError(
            "Prototype requires the existing offline glossary database; no online fallback."
        )
    # These lookups must already exist so importing/running the shared glossary
    # cannot build or overwrite corpus-side caches.
    for name in ("kuperman_aoa.json", "paper_counts.json"):
        if not (G.DATA / name).exists():
            raise RuntimeError(f"Missing shared glossary lookup: {name}")

    def no_online_lookup(*args, **kwargs):
        raise RuntimeError(
            "The offline reference snapshot is incomplete; online lookups are disabled for private documents."
        )

    G.get_json = no_online_lookup
    # The offline snapshot has no Wiktionary rows, so the shared glossary would ask
    # Wiktionary online for single words. Private documents skip that source: a
    # few one-word definitions may be missing, but no term leaves this server.
    G.wiktionary = lambda term: None
    print("Building the reference glossary from PDF text…", flush=True)
    entries = G.glossary(
        {
            "paper_id": "pdf-" + labelled["extraction_signature"]["pdf_sha256"][:16],
            **sections,
        }
    )
    whole = "\n\n".join(
        f"## {s}\n\n{sections[s]}" for s in OPENING + OTHER if sections.get(s)
    )
    enc = tiktoken.get_encoding("cl100k_base")
    paper = {
        "sections": sections,
        "sections_sha256": signature,
        "glossary": {
            "opening": glossary_lines(entries, whole),
            **{
                s: glossary_lines(entries, sections[s])
                for s in OTHER
                if sections.get(s)
            },
        },
        "glossary_terms": len(entries),
        "max_tokens": {
            "opening": 6000,
            **{
                s: min(16000, max(1500, 2 * len(enc.encode(sections[s]))))
                for s in OTHER
                if sections.get(s)
            },
        },
        "seconds": time.monotonic() - start,
        "source": "PDF-derived sections only; no publisher XML or saved paper glossary used",
    }
    (run / "glossary.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2)
    )
    dest.write_text(json.dumps(paper, ensure_ascii=False, indent=2))
    print(f"Prepared {len(entries)} glossary entries in {paper['seconds']:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    prepare(p.parse_args().run)
