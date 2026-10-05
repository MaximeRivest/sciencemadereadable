"""Build the offline reference database for the glossary from the downloaded dumps.

    .venv/bin/python glossary/build_offline.py

Reads data/offline/ (from data/offline/download.sh):
  simplewiki-00000.json.bz2, enwiki-000NN.json.bz2   Wikipedia's search index dumps (2026-09-27):
                                                     one JSON document per article, with its plain
                                                     text, lead text, redirects and templates
  kaikki-English.jsonl.gz                            English Wiktionary, already parsed (kaikki.org)
Writes data/offline/reference.sqlite:
  titles(site, title, final, disamb)   every article title and every redirect to it
  extracts(site, title, text)          the start of the article's lead (plain text, ~800 characters)
  wiktionary(word, pos, gloss)         the first real definition of each English word
glossary.py uses this file instead of the web when it exists (same answers, no limits).
"""
from __future__ import annotations

import bz2
import gzip
import json
import re
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFLINE = HERE / "data" / "offline"
DB = OFFLINE / "reference.sqlite"
KEEP = 800          # characters of lead text kept per article (glossary.py uses the first two sentences)
STUB = re.compile(r"\s*(This (short )?article .{0,80}(can be made longer|is a stub).*$)", re.S)


def parse_shard(args):
    path, site = args
    titles, extracts = [], []
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith('{"index"'):
                continue
            d = json.loads(line)
            if d.get("namespace") != 0 or d.get("page_type") == "redirect":
                continue
            title = d["title"]
            # not the templates: on English Wikipedia nearly every article uses a template
            # (via its short description) whose name contains "disambiguation"
            disamb = int(any("isambiguation pages" in c for c in d.get("category", [])))
            text = d.get("opening_text") or d.get("text") or ""
            text = STUB.sub("", text)[:KEEP]
            titles.append((site, title, title, disamb))
            for r in d.get("redirect", []):
                if r.get("namespace") == 0:
                    titles.append((site, r["title"], title, disamb))
            extracts.append((site, title, text))
    return path.name, titles, extracts


def wiktionary_rows(path):
    skip = ("plural of", "alternative", "obsolete", "archaic", "misspelling", "abbreviation of")
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("lang_code") != "en":
                continue
            word = d.get("word")
            if not word or word in seen:
                continue
            for sense in d.get("senses", []):
                if sense.get("form_of") or "form-of" in sense.get("tags", []):
                    continue
                glosses = sense.get("glosses") or []
                if glosses and not glosses[0].lower().startswith(skip):
                    seen.add(word)
                    yield word, d.get("pos", ""), glosses[0]
                    break


def main(test=False):
    global DB
    shards = [(OFFLINE / "simplewiki-00000.json.bz2", "simple.wikipedia.org")] + \
             [(p, "en.wikipedia.org") for p in sorted(OFFLINE.glob("enwiki-*.json.bz2"))]
    if test:                                    # simple English Wikipedia only, to check the pipeline
        shards, DB = shards[:1], OFFLINE / "test.sqlite"
    elif len(shards) < 67:
        sys.exit(f"downloads not finished: {len(shards) - 1} of 66 enwiki shards present")
    tmp = DB.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    db.execute("pragma journal_mode=off")
    db.execute("pragma synchronous=off")
    db.execute("create table titles (site text, title text, final text, disamb int)")
    db.execute("create table extracts (site text, title text, text text)")
    db.execute("create table wiktionary (word text, pos text, gloss text)")
    started = time.time()
    with Pool(32) as pool:
        for k, (name, titles, extracts) in enumerate(pool.imap_unordered(parse_shard, shards), 1):
            db.executemany("insert into titles values (?,?,?,?)", titles)
            db.executemany("insert into extracts values (?,?,?)", extracts)
            db.commit()
            print(f"{k}/{len(shards)} {name}: {len(extracts):,} articles, {len(titles):,} titles "
                  f"({time.time() - started:.0f}s)", flush=True)
    kaikki = OFFLINE / "kaikki-English.jsonl.gz"
    if kaikki.exists() and not test:
        db.executemany("insert into wiktionary values (?,?,?)", wiktionary_rows(kaikki))
        db.commit()
        print(f"wiktionary: {db.execute('select count(*) from wiktionary').fetchone()[0]:,} words", flush=True)
    print("indexing", flush=True)
    # a title can appear twice (an article and a redirect of the same name): keep the article
    # (SQLite takes the other columns from the row where max() is reached)
    db.execute("create table titles2 as select site, title, final, disamb, max(title = final) as is_article "
               "from titles group by site, title")
    db.execute("drop table titles")
    db.execute("alter table titles2 rename to titles")
    db.execute("create index titles_key on titles (site, title)")
    db.execute("create index extracts_key on extracts (site, title)")
    db.execute("create index wiktionary_key on wiktionary (word)")
    db.commit()
    db.close()
    tmp.replace(DB)
    print(f"-> {DB} ({DB.stat().st_size / 1e9:.1f} GB, {time.time() - started:.0f}s)")


if __name__ == "__main__":
    main(test="--test" in sys.argv)
