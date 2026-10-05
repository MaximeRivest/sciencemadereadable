"""A reference glossary for one paper: the terms a smart, curious 12–16-year-old
probably does not know, each with a plain explanation taken from a reference work
(not from a language model's memory).

    .venv/bin/python glossary/glossary.py               # the first 3 benchmark test papers
    .venv/bin/python glossary/glossary.py PMC12995867   # one paper (training or test)

Writes glossary/out/<paper_id>.json and review.md (one table per paper).

How it works
1. Which words are hard. Kuperman et al. (2012) age-of-acquisition ratings: the age
   (in years) at which ~30,000 English words are learned. A word learned at
   HARD_AGE or later is hard; a word not in that list is hard when it is rare in
   everyday English (wordfreq Zipf frequency below RARE_ZIPF; 3 = once per million
   words).
2. Which terms. Only what the model rarely saw: a term used in more than
   MAX_PAPER_SHARE of corpus papers is learned from the training data, and a single
   word common in everyday English (Zipf >= EVERYDAY_ZIPF) is known from pre-training;
   both are left out. Then: (a) Abbreviations defined in the paper itself ("chemical oxygen
   demand (COD)"): the paper's own long form wins, so COD is never the fish.
   (b) Phrases of 1–3 words that are a Wikipedia article title and contain a hard
   word ("structural equation modeling", "anammox", "plateau pika"). Longer phrases
   win over the words inside them.
3. The explanation. Simple English Wikipedia (written for learners) when it has the
   page, else English Wikipedia; the first two sentences. Disambiguation pages are
   left out. A page whose text shares too few words with the paper is marked as a
   possible wrong sense.
Every web answer is cached in data/cache/, so reruns are free.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import openpyxl
from wordfreq import zipf_frequency

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
CACHE = DATA / "cache"
OUT = HERE / "out"
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))

HARD_AGE = 12.0          # learned at 12 or later: needs an explanation
RARE_ZIPF = 3.0          # words outside the age list: hard when rarer than once per million words
EVERYDAY_ZIPF = 3.3      # a single word this common in everyday English (~2 per million words) is not looked up
MAX_PAPER_SHARE = 0.01   # a term used in more than 1% of corpus papers is learned from training data: no lookup
MIN_OVERLAP = 2          # shared content words between a page and the paper, else "check the sense"
USER_AGENT = "ScholarsReadingList-glossary/0.1 (https://scholarsreadinglist.com; research use)"
SECTIONS = ["title", "abstract", "introduction_first", "introduction_rest", "methods", "results",
            "discussion", "conclusion"]
STOP = set("""a an the of and or in on at to for from by with without into onto over under than then
this that these those is are was were be been being it its as we our us they their them he she his her
not no nor but if so such which who whom whose what when where while also both each either neither
more most less least very can could may might will would should shall do does did done have has had
using used use between among within across per via about after before during through against""".split())


# ---------------------------------------------------------------- word difficulty

def load_ages() -> dict[str, float]:
    cache = DATA / "kuperman_aoa.json"
    if cache.exists():
        return json.loads(cache.read_text())
    wb = openpyxl.load_workbook(DATA / "osf-vb9je.xlsx", read_only=True)   # Kuperman et al. 2012
    ages = {}
    for word, *_, rating, _sd, _dunno, _pos in wb.worksheets[0].iter_rows(min_row=2, values_only=True):
        if isinstance(word, str) and isinstance(rating, (int, float)):
            ages[word.lower()] = float(rating)
    cache.write_text(json.dumps(ages))
    return ages


AGES = load_ages()


def load_paper_counts() -> tuple[dict[str, int], int]:
    """How many corpus papers use each word (lower case; abbreviations kept as written)."""
    cache = DATA / "paper_counts.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["counts"], d["papers"]
    import dpyr
    counts: Counter = Counter()
    rows = dpyr.read_parquet(ROOT / "paper_corpus/papers.parquet").to_dicts()
    for p in rows:
        text = "\n".join(p.get(s) or "" for s in SECTIONS)
        words = set(re.findall(r"[A-Za-z][A-Za-z0-9/\-]*", text))
        counts.update({w if sum(c.isupper() for c in w) > 1 else w.lower() for w in words})
    cache.write_text(json.dumps({"counts": counts, "papers": len(rows)}))
    return dict(counts), len(rows)


PAPER_COUNTS, N_PAPERS = load_paper_counts()


def paper_share(term: str) -> float:
    """Share of corpus papers using the term's rarest word (an upper bound for a phrase)."""
    words = term.split() if " " in term else [term]
    keys = [w if sum(c.isupper() for c in w) > 1 else w.lower() for w in words]
    return min(min(PAPER_COUNTS.get(k, 0), *(PAPER_COUNTS.get(f, 0) for f in word_forms(k))) if k.islower()
               else PAPER_COUNTS.get(k, 0) for k in keys) / N_PAPERS


def word_forms(w: str) -> list[str]:
    w = w.lower()
    forms = [w]
    if w.endswith("ies"):
        forms.append(w[:-3] + "y")
    if w.endswith("es"):
        forms.append(w[:-2])
    if w.endswith("s") and not w.endswith("ss"):
        forms.append(w[:-1])
    return forms


def difficulty(w: str) -> tuple[bool, str]:
    """(hard?, why) for one word."""
    for f in word_forms(w):
        if f in AGES:
            age = AGES[f]
            return age >= HARD_AGE, f"learned at {age:.1f}"
    z = max(zipf_frequency(f, "en") for f in word_forms(w))
    return z < RARE_ZIPF, f"not in the age list; frequency {z:.1f}"


# ---------------------------------------------------------------- web lookups (cached)

class _RateLimit:
    """At most RATE requests per second to Wikimedia, across all threads."""
    def __init__(self, rate: float):
        import threading
        self.gap, self.next, self.lock = 1.0 / rate, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next)
            self.next = at + self.gap
        time.sleep(max(0.0, at - now))


RATE = _RateLimit(float(__import__("os").environ.get("GLOSSARY_RATE", "20")))


def get_json(url: str, cache: bool = True) -> dict:
    import hashlib
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".json")
    if cache and path.exists():
        return json.loads(path.read_text())
    for attempt in range(10):
        try:
            RATE.wait()
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            data = json.loads(urllib.request.urlopen(req, timeout=60).read())
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data = {}
                break
            wait = e.headers.get("Retry-After") if e.code == 429 else None
            time.sleep(float(wait) + 1 if wait and wait.isdigit() else 3 * (attempt + 1))
        except OSError:
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"cannot reach {url}")
    if cache:
        path.write_text(json.dumps(data))
    return data


class _Store:
    """Title lookups and page extracts, one row per title (shared by all papers and threads)."""
    def __init__(self, path: Path):
        import sqlite3
        import threading
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("create table if not exists titles (site text, title text, final text, disamb int, "
                        "primary key (site, title))")
        self.db.execute("create table if not exists extracts (site text, title text, text text, "
                        "primary key (site, title))")

    def get(self, table, site, keys):
        out = {}
        with self.lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                q = f"select * from {table} where site=? and title in ({','.join('?' * len(chunk))})"
                for row in self.db.execute(q, [site, *chunk]):
                    out[row[1]] = row[2:]
        return out

    def put(self, table, rows):
        with self.lock:
            self.db.executemany(f"insert or replace into {table} values ({','.join('?' * len(rows[0]))})", rows)
            self.db.commit()


STORE = _Store(DATA / "wiki.sqlite")

# The offline snapshot (build_offline.py): when present, every lookup is local.
OFFLINE_DB = Path(__import__("os").environ.get("GLOSSARY_DB", DATA / "offline" / "reference.sqlite"))
OFFLINE = None
if OFFLINE_DB.exists() and not __import__("os").environ.get("GLOSSARY_ONLINE"):
    import sqlite3 as _sqlite3
    import threading as _threading
    OFFLINE = _sqlite3.connect(f"file:{OFFLINE_DB}?mode=ro", uri=True, check_same_thread=False)
    _OFFLINE_LOCK = _threading.Lock()


def _offline(query, args):
    with _OFFLINE_LOCK:
        return OFFLINE.execute(query, args).fetchall()


def existing_titles(site: str, titles: list[str]) -> dict[str, dict]:
    """For each candidate title: the article it lands on (after redirects), if any.
    {candidate: {"title": ..., "disambiguation": bool}}. Asked 50 titles at a time; remembered."""
    titles = list(dict.fromkeys(titles))
    if OFFLINE:
        found = {}
        for i in range(0, len(titles), 500):
            chunk = titles[i:i + 500]
            for title, final, disamb in _offline(
                    f"select title, final, disamb from titles where site=? and title in ({','.join('?' * len(chunk))})",
                    [site, *chunk]):
                found[title] = {"title": final, "disambiguation": bool(disamb)}
        return found
    known = STORE.get("titles", site, titles)
    missing = [t for t in titles if t not in known]
    for i in range(0, len(missing), 50):
        chunk = missing[i:i + 50]
        q = urllib.parse.urlencode({"action": "query", "format": "json", "redirects": 1, "prop": "pageprops",
                                    "ppprop": "disambiguation", "titles": "|".join(chunk)})
        data = get_json(f"https://{site}/w/api.php?{q}", cache=False).get("query", {})
        norm = {n["from"]: n["to"] for n in data.get("normalized", [])}
        redir = {r["from"]: r["to"] for r in data.get("redirects", [])}
        pages = {p["title"]: p for p in data.get("pages", {}).values()}
        rows = []
        for t in chunk:
            final = redir.get(norm.get(t, t), norm.get(t, t))
            page = pages.get(final)
            ok = page and "missing" not in page and "invalid" not in page
            rows.append((site, t, final if ok else None, int(bool(ok) and "disambiguation" in page.get("pageprops", {}))))
            known[t] = rows[-1][2:]
        STORE.put("titles", rows)
    return {t: {"title": known[t][0], "disambiguation": bool(known[t][1])} for t in titles
            if t in known and known[t][0]}


def extracts(site: str, titles: list[str]) -> dict[str, str]:
    """The lead section (plain text) of each page, 20 pages a request; remembered."""
    titles = list(dict.fromkeys(titles))
    if OFFLINE:
        out = {}
        for i in range(0, len(titles), 500):
            chunk = titles[i:i + 500]
            out.update(_offline(f"select title, text from extracts where site=? and title in "
                                f"({','.join('?' * len(chunk))})", [site, *chunk]))
        return out
    known = {t: v[0] for t, v in STORE.get("extracts", site, titles).items()}
    missing = [t for t in titles if t not in known]
    for i in range(0, len(missing), 20):
        chunk = missing[i:i + 20]
        q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "extracts", "exintro": 1,
                                    "explaintext": 1, "exlimit": 20, "titles": "|".join(chunk)})
        data = get_json(f"https://{site}/w/api.php?{q}", cache=False).get("query", {})
        got = {p["title"]: p.get("extract", "") for p in data.get("pages", {}).values()}
        rows = [(site, t, got.get(t, "")) for t in chunk]
        known.update({t: x for _, t, x in rows})
        STORE.put("extracts", rows)
    return known


def summary(site: str, title: str) -> str:
    text = extracts(site, [title]).get(title, "")
    # pronunciation and other bracketed asides that open many articles
    text = re.sub(r"\s*\([^()]*(?:/|ⓘ|listen|pronounced|lit\.)[^()]*\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return " ".join(parts[:n])


# ---------------------------------------------------------------- the paper's own abbreviations

def abbreviations(text: str) -> dict[str, str]:
    """"long form (SF)" pairs, Schwartz & Hearst (2003) style: the short form's
    letters appear in order in the words just before the parenthesis, the first
    letter starting a word."""
    found = {}
    for m in re.finditer(r"\(([A-Za-z][A-Za-z0-9/+\-]{1,9})\)", text):
        sf = m.group(1)
        if not any(c.isupper() for c in sf) or sf.lower() in ("i", "ii", "iii", "iv"):
            continue
        words = re.findall(r"[\w\-]+", text[max(0, m.start() - 200):m.start()])[-(min(len(sf) + 5, 2 * len(sf))):]
        letters = [c.lower() for c in sf if c.isalnum()]
        best = None
        for start in range(len(words) - 1, -1, -1):            # shortest long form that fits
            cand = " ".join(words[start:])
            if cand[0].lower() != letters[0]:
                continue
            i = 0
            for c in cand.lower():
                if i < len(letters) and c == letters[i]:
                    i += 1
            if i == len(letters):
                best = cand
                break
        if best and best.lower() != sf.lower() and sf not in found:
            found[sf] = best
    return found


# ---------------------------------------------------------------- candidate phrases

def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-]*[A-Za-z]|[A-Za-z]", text)


def candidates(text: str) -> Counter:
    """1–3-word phrases with no stop word at either end and at least one hard word."""
    out = Counter()
    for sentence in re.split(r"[.;:!?()\[\]\n]", text):
        ws = tokens(sentence)
        for n in (1, 2, 3):
            for i in range(len(ws) - n + 1):
                gram = ws[i:i + n]
                if gram[0].lower() in STOP or gram[-1].lower() in STOP or any(len(g) < 3 for g in gram):
                    continue
                if any(g.isupper() for g in gram):           # abbreviations are handled separately
                    continue
                if any(difficulty(g)[0] for g in gram):
                    out[" ".join(gram)] += 1
    return out


# ---------------------------------------------------------------- the glossary

def content_words(text: str) -> set[str]:
    return {w.lower() for w in tokens(text) if len(w) > 3 and w.lower() not in STOP}


def same_stem(term: str, title: str) -> bool:
    """A one-word term must land on a page about itself ("palatable" -> "Palatability"),
    not on a neighbour ("indiscriminate" -> "Discrimination")."""
    t = re.sub(r"[^a-z]", "", term.lower())[:5]
    return any(re.sub(r"[^a-z]", "", w.lower()).startswith(t) for w in title.split())


def wiktionary(term: str) -> dict | None:
    # offline when the snapshot has Wiktionary (kaikki.org); else the online API, cached on disk
    if OFFLINE and _offline("select count(*) from (select 1 from wiktionary limit 1)", [])[0][0]:
        names = {"adj": "adjective", "adv": "adverb", "prep": "preposition", "conj": "conjunction"}
        for form in dict.fromkeys([term, term.lower(), *word_forms(term)]):
            rows = _offline("select pos, gloss from wiktionary where word=? limit 1", [form])
            if rows:
                pos, gloss = rows[0]
                return {"explanation": f"{form} ({names.get(pos, pos)}): {gloss}", "source": "Wiktionary",
                        "page": form, "url": f"https://en.wiktionary.org/wiki/{urllib.parse.quote(form)}"}
        return None
    for form in dict.fromkeys([term, term.lower(), *word_forms(term)]):
        data = get_json(f"https://en.wiktionary.org/api/rest_v1/page/definition/{urllib.parse.quote(form)}")
        for sense in data.get("en", []):
            for d in sense.get("definitions", []):
                text = re.sub(r"<[^>]+>", "", d.get("definition", "")).strip()
                if text and not text.lower().startswith(("plural of", "alternative", "obsolete")):
                    return {"explanation": f"{form} ({sense.get('partOfSpeech', '').lower()}): {text}",
                            "source": "Wiktionary", "page": form,
                            "url": f"https://en.wiktionary.org/wiki/{urllib.parse.quote(form)}"}
    return None


def prefetch(terms: list[str]):
    """Look up, in a few batched requests, everything explain() may ask about these terms."""
    keys = list(dict.fromkeys(t[0].upper() + t[1:] for t in terms if t))
    simple = existing_titles("simple.wikipedia.org", keys)
    en = existing_titles("en.wikipedia.org", keys)
    extracts("simple.wikipedia.org", [h["title"] for h in simple.values() if not h["disambiguation"]])
    extracts("en.wikipedia.org", [h["title"] for k, h in en.items() if not h["disambiguation"]
                                  and (k not in simple or simple[k]["disambiguation"])])


def explain(term: str, paper_words: set[str]) -> dict | None:
    """The reference explanation of a term (its exact title): Simple Wikipedia, then
    Wikipedia, then Wiktionary."""
    term = term.strip()
    if len(term) < 2:
        return None
    one_word = " " not in term
    for site, label in (("simple.wikipedia.org", "Simple English Wikipedia"), ("en.wikipedia.org", "Wikipedia")):
        hit = existing_titles(site, [term[0].upper() + term[1:]]).get(term[0].upper() + term[1:])
        if not hit or hit["disambiguation"]:
            continue
        if one_word and not term.isupper() and not same_stem(term, hit["title"]):
            continue
        text = first_sentences(summary(site, hit["title"]))
        if not text or "may refer to" in text[:200]:
            continue
        overlap = len(content_words(text) & paper_words)
        return {"explanation": text, "source": label, "page": hit["title"],
                "url": f"https://{site}/wiki/{urllib.parse.quote(hit['title'].replace(' ', '_'))}",
                "sense_check": "ok" if overlap >= MIN_OVERLAP else "check: shares few words with the paper"}
    if one_word:
        w = wiktionary(term)
        if w:
            return {**w, "sense_check": "ok"}
    return None


def glossary(paper: dict, notes: dict | None = None) -> list[dict]:
    text = "\n\n".join(paper.get(s) or "" for s in SECTIONS)
    paper_words = content_words(text)
    entries = []

    skipped_common = []

    # (a) abbreviations, explained through their long form, or else its longest part that has a page
    abbr = abbreviations(text)
    parts = []
    for lf in abbr.values():
        ws = lf.split()
        if len(ws) > 1 and ws[-1].endswith("s") and not ws[-1].endswith("ss"):
            ws = ws[:-1] + [ws[-1][:-1]]
        parts += [" ".join(ws[i:i + n]) for n in range(len(ws), 0, -1) for i in range(len(ws) - n + 1)]
    prefetch(parts)
    for sf, lf in abbr.items():
        ws = lf.split()
        e = None
        if len(ws) > 1 and ws[-1].endswith("s") and not ws[-1].endswith("ss"):
            ws = ws[:-1] + [ws[-1][:-1]]                          # "batch reactors" -> "batch reactor"
        for n in range(len(ws), 0, -1):
            for i in range(len(ws) - n + 1):
                part = " ".join(ws[i:i + n])
                if n < len(ws) and n == 1 and not (difficulty(part)[0] and paper_share(part) <= MAX_PAPER_SHARE):
                    continue
                e = explain(part, paper_words)
                if e:
                    if n < len(ws):
                        e["explains_part"] = part
                    break
            if e:
                break
        e = e or {"explanation": "", "source": "", "page": "", "url": "", "sense_check": "no reference page"}
        entries.append({"term": sf, "kind": "abbreviation", "stands_for": lf, "uses": text.count(sf),
                        "paper_share": round(paper_share(sf), 4), **e})

    # (b) phrases that are Wikipedia titles and contain a hard word; longest first
    counts = candidates(text)
    cands = sorted(counts, key=lambda c: (-len(c.split()), -counts[c]))
    # the cheap local filters first (the same ones applied below), so only real candidates are looked up
    def worth_looking_up(c):
        if paper_share(c) > MAX_PAPER_SHARE or one_word_name(c, text):
            return False
        return not (" " not in c and not c[0].isupper()
                    and max(zipf_frequency(f, "en") for f in word_forms(c)) >= EVERYDAY_ZIPF)
    lookup = [c for c in cands if worth_looking_up(c)]
    titled = existing_titles("en.wikipedia.org", [c[0].upper() + c[1:] for c in lookup])
    prefetch([c for c in lookup if c[0].upper() + c[1:] in titled])
    taken: list[str] = []
    seen_forms: set[str] = set()
    general: list[str] = []
    for c in cands:
        key = c[0].upper() + c[1:]
        if key not in titled or titled[key]["disambiguation"]:
            continue
        low = c.lower()
        if any(re.search(rf"\b{re.escape(low)}\b", t) for t in taken):   # inside a longer term already kept
            continue
        if counts[c] < 2 and len(c.split()) == 1 and not difficulty(c)[0]:
            continue
        forms = set(word_forms(low))
        if one_word_name(c, text):
            continue
        if " " not in c and not c[0].isupper() and \
                max(zipf_frequency(f, "en") for f in word_forms(c)) >= EVERYDAY_ZIPF:
            general.append(c)                    # common in everyday English: the model knows it; a writing problem
            continue
        if forms & seen_forms:                                         # "pika" after "pikas"
            continue
        share = paper_share(c)
        if share > MAX_PAPER_SHARE:
            skipped_common.append(c)
            continue
        e = explain(c, paper_words)
        if not e:
            continue
        hard = [f"{w}: {difficulty(w)[1]}" for w in c.split() if difficulty(w)[0]]
        entries.append({"term": c, "kind": "term", "uses": counts[c], "why": "; ".join(hard),
                        "paper_share": round(share, 4), **e})
        taken.append(low)
        seen_forms |= forms
    if notes is not None:
        notes.update(skipped_common=skipped_common, general=general)
    return entries


def one_word_name(term: str, text: str) -> bool:
    """A capitalised word used as an author's name in citations ("Kartal et al.", "Strous, 2011")."""
    if " " in term or not term[0].isupper() or term.isupper():
        return False
    cited = len(re.findall(rf"\b{re.escape(term)}\b(?:\s+et al|,?\s+\(?(?:19|20)\d\d|\s+and\s+[A-Z])", text))
    return cited > 0 and cited >= text.count(term) / 2


# ---------------------------------------------------------------- run

def review(paper, entries, notes) -> str:
    lines = [f"## {paper['paper_id']}: {paper['title']}", "",
             f"{len(entries)} terms: {sum(e['kind'] == 'abbreviation' for e in entries)} abbreviations, "
             f"{sum(e['kind'] == 'term' for e in entries)} other terms; "
             f"{sum(e['sense_check'] != 'ok' for e in entries)} to check.", "",
             f"Left out, used in more than {MAX_PAPER_SHARE:.0%} of corpus papers (learned from training data): "
             + ", ".join(notes.get("skipped_common", [])[:80]), "",
             f"Left out, common in everyday English (Zipf ≥ {EVERYDAY_ZIPF:g}; the model knows them, rewording is a writing job): "
             + ", ".join(notes.get("general", [])), "",
             "| term | uses | papers using it | explanation (source) | check |", "|---|---|---|---|---|"]
    for e in sorted(entries, key=lambda e: -e["uses"]):
        term = f"**{e['term']}**" + (f" = {e['stands_for']}" if e.get("stands_for") else "") \
            + (f" (via *{e['explains_part']}*)" if e.get("explains_part") else "")
        expl = e["explanation"].replace("|", "/").replace("\n", " ")
        src = f" ([{e['source']}]({e['url']}))" if e["url"] else ""
        lines.append(f"| {term} | {e['uses']} | {e['paper_share']:.1%} | {expl}{src} | {'' if e['sense_check'] == 'ok' else e['sense_check']} |")
    return "\n".join(lines) + "\n"


def write_one(paper):
    try:
        entries = glossary(paper)
    except Exception as error:   # noqa: BLE001 - one odd paper must not stop the batch; it gets no glossary
        print(f"{paper['paper_id']}: failed ({type(error).__name__}: {error})", flush=True)
        entries = []
    tmp = OUT / f"{paper['paper_id']}.json.tmp"
    tmp.write_text(json.dumps(entries, indent=1, ensure_ascii=False))
    tmp.replace(OUT / f"{paper['paper_id']}.json")


def main(args: list[str]):
    import dpyr
    import eval_v3
    OUT.mkdir(parents=True, exist_ok=True)
    if args == ["--all"]:
        # every paper with a rewrite (training + validation), plus the benchmark test papers
        files = sorted((ROOT / "paper_corpus").glob("rewrites*/rewrites/*.parquet"))
        ids = {f.stem for f in files}
        papers = [p for p in dpyr.read_parquet(ROOT / "paper_corpus/papers.parquet").to_dicts()
                  if p["paper_id"] in ids] + eval_v3.test_papers()
        todo = [p for p in papers if not (OUT / f"{p['paper_id']}.json").exists()]
        print(f"{len(papers)} papers, {len(todo)} to do", flush=True)
        started = time.time()
        if OFFLINE:
            # local lookups are CPU work: one process per core (each opens its own database);
            # the rare online Wiktionary calls share 20 requests per second between them
            import multiprocessing
            import os
            from concurrent.futures import ProcessPoolExecutor
            os.environ["GLOSSARY_RATE"] = str(20 / 24)
            pool = ProcessPoolExecutor(24, mp_context=multiprocessing.get_context("spawn"))
        else:
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(40)
        with pool:
            for k, _ in enumerate(pool.map(write_one, todo, chunksize=4), 1):
                if k % 100 == 0:
                    rate = k / (time.time() - started)
                    print(f"{k}/{len(todo)} papers, {rate * 60:.0f}/min, "
                          f"{(len(todo) - k) / rate / 3600:.1f} h left", flush=True)
        print("all done", flush=True)
        return
    if args:
        papers = [p for p in dpyr.read_parquet(ROOT / "paper_corpus/papers.parquet").to_dicts()
                  if p["paper_id"] in args]
    else:
        papers = eval_v3.test_papers()[:3]
    parts = ["# Reference glossaries\n",
             f"Hard word: learned at {HARD_AGE:g} or later (Kuperman et al. 2012 age-of-acquisition ratings), "
             f"or not in that list and rarer than Zipf {RARE_ZIPF:g} (wordfreq). Explanations: first two "
             "sentences of Simple English Wikipedia, else English Wikipedia, else Wiktionary.\n"]
    for paper in papers:
        t = time.time()
        notes: dict = {}
        entries = glossary(paper, notes)
        (OUT / f"{paper['paper_id']}.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False))
        parts.append(review(paper, entries, notes))
        print(f"{paper['paper_id']}: {len(entries)} terms ({time.time() - t:.0f}s)")
    (OUT / "review.md").write_text("\n".join(parts))
    print(f"-> {OUT / 'review.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
