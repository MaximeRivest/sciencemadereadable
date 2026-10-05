"""Find 5,000 recent, openly licensed ecology & environmental-science papers,
download their full text, and split each into the sections translator.py uses.

    .venv/bin/python paper_corpus/harvest.py

Where the papers come from
- OpenAlex: 2023-2026 English research articles whose main topic is in ecology
  or environmental science, licensed CC BY, with a copy in PubMed Central.
- Europe PMC: the PMCID for each DOI, and the full text as publisher XML (JATS).

Diversity
- Candidates are drawn evenly across the 11 subfields and the four years.
- At most PER_JOURNAL papers from any one journal.
- Journals that OpenAlex mislabels as ecology (materials, sensors, chemistry...)
  are excluded.

Resuming
Everything is saved as it arrives (candidates, downloaded XML, accepted papers),
so running the script again continues where it stopped.
"""
from __future__ import annotations

import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import dpyr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rewrite_benchmark"))
from prepare import text  # the same XML-to-text rules as the benchmark

HERE = Path(__file__).resolve().parent
XML_DIR = HERE / "xml"
XML_DIR.mkdir(exist_ok=True)
CANDIDATES = HERE / "candidates.jsonl"
ACCEPTED = HERE / "papers.parquet"
REJECTED = HERE / "rejected.jsonl"
LOG = HERE / "progress.log"

TARGET = 5000
PER_JOURNAL = 150
EMAIL = "maxime.rivest@gmail.com"
PMC_SOURCE = "S2764455111"      # OpenAlex's id for PubMed Central
SUBFIELDS = {
    "1105": "Ecology, Evolution, Behavior and Systematics", "2303": "Ecology",
    "2302": "Ecological Modeling", "2304": "Environmental Chemistry", "2305": "Environmental Engineering",
    "2306": "Global and Planetary Change", "2308": "Management, Monitoring, Policy and Law",
    "2309": "Nature and Landscape Conservation", "2310": "Pollution",
    "2311": "Waste Management and Disposal", "2312": "Water Science and Technology",
}
YEARS = ["2023", "2024", "2025", "2026"]
# A journal is kept when enough of what it publishes is in these fields
# (OpenAlex's own tally of the journal's topics), or when it is a broad
# journal that publishes every field.
NATURE_FIELDS = {"Environmental Science", "Agricultural and Biological Sciences",
                 "Earth and Planetary Sciences", "Immunology and Microbiology"}
MIN_NATURE_SHARE = 0.30
BROAD_JOURNALS = {"PLoS ONE", "Scientific Reports", "Nature Communications", "PeerJ", "Heliyon",
                  "Science Advances", "iScience", "Royal Society Open Science", "eLife",
                  "Proceedings of the National Academy of Sciences", "PNAS Nexus", "Communications Biology"}
# Short genome notices, not research papers.
OFF_TOPIC = {"Microbiology Resource Announcements"}
MIN_WORDS, MAX_WORDS = 2500, 20000


def log(message):
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def fetch(url, tries=5):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as error:
            code = getattr(error, "code", None)
            if code and code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"gave up on {url}")


def openalex(**params):
    params["mailto"] = EMAIL
    return json.loads(fetch("https://api.openalex.org/works?" + urllib.parse.urlencode(params)))


# ---------------------------------------------------------------- 1. candidates

def gather_candidates(per_cell=400):
    """Up to per_cell random candidates for each subfield x year (11 x 4 cells)."""
    if CANDIDATES.exists():
        return [json.loads(l) for l in CANDIDATES.read_text().splitlines()]
    rows = []
    for sub in SUBFIELDS:
        for year in YEARS:
            f = (f"publication_year:{year},type:article,language:en,primary_topic.subfield.id:{sub},"
                 f"best_oa_location.license:cc-by,locations.source.id:{PMC_SOURCE}")
            results = []
            for page in (1, 2):   # OpenAlex returns at most 200 per page
                data = openalex(filter=f, sample=per_cell, seed=7, per_page=200, page=page,
                                select="id,doi,title,publication_year,primary_location,primary_topic")
                results += data["results"]
                if len(data["results"]) < 200:
                    break
            for w in results:
                source = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
                if not w.get("doi") or source in OFF_TOPIC:
                    continue
                source_id = ((w.get("primary_location") or {}).get("source") or {}).get("id", "")
                rows.append({"openalex_id": w["id"].rsplit("/", 1)[-1], "doi": w["doi"].replace("https://doi.org/", ""),
                             "title": w["title"], "year": w["publication_year"], "journal": source,
                             "journal_id": source_id.rsplit("/", 1)[-1], "subfield": SUBFIELDS[sub]})
            log(f"candidates: {SUBFIELDS[sub]} {year}: {len(results)} (total {len(rows)})")
            time.sleep(0.2)
    rows = keep_nature_journals(rows)
    random.Random(7).shuffle(rows)
    CANDIDATES.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def keep_nature_journals(rows):
    """Drop papers from journals that mostly publish other fields."""
    ids = sorted({r["journal_id"] for r in rows if r["journal_id"]})
    share = {}
    for i in range(0, len(ids), 50):
        data = json.loads(fetch("https://api.openalex.org/sources?" + urllib.parse.urlencode(
            {"filter": "openalex_id:" + "|".join(ids[i:i + 50]), "select": "id,topics",
             "per_page": 50, "mailto": EMAIL})))
        for s in data["results"]:
            topics = s.get("topics") or []
            total = sum(t["count"] for t in topics) or 1
            share[s["id"].rsplit("/", 1)[-1]] = sum(t["count"] for t in topics
                                                    if t["field"]["display_name"] in NATURE_FIELDS) / total
        time.sleep(0.2)
    kept = [r for r in rows if r["journal"] in BROAD_JOURNALS or share.get(r["journal_id"], 0) >= MIN_NATURE_SHARE]
    dropped = Counter(r["journal"] for r in rows if r not in kept)
    log(f"journal filter: kept {len(kept)} of {len(rows)}; most dropped: {dropped.most_common(12)}")
    return kept


# ---------------------------------------------------------------- 2. PMCIDs

def add_pmcids(rows):
    """DOI -> PMCID with Europe PMC's search (NCBI's converter rate-limits hard)."""
    missing = [r for r in rows if "pmcid" not in r]
    for i in range(0, len(missing), 25):
        batch = missing[i:i + 25]
        query = " OR ".join(f'DOI:"{r["doi"]}"' for r in batch)
        url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
            {"query": query, "format": "json", "resultType": "lite", "pageSize": 100}))
        try:
            results = json.loads(fetch(url))["resultList"]["result"]
            found = {(x.get("doi") or "").lower(): x.get("pmcid") for x in results
                     if x.get("pmcid") and x.get("isOpenAccess") == "Y"}
        except Exception as error:   # skip this batch; the papers are simply not used
            log(f"ID lookup failed for one batch ({type(error).__name__}); skipping it")
            found = {}
        for r in batch:
            r["pmcid"] = found.get(r["doi"].lower())
        time.sleep(0.1)
        if (i // 25) % 80 == 0:
            log(f"ID lookup {min(i + 25, len(missing))} of {len(missing)}")
    CANDIDATES.write_text("".join(json.dumps(r) + "\n" for r in rows))
    log(f"PMCIDs found for {sum(1 for r in rows if r.get('pmcid'))} of {len(rows)}")


# ---------------------------------------------------------------- 3. full text

def download(pmcids):
    """Publisher XML from Europe PMC, several at once, cached on disk."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [p for p in pmcids if not (XML_DIR / f"{p}.xml").exists()]

    def one(pmcid):
        try:
            data = fetch(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
            ET.fromstring(data)   # only save well-formed XML
            (XML_DIR / f"{pmcid}.xml").write_bytes(data)
        except Exception:
            pass   # no full text: counted as rejected later

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, todo))
    if todo:
        log(f"downloaded {sum((XML_DIR / f'{p}.xml').exists() for p in todo)} of {len(todo)}")


# ---------------------------------------------------------------- 4. sections

KINDS = [("introduction", ("introduction", "background")), ("methods", ("method", "material")),
         ("results", ("result",)), ("discussion", ("discussion",)), ("conclusion", ("conclusion",))]


def split_sections(root):
    """The translator's sections, from a JATS article. None if it is not a normal
    research paper (no introduction, methods or results, or no body text)."""
    meta = root.find("./front/article-meta")
    found, order = {}, []
    for sec in root.findall("./body/sec"):
        heading = (sec.findtext("title") or "").lower()
        for kind, words in KINDS:
            if kind not in found and any(w in heading for w in words):
                found[kind] = sec
                order.append(kind)
                break
    if not all(k in found for k in ("introduction", "methods", "results")) or meta is None:
        return None
    intro = found["introduction"]
    paragraphs = intro.findall("./p")
    sections = {
        "title": text(meta.find("./title-group/article-title")),
        "abstract": text(meta.find("abstract")) if meta.find("abstract") is not None else "",
        "introduction_first": text(paragraphs[0]) if paragraphs else "",
        "introduction_rest": "\n\n".join([text(p) for p in paragraphs[1:]] + [text(s) for s in intro.findall("./sec")]),
    }
    for kind in ("methods", "results", "discussion", "conclusion"):
        sections[kind] = text(found[kind]) if kind in found else ""
    # "Results and discussion" together: the results section holds both.
    sections["original_section_order"] = ",".join(order)
    return sections


def license_ok(root):
    """CC BY (any version), stated in the licence element's link or its text.
    Non-commercial or no-derivatives licences (by-nc, by-nd, by-sa...) are refused."""
    for lic in root.iter():
        if not lic.tag.endswith("license"):
            continue
        links = [v for e in lic.iter() for v in e.attrib.values()]
        blob = (" ".join(links) + " " + " ".join(lic.itertext())).lower()
        if any(x in blob for x in ("licenses/by-nc", "licenses/by-nd", "licenses/by-sa", "noncommercial", "no derivatives")):
            return False
        if "creativecommons.org/licenses/by/" in blob or "creative commons attribution" in blob \
                or "cc by license" in blob or "(cc by)" in blob:
            return True
        if "broadest form of re-use" in blob and "commercial" in blob:   # ACS AuthorChoice CC-BY wording
            return True
    return False


# ---------------------------------------------------------------- main

def main():
    rows = gather_candidates()
    if any("pmcid" not in r for r in rows):
        add_pmcids(rows)
    rows = [r for r in rows if r.get("pmcid")]

    # Balanced order: round-robin over subfield x year cells, so the first 5,000
    # accepted are spread evenly even if some cells run dry.
    cells = {}
    for r in rows:
        cells.setdefault((r["subfield"], r["year"]), []).append(r)
    ordered = []
    while any(cells.values()):
        for key in list(cells):
            if cells[key]:
                ordered.append(cells[key].pop())

    accepted, rejected = [], []
    per_journal = Counter()
    for start in range(0, len(ordered), 200):
        if len(accepted) >= TARGET:
            break
        chunk = [r for r in ordered[start:start + 200] if per_journal[r["journal"]] < PER_JOURNAL]
        download([r["pmcid"] for r in chunk])
        for r in chunk:
            if len(accepted) >= TARGET:
                break
            if per_journal[r["journal"]] >= PER_JOURNAL:
                continue
            path = XML_DIR / f"{r['pmcid']}.xml"
            reason = None
            if not path.exists():
                reason = "no full text in Europe PMC"
            else:
                root = ET.fromstring(path.read_bytes())
                sections = split_sections(root)
                words = sum(len(v.split()) for k, v in (sections or {}).items() if k != "original_section_order")
                if sections is None:
                    reason = "no introduction, methods and results sections"
                elif not license_ok(root):
                    reason = "not CC BY in the full text"
                elif not MIN_WORDS <= words <= MAX_WORDS:
                    reason = f"length {words} words"
            if reason:
                rejected.append({**r, "reason": reason})
                continue
            per_journal[r["journal"]] += 1
            accepted.append({"paper_id": r["pmcid"], **{k: r[k] for k in ("doi", "openalex_id", "journal", "year", "subfield")},
                             "words": words, **sections})
        dpyr.read(accepted).write_parquet(ACCEPTED)
        REJECTED.write_text("".join(json.dumps(x) + "\n" for x in rejected))
        log(f"accepted {len(accepted)} · rejected {len(rejected)} · journals {len(per_journal)}")

    reasons = Counter(x["reason"].split(" ")[0] if x["reason"].startswith("length") else x["reason"] for x in rejected)
    log(f"DONE: {len(accepted)} papers from {len(per_journal)} journals. Rejected: {dict(reasons)}")


if __name__ == "__main__":
    main()
