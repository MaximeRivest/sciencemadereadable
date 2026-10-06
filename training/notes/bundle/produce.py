"""
The 300-paper run: bundles and texts for training
=================================================

Makes the training data: for each paper, the bundle (the 4e recipe) and the accessible text
written from it, with every step saved so that a writer can be trained (bundle -> text) and,
later, a bundle-maker (paragraph -> facts, paper -> things and glossary) and checkers.

Two teachers, half the papers each: Opus (Claude subscription) and Astra (ChatGPT/Codex
subscription). Each paper is made entirely by one teacher.

Per paper (all steps of 4e, plus a check-and-revise loop on the written text):
  1. code: the paper's structure, verbatim pieces, reference list (structure.py, sources.py);
     25% of papers get invented species names (as in v3)
  2. teacher: things + new glossary words (cache per teacher) -> glossary check -> corrections
  3. teacher: each part's paragraphs -> dense facts
     code check + model check on every part -> fix if anything -> code check again;
     after a major problem, the model checks again and fixes once more if needed
  4. teacher: writes the opening, then the 4 sections (with the opening as already read)
  5. code form check + trace -> if anything: the teacher revises, and both run again

Saved per paper, in out/data300/<teacher>/<paper_id>/:
  record.json    selection, metadata, renaming, offline-Wikipedia terms
  paper.json     structure (headings, numbered paragraphs, citations [n], placement), verbatim pieces
  calls.jsonl    every model call: step, function, teacher, settings, full inputs, full output,
                 seconds, errors (the raw material for training a bundle-maker or a checker)
  bundle.json    front (raw, check, final), per part: notes (first, checks, fixes, final),
                 the assembled parts, and bundle.txt (exactly what the writer read)
  texts.json     per part: first draft, form check, trace, revision (if any) and final text
  done.json      summary, timings, prompt-file hash, git commit

Limits:
  Opus   stops starting papers at STOP_CLAUDE_WEEKLY % of the weekly allowance; waits when the
         5-hour window is nearly full.
  Astra  stops at STOP_CODEX_WEEKLY % of the weekly allowance, or the moment the credit
         balance drops below its value at start (so it never spends credits); the reason is
         written to out/data300/STATUS-astra.txt.

    .venv/bin/python training/notes/bundle/produce.py select           # choose the 300, once
    .venv/bin/python training/notes/bundle/produce.py run opus         # resumable
    .venv/bin/python training/notes/bundle/produce.py run astra
    .venv/bin/python training/notes/bundle/produce.py status
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import lm15

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "paper_corpus"))
sys.path.insert(0, str(ROOT / "training/v3"))
import cache                                        # noqa: E402
import checks                                       # noqa: E402
import prompts as PR                                # noqa: E402
import render                                       # noqa: E402
import structure                                    # noqa: E402
from sources import XML, verbatim                   # noqa: E402
from usage import claude_usage, codex_usage         # noqa: E402

# --------------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------------

OUT = HERE / "out" / "data300"
N_PAPERS = 300
SEED = 2026
PAPER_WORDS = (3000, 7000)
RENAME_SHARE = 0.25
PAPERS_AT_ONCE = 3
STOP_CLAUDE_WEEKLY = 97            # % (the user allowed 100%; 3% left for papers in flight)
STOP_CODEX_WEEKLY = 97             # % ; above 100% the Codex plan would spend credits
HELD_OUT = {"PMC12291832", "PMC12010427", "PMC7616100", "PMC12466960", "PMC11227166",   # the 5 test papers
            "PMC10203038",                                                             # the prompt examples
            "PMC13153078", "PMC10234942", "PMC11117235"}                               # v3's 3-paper pilot

ASTRA = dict(lm="openai-codex:gpt-6-astra", reasoning=lm15.Reasoning(effort="low"))
TEACHERS = {
    "opus": {"main": PR.OPUS, "check": PR.OPUS_CHECK},
    "astra": {"main": ASTRA, "check": ASTRA},       # "low" is Astra's lowest reasoning level
}
STEPS = {   # step -> (AI function, settings key)
    "front": (PR.make_front_cached, "main"), "check glossary": (PR.check_glossary, "check"),
    "notes": (PR.make_paragraph_notes, "main"), "check notes": (PR.check_part, "check"),
    "fix notes": (PR.fix_paragraph_notes, "main"), "write": (PR.write_part, "main"),
    "trace": (PR.trace, "check"), "revise": (PR.revise_part, "main"),
}


def functions_for(teacher: str) -> dict:
    return {step: fn.using(**TEACHERS[teacher][key]) for step, (fn, key) in STEPS.items()}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def prompt_hash() -> str:
    return hashlib.sha256((HERE / "prompts.py").read_bytes()).hexdigest()[:12]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------------------------

class StopRun(Exception):
    """A limit was reached: no new calls."""


class Guard:
    """Checks the subscription before each call (at most once a minute)."""

    def __init__(self, teacher: str):
        self.teacher, self.stopped, self.reason, self.last, self.lock = teacher, False, "", 0.0, threading.Lock()
        self.credits_at_start = None
        if teacher == "astra":
            u = codex_usage_full()
            self.credits_at_start = u["credits"]
            self.status(f"started; weekly {u['weekly']}%, credits {u['credits']}")

    def status(self, text: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"STATUS-{self.teacher}.txt").write_text(f"{now()} {text}\n")

    def check(self, force: bool = False) -> None:
        with self.lock:
            if self.stopped:
                raise StopRun(self.reason)
            if not force and time.time() - self.last < 60:
                return
            self.last = time.time()
            try:
                if self.teacher == "opus":
                    u = claude_usage()
                    if u["seven_day"] >= STOP_CLAUDE_WEEKLY:
                        self.stop(f"Claude weekly allowance at {u['seven_day']}%")
                    while u["five_hour"] >= 95:                         # wait for the 5-hour window
                        self.status(f"waiting for the 5-hour window ({u['five_hour']}%, resets {u['five_hour_resets']})")
                        time.sleep(600)
                        u = claude_usage()
                else:
                    u = codex_usage_full()
                    if u["credits"] is not None and self.credits_at_start is not None and u["credits"] < self.credits_at_start:
                        self.stop(f"Codex credits started to be used ({self.credits_at_start} -> {u['credits']})")
                    elif u["limit_reached"] or u["weekly"] >= STOP_CODEX_WEEKLY:
                        self.stop(f"Codex weekly allowance at {u['weekly']}% (credits untouched: {u['credits']})")
            except StopRun:
                raise
            except Exception as error:                                      # noqa: BLE001  usage endpoint down
                self.status(f"could not read usage ({type(error).__name__}); continuing")

    def stop(self, reason: str) -> None:
        self.stopped, self.reason = True, reason
        self.status(f"STOPPED: {reason}")
        raise StopRun(reason)


def codex_usage_full() -> dict:
    """Weekly % and the credit balance (codex_usage plus the credits field)."""
    import urllib.request
    tokens = json.loads((Path.home() / ".codex" / "auth.json").read_text())["tokens"]
    request = urllib.request.Request("https://chatgpt.com/backend-api/wham/usage", headers={
        "Authorization": f"Bearer {tokens['access_token']}", "chatgpt-account-id": tokens.get("account_id") or "",
        "User-Agent": "codex_cli_rs", "originator": "codex_cli_rs"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    credits = (data.get("credits") or {}).get("balance")
    return {"weekly": data["rate_limit"]["primary_window"]["used_percent"],
            "limit_reached": data["rate_limit"]["limit_reached"],
            "credits": float(credits) if credits is not None else None}


# --------------------------------------------------------------------------------------------
# One logged model call
# --------------------------------------------------------------------------------------------

class Calls:
    """Every call of one paper, appended to calls.jsonl as it happens."""

    def __init__(self, folder: Path, teacher: str, guard: Guard):
        self.path, self.teacher, self.guard = folder / "calls.jsonl", teacher, guard
        self.fns, self.lock, self.n = functions_for(teacher), threading.Lock(), 0

    def ask(self, step: str, part: str = "", **inputs):
        fn = self.fns[step]
        errors = []
        for attempt in range(12):
            self.guard.check()
            t = time.time()
            try:
                result = fn(**inputs)
                out = result.model_dump() if hasattr(result, "model_dump") else result
                self.log({"step": step, "part": part, "function": STEPS[step][0].__name__, "teacher": self.teacher,
                          "settings": {k: str(v) for k, v in TEACHERS[self.teacher][STEPS[step][1]].items()
                                       if k != "client"},
                          "inputs": inputs, "output": out, "seconds": round(time.time() - t, 1),
                          "errors_before": errors, "time": now()})
                return result
            except lm15.RateLimitError as error:
                errors.append(f"rate limit: {str(error)[:200]}")
                self.guard.check(force=True)
                time.sleep(300)
            except Exception as error:                                       # noqa: BLE001
                errors.append(f"{type(error).__name__}: {str(error)[:300]}")
                if attempt >= 3 and not ("Overloaded" in str(error) or "529" in str(error)
                                         or type(error).__name__ in ("ServerError", "TransportError")):
                    raise
                time.sleep(30 * (attempt + 1))
        raise RuntimeError(f"{step}: the model kept failing: {errors[-3:]}")

    def log(self, row: dict) -> None:
        with self.lock:
            self.n += 1
            with self.path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------------
# Choosing the 300 papers (once)
# --------------------------------------------------------------------------------------------

def select() -> None:
    import dpyr
    taboo = set((ROOT / "training/data/validation_ids.txt").read_text().split())
    taboo |= set((ROOT / "paper_corpus/test_ids.txt").read_text().split()) | HELD_OUT
    rows = [p for p in dpyr.read_parquet(ROOT / "paper_corpus/papers.parquet").to_dicts()
            if p["paper_id"] not in taboo and PAPER_WORDS[0] <= p["words"] <= PAPER_WORDS[1] and p.get("discussion")
            and (XML / f"{p['paper_id']}.xml").exists()]
    random.Random(SEED).shuffle(rows)
    by_field: dict[str, list] = {}
    for p in rows:
        by_field.setdefault(p["subfield"], []).append(p)
    chosen, refused = [], []
    while len(chosen) < N_PAPERS and any(by_field.values()):
        for field in sorted(by_field):
            if not by_field[field] or len(chosen) >= N_PAPERS:
                continue
            p = by_field[field].pop()
            reason = quality_problem(p)
            (refused.append({"paper_id": p["paper_id"], "reason": reason}) if reason else chosen.append(p))
    papers = [{"paper_id": p["paper_id"], "subfield": p["subfield"], "words": p["words"], "journal": p["journal"],
               "year": p["year"], "teacher": "opus" if i % 2 == 0 else "astra"} for i, p in enumerate(chosen)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "papers.json").write_text(json.dumps({"made": now(), "seed": SEED, "rules": {
        "words": PAPER_WORDS, "excluded": "validation, test, the 5 test papers of rounds 1-4, the prompt-example paper, v3's 3-paper pilot",
        "needs": "XML; introduction, methods, results and discussion with paragraphs; >= 10 references and >= 5 linked citations; >= 1 figure or table",
        "order": "round robin over subfields; teachers alternate (opus, astra, ...)"},
        "papers": papers, "refused": refused}, indent=1))
    print(f"{len(papers)} papers chosen ({sum(p['teacher'] == 'opus' for p in papers)} opus, "
          f"{sum(p['teacher'] == 'astra' for p in papers)} astra); {len(refused)} refused")


def quality_problem(p: dict) -> str:
    """Why a paper would make poor training data (None if fine)."""
    xml = (XML / f"{p['paper_id']}.xml").read_text()
    if len(re.findall(r"<ref[ >]", xml)) < 10:
        return "fewer than 10 references"
    if len(re.findall(r'<xref[^>]*ref-type="bibr"', xml)) < 5:
        return "citations not linked in the XML"
    if not re.search(r"<fig[ >]|<table-wrap[ >]", xml):
        return "no figure or table"
    s = structure.paper_structure({"paper_id": p["paper_id"], "renamed": []})
    for part in ("introduction_rest", "methods", "results", "discussion"):
        if not s["parts"].get(part, {}).get("paragraphs"):
            return f"no paragraphs found for {part}"
    if not s["parts"].get("opening", {}).get("paragraphs"):
        return "no first introduction paragraph or conclusion"
    return None


def make_record(row: dict, paper: dict) -> dict:
    """What the v3 records held, for a new paper: originals per part, renaming, Wikipedia terms."""
    from build import apply_map, candidate_terms, taxa_map
    rng = random.Random(int(hashlib.sha256(f"{SEED}{row['paper_id']}".encode()).hexdigest(), 16))
    sections = ["title", "abstract", "introduction_first", "introduction_rest", "methods", "results", "discussion", "conclusion"]
    renamed = []
    if rng.random() < RENAME_SHARE:
        whole = "\n\n".join(paper.get(s) or "" for s in sections)
        mapping, renamed = taxa_map(whole, rng)
        if mapping:
            paper = {k: apply_map(v, mapping) if isinstance(v, str) else v for k, v in paper.items()}
            renamed = renamed or [{"forms": mapping}]
    if renamed and not all("forms" in r for r in renamed):
        renamed = [{"forms": mapping, "info": renamed}]
    terms = candidate_terms(paper)
    mapping = {f: n for r in renamed for f, n in r["forms"].items()}
    for t in terms:
        for key in ("term", "context", "reference"):
            if t.get(key):
                t[key] = apply_map(t[key], mapping)
    opening = "\n\n".join(f"## {s}\n\n{paper[s]}" for s in ("title", "abstract", "introduction_first", "conclusion") if paper.get(s))
    convs = [{"section": "opening", "original": opening}] + [
        {"section": s, "original": paper[s]} for s in ("introduction_rest", "methods", "results", "discussion") if paper.get(s)]
    return {**row, "title": paper["title"], "doi": paper.get("doi"), "licence": "CC BY (checked at harvest)",
            "renamed": renamed, "terms": terms, "conversations": convs}


# --------------------------------------------------------------------------------------------
# One paper
# --------------------------------------------------------------------------------------------

def do_paper(row: dict, teacher: str, gloss: dict, gloss_lock: threading.Lock, guard: Guard) -> dict:
    import dpyr
    pid = row["paper_id"]
    folder = OUT / teacher / pid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "calls.jsonl").unlink(missing_ok=True)                        # a fresh start for an unfinished paper
    started = time.time()
    paper = next(p for p in dpyr.read_parquet(ROOT / "paper_corpus/papers.parquet").to_dicts() if p["paper_id"] == pid)
    record = make_record(row, paper)
    (folder / "record.json").write_text(json.dumps(record, indent=1, ensure_ascii=False))
    calls = Calls(folder, teacher, guard)

    v = verbatim(record)
    s = structure.paper_structure(record)
    names = [n for n in render.PARTS if s["parts"].get(n, {}).get("paragraphs")]
    for f in v["figures"]:
        if f["id"] in s["placement"]:
            f["part"] = s["placement"][f["id"]][0]
    texts_of = {n: structure.part_text(s["parts"][n]) for n in names}
    whole = "\n\n".join(texts_of.values())
    all_figures = "\n\n".join(render._figure(f) for f in v["figures"]) or "(none)"
    figs = lambda n: "\n\n".join(render._figure(f) for f in v["figures"] if f["part"] == n) or "(none)"
    rest = lambda n: "\n\n".join(texts_of[m] for m in names if m != n)
    (folder / "paper.json").write_text(json.dumps({"structure": s, "verbatim": v, "part_texts": texts_of}, indent=1, ensure_ascii=False))
    reference_glossary = "\n".join(f"- {t['term']}" + (f" ({t['stands_for']})" if t.get("stands_for") else "")
                                   + f": {t['reference'][:300]}" for t in record["terms"] if t.get("reference"))

    # 2. front, glossary check, corrections, cache
    with gloss_lock:
        known = cache.known_in(gloss, v["title"] + " " + v["abstract"] + " " + whole)
    fr = calls.ask("front", paper=whole, figures_and_tables=all_figures,
                   already_defined="\n".join(f"- {e['term']}: {e['meaning']}" for e in known) or "(none)",
                   reference_glossary=reference_glossary + "\n\nStatistics sheet:\n" + PR.STATISTICS_SHEET,
                   reader=PR.READER, example=PR.EXAMPLE_FRONT).model_dump()
    new_terms = {d["term"].lower() for d in fr["glossary"]}
    central = {c["term"].lower(): c["picture"] for c in fr["central_known"]}
    glossary = [{"term": e["term"], "meaning": e["meaning"], "central": e["term"].lower() in central,
                 "picture": central.get(e["term"].lower(), ""), "from_cache": True}
                for e in known if e["term"].lower() not in new_terms]
    glossary += [{**d, "from_cache": False} for d in fr["glossary"]]
    things_list = [dict(t) for t in fr["things"]]
    gtext = "\n".join(f"- {d['term']}: {d['meaning']}" + (f" (picture: {d['picture']})" if d["picture"] else "") for d in glossary)
    gcheck = [x.model_dump() for x in calls.ask("check glossary", paper=whole,
              things="\n".join(f"- {t['name']}: {t['what']}" for t in things_list), glossary=gtext, reader=PR.READER).problems]
    for x in gcheck:
        key = x["entry"].strip().lower()
        if x["field"] == "thing":
            for t in things_list:
                if t["name"].lower() == key:
                    t["what"] = x["correction"]
        else:
            for d in glossary:
                if d["term"].lower() == key:
                    d["meaning" if x["field"] == "meaning" else "picture"] = x["correction"]
    things_list = [t for t in things_list if t["what"]]
    glossary = [d for d in glossary if d["meaning"]]
    with gloss_lock:
        added = cache.add(gloss, [d for d in glossary if not d["from_cache"]], pid)
        cache.PATH = OUT / teacher / "glossary_cache.json"
        cache.save(gloss)
    front = {"things": things_list, "glossary": glossary}
    things = "\n".join(f"- {t['name']}: {t['what']}" for t in things_list)

    with ThreadPoolExecutor(len(names) + 1) as pool:
        # 3. notes, checks, fixes
        def part_notes(n):
            log = {}
            nt = calls.ask("notes", n, part_name=n, id_letter=render.LETTER[n], original=texts_of[n], things=things,
                           figures_and_tables_here=figs(n), rest_of_paper=rest(n), example=PR.EXAMPLE_PARAGRAPHS).model_dump()
            log["first"] = nt
            for round_ in (1, 2):
                code = checks.part_problems(s["parts"][n], nt)
                model = [x.model_dump() for x in calls.ask("check notes", n, original=texts_of[n],
                         bundle_part=render.part_view(v, front, n, assemble(n, nt)), rest_of_paper=rest(n)).problems]
                log[f"check_{round_}"] = {"code": code, "model": model}
                if not code and not model:
                    break
                problems = "\n".join([f"- ¶{x['paragraph']}: {x['kind']}: {x['detail']}" for x in code]
                                     + [f"- {x['kind']} ({x['severity']}): {x['detail']} [{x['quote']}]" for x in model])
                nt = calls.ask("fix notes", n, original=texts_of[n], notes=json.dumps(nt, ensure_ascii=False),
                               problems=problems, things=things, rest_of_paper=rest(n)).model_dump()
                log[f"fix_{round_}"] = nt
                if not any(x["severity"] == "major" for x in model):          # a second round only after a major problem
                    break
            log["code_check_final"] = checks.part_problems(s["parts"][n], nt)
            log["final"] = nt
            return log

        def assemble(n, nt):
            placed = {}
            for fid, (part, para) in s["placement"].items():
                if part == n:
                    placed.setdefault(para, []).append(fid)
            unplaced = [f["id"] for f in v["figures"] if f["part"] == n and f["id"] not in s["placement"]]
            from pilot_4d import build_part
            return build_part(n, s["parts"][n], nt, placed, unplaced)

        notes_log = dict(zip(names, pool.map(part_notes, names)))
        parts = {n: assemble(n, notes_log[n]["final"]) for n in names}
        bundle_text = render.bundle(v, front, parts, render.sheets(whole))
        (folder / "bundle.json").write_text(json.dumps({
            "front_raw": fr, "glossary_check": gcheck, "front": front, "glossary_added_to_cache": added,
            "notes": notes_log, "parts": parts}, indent=1, ensure_ascii=False))
        (folder / "bundle.txt").write_text(bundle_text)

        # 4-5. write, check, revise
        import pilot as P4
        expected = lambda n: [f["id"] for f in v["figures"] if f["part"] == n]

        def checked_text(n, text):
            form = P4.form_checks(text, parts[n], expected(n), len(v["references"]))
            tr = calls.ask("trace", n, items=render.trace_view(v, n, parts[n]), text=text,
                           allowed_sources=bundle_text, reader=PR.READER).model_dump()
            return form, tr

        def problems_of(form, tr):
            out = [f"- form: {k}: {val}" for k, val in form.items() if val]
            out += [f"- {x['id']} {x['status']}: {x['note']}" for x in tr["facts"] if x["status"] != "kept"]
            out += [f"- added ({x['kind']}): \"{x['quote']}\" {x['note']}" for x in tr["additions"]]
            return out

        def write_part(n, already):
            first = calls.ask("write", n, bundle=bundle_text, part_name=n, already_read=already,
                              brief=PR.BRIEF, reader=PR.READER).text
            form, tr = checked_text(n, first)
            log = {"first": first, "first_form": form, "first_trace": tr, "final": first}
            todo = problems_of(form, tr)
            if todo:
                revised = calls.ask("revise", n, bundle=bundle_text, part_name=n, text=first, problems="\n".join(todo),
                                    brief=PR.BRIEF, reader=PR.READER).text
                form2, tr2 = checked_text(n, revised)
                log.update({"problems": todo, "revised": revised, "revised_form": form2, "revised_trace": tr2})
                if len(problems_of(form2, tr2)) <= len(todo):
                    log["final"] = revised
            return log

        texts = {names[0]: write_part(names[0], "")}
        opening_text = texts[names[0]]["final"]
        texts.update(dict(zip(names[1:], pool.map(lambda n: write_part(n, opening_text), names[1:]))))
    (folder / "texts.json").write_text(json.dumps(texts, indent=1, ensure_ascii=False))

    summary = {"paper_id": pid, "teacher": teacher, "finished": now(), "minutes": round((time.time() - started) / 60, 1),
               "calls": calls.n, "prompt_hash": prompt_hash(), "git_commit": git_commit(),
               "renamed": bool(record["renamed"]), "facts": sum(len(parts[n]["facts"]) for n in names),
               "glossary": {"entries": len(glossary), "from_cache": sum(d["from_cache"] for d in glossary),
                            "check_corrections": len(gcheck)},
               "notes_major_problems_first_check": sum(any(x["severity"] == "major" for x in notes_log[n]["check_1"]["model"])
                                                       for n in names),
               "code_problems_left": sum(len(notes_log[n]["code_check_final"]) for n in names),
               "revised_parts": sum("revised" in texts[n] for n in names),
               "problems_left": {n: len(problems_of(texts[n]["revised_form"], texts[n]["revised_trace"]))
                                 if texts[n]["final"] == texts[n].get("revised") else
                                 len(problems_of(texts[n]["first_form"], texts[n]["first_trace"])) for n in names}}
    (folder / "done.json").write_text(json.dumps(summary, indent=1))
    return summary


# --------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------

def run(teacher: str) -> None:
    rows = [p for p in json.loads((OUT / "papers.json").read_text())["papers"] if p["teacher"] == teacher]
    todo = [p for p in rows if not (OUT / teacher / p["paper_id"] / "done.json").exists()]
    guard = Guard(teacher)
    cache.PATH = OUT / teacher / "glossary_cache.json"
    (OUT / teacher).mkdir(parents=True, exist_ok=True)
    gloss, gloss_lock = cache.load(), threading.Lock()
    (OUT / f"run-{teacher}.json").write_text(json.dumps({"started": now(), "git_commit": git_commit(),
                                                         "prompt_hash": prompt_hash(), "to_do": len(todo)}, indent=1))
    print(f"{teacher}: {len(rows)} papers, {len(todo)} to do", flush=True)

    def one(row):
        if guard.stopped:
            return
        try:
            guard.check(force=True)
            r = do_paper(row, teacher, gloss, gloss_lock, guard)
            print(f"{now()} done {row['paper_id']} · {r['calls']} calls · {r['minutes']} min · "
                  f"revised {r['revised_parts']} parts", flush=True)
        except StopRun as stop:
            print(f"{now()} not started or stopped {row['paper_id']}: {stop}", flush=True)
        except Exception as error:                                           # noqa: BLE001  one paper must not stop the run
            print(f"{now()} FAILED {row['paper_id']}: {type(error).__name__}: {str(error)[:300]}", flush=True)
            (OUT / teacher / row["paper_id"] / "error.txt").write_text(traceback.format_exc())

    with ThreadPoolExecutor(PAPERS_AT_ONCE) as pool:
        list(pool.map(one, todo))
    done = sum((OUT / teacher / p["paper_id"] / "done.json").exists() for p in rows)
    guard.status(f"finished this run: {done} of {len(rows)} papers done" + (f"; {guard.reason}" if guard.stopped else ""))
    print(f"{teacher}: {done} of {len(rows)} done" + (f" · stopped: {guard.reason}" if guard.stopped else ""), flush=True)


def status() -> None:
    papers = json.loads((OUT / "papers.json").read_text())["papers"]
    for teacher in TEACHERS:
        rows = [p for p in papers if p["teacher"] == teacher]
        done = [json.loads((OUT / teacher / p["paper_id"] / "done.json").read_text())
                for p in rows if (OUT / teacher / p["paper_id"] / "done.json").exists()]
        failed = sum((OUT / teacher / p["paper_id"] / "error.txt").exists() and
                     not (OUT / teacher / p["paper_id"] / "done.json").exists() for p in rows)
        st = (OUT / f"STATUS-{teacher}.txt").read_text().strip() if (OUT / f"STATUS-{teacher}.txt").exists() else ""
        calls = sum(d["calls"] for d in done)
        print(f"{teacher}: {len(done)}/{len(rows)} done, {failed} failed, {calls} calls"
              + (f", {calls / len(done):.0f} per paper" if done else "") + f"\n   {st}")
    u, c = claude_usage(), codex_usage_full()
    print(f"Claude weekly {u['seven_day']}% (5-hour {u['five_hour']}%) · Codex weekly {c['weekly']}%, credits {c['credits']}")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"select": select, "status": status}.get(command, lambda: run(sys.argv[2]))()
