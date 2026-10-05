"""Requests the Python programs send, for checking the TypeScript twins (check_prompts.ts).

    .venv/bin/python app/tools/python_requests.py

Uses the functai version the scored rewrites and the students' training data were made with.
Writes app/tools/fixtures.json: for each program, the inputs, the model, and the request."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "glossary"))
import eval_v3, translator                                     # noqa: E402
from inputs import glossary_text                                # noqa: E402
import importlib.util                                           # noqa: E402
_spec = importlib.util.spec_from_file_location("student_prepare", ROOT / "training/prepare.py")
student = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(student)

paper = eval_v3.test_papers()[0]
pid = paper["paper_id"]
text = translator.whole_paper_text(paper) if hasattr(translator, "whole_paper_text") else None
student_text = "\n\n".join(f"## {s}\n\n{paper[s]}" for s in translator.ALL_SECTIONS if paper.get(s))
glossary = [{"source_term": "structural equation model", "plain_term": "a map of causes",
             "explanation": "A way to test how several things affect each other at once."}]
opening_md = "## title\n\nA plain title\n\n## abstract\n\nA plain abstract."

def req(fn, lm, **inputs):
    r = fn.using(lm=lm).render(**inputs)
    d = r.to_dict() if hasattr(r, "to_dict") else json.loads(json.dumps(r, default=lambda o: getattr(o, "__dict__", str(o))))
    return d

cases = {
  "rewriteOpening": (translator._opening_program, "claude:claude-opus-5-5",
      dict(title=paper["title"], abstract=paper["abstract"], introduction_first=paper["introduction_first"],
           conclusion=paper.get("conclusion") or "", original_paper=text, writing_brief=translator.WRITING_BRIEF)),
  "rewriteSection": (translator._section_program, "openai:gpt-6-luna",
      dict(section_name="methods", original_section=paper["methods"], section_guidance=translator.SECTION_GUIDANCE["methods"],
           rewritten_opening=opening_md, glossary=[translator.Term(**g) for g in glossary],
           reader_habits=translator.READER_HABITS, writing_brief=translator.WRITING_BRIEF)),
  "studentOpening": (student.OPENING_G, "openai:local",
      dict(paper=student_text, reference_glossary=glossary_text(pid, student_text), writer="opus")),
  "studentSection": (student.SECTION_G, "openai:local",
      dict(section_name="results", original_section=paper["results"], rewritten_opening=opening_md,
           reference_glossary=glossary_text(pid, paper["results"]), writer="opus")),
}
out = {}
for name, (fn, lm, inputs) in cases.items():
    try:
        r = fn.using(lm=lm).render(**inputs)
    except Exception as e:
        print(name, "render failed:", type(e).__name__, str(e)[:300]); continue
    print(name, type(r).__name__, [a for a in dir(r) if not a.startswith("_")][:25])
    out[name] = {"lm": lm, "inputs": {k: (v if not isinstance(v, list) else [g if isinstance(g, dict) else g.model_dump() for g in v]) for k, v in inputs.items()},
                 "request": r}
# the students' training prompt for the same inputs (sft layout, as prepare.py built it)
out["studentOpening"]["training_prompt"] = student.conversation(student.OPENING_G, cases["studentOpening"][2],
    {"title": "", "abstract": "", "introduction_first": "", "conclusion": ""})[0]
out["studentSection"]["training_prompt"] = student.conversation(student.SECTION_G, cases["studentSection"][2], {"result": ""})[0]
import pickle; pickle.dump(out, open("/tmp/fx/py_requests.pkl", "wb"))
