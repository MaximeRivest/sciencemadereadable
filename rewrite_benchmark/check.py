"""Run the notebook's offline tests, including simulated calls in a temporary run.

Run from the project root: .venv/bin/python rewrite_benchmark/check.py
No network inference. Simulated outputs and scores never enter the real run folder.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/scientific-rewriting.md"


def main():
    cells = re.findall(r"^```python\n(.*?)^```", NOTEBOOK.read_text(), re.M | re.S)
    shell = InteractiveShell.instance()
    with tempfile.TemporaryDirectory(prefix=".paper-rewrite-check-", dir=ROOT) as temporary:
        run = Path(temporary)
        first = cells[0].replace('RUN = LAB / "runs" / EXPERIMENT / f"replicate-{REPLICATE}"', f"RUN = Path({str(run)!r})")
        for index, cell in enumerate([first, *cells[1:]], 1):
            with capture_output():
                result = shell.run_cell(cell, store_history=True)
            if not result.success:
                raise RuntimeError(f"Notebook cell {index} failed") from (result.error_before_exec or result.error_in_exec)
        ns = shell.user_ns
        assert not list((run / "calls").glob("*.json")), "Safe defaults made a call"
        pairs = ns["reference_pairs"]
        assert len(pairs) == 8
        messages = ns["reference_messages"]
        assert len(messages) == 5
        assert all(set(m) <= {"stage", "entry_id", "session_id", "timestamp", "provider", "model", "response_model", "reasoning_level", "text", "text_sha256"} for m in messages)

        # Render the real prompt and output schema without sending it.
        common = dict(original_paper=ns["SOURCE_CONTEXT"], source_evidence=ns["SOURCE_EVIDENCE"], writing_brief=ns["WRITING_BRIEF"])
        opening_inputs = {k: ns["paper"]["sections"][k] for k in ("title", "abstract", "introduction_first", "conclusion")}
        request = ns["rewrite_opening"].using(lm=ns["MODELS"]["astra"]).render(**opening_inputs, **common)
        assert request.model == ns["MODELS"]["astra"]
        assert "conclusion" in str(request.system)

        reference = {r["section_id"]: r["rewrite_text"] for r in pairs}
        observed = []

        def fake_predict(fn, **inputs):
            name = fn.__name__
            request = fn.render(**inputs)  # Real typed prompt construction, no send.
            assert request.model in ns["MODELS"].values()
            observed.append((name, inputs))
            if name == "rewrite_opening":
                result = ns["OpeningRewrite"](**{k: reference[k] for k in opening_inputs}, editorial_notes=[], glossary=[])
            elif name.startswith("rewrite_"):
                section = {"rewrite_introduction": "introduction_rest", "rewrite_methods": "methods", "rewrite_results": "results", "rewrite_discussion": "discussion"}[name]
                assert "opening" in inputs["previous_rewrites"]
                if section == "methods":
                    assert "introduction_rest" in inputs["previous_rewrites"]
                result = ns["SectionRewrite"](text=reference[section], editorial_notes=[], glossary_updates=[])
            else:
                result = ns["Assessment"](score=3, summary="Offline test fixture, not a model judgment.", issues=[], strengths=[])
            return SimpleNamespace(result=result, usage={"input_tokens": 10, "output_tokens": 20}, call_id="offline-fixture", responses=[])

        function_class = type(ns["rewrite_opening"])
        with patch.object(function_class, "predict", fake_predict):
            for label, model in ns["MODELS"].items():
                ns["candidates"][label] = ns["run_writer"](label, model, enabled=True)
            assert len(observed) == 10
            # Identical reruns load exact completed calls, not a new prediction.
            ns["run_writer"]("astra", ns["MODELS"]["astra"], enabled=True)
            assert len(observed) == 10
            ns["RUN_JUDGES"] = True
            ns["RUBRIC_APPROVED"] = True
            for cell in cells:
                if "tasks = []" in cell or "valid = scores[" in cell or "human_rows = []" in cell:
                    with capture_output():
                        result = shell.run_cell(cell, store_history=True)
                    if not result.success:
                        raise RuntimeError("Offline scoring/analysis failed") from result.error_in_exec
            # Fixture-identical writer texts correctly share cached evaluator
            # calls. There are still 156 comparison rows, not 156 paid requests.
            assert len(ns["assessments"]) == 156
            assert set(ns["scores"]["status"]) == {"ok"}
            assert all("candidate_id" not in inputs and "writer_model" not in inputs for name, inputs in observed if name.startswith("evaluate_"))
            assert all("reading_context" in inputs for name, inputs in observed if name in {"evaluate_faithfulness", "evaluate_completeness", "evaluate_accessibility"})
        invalid = ns["Assessment"](score=1, summary="Bad quote fixture", issues=[ns["Issue"](severity="major", explanation="Fixture", source_quote="a quote that does not occur", rewrite_quote="")], strengths=[])
        assert ns["quote_problems"](invalid, "text", "source")
        assert len(json.loads((run / "assessments.json").read_text())) == 156
        import pandas as pd
        sheet = pd.read_csv(run / "human_review.csv")
        assert set(sheet["candidate"]) == {"conversation_reference", "astra", "luna"}
    print(f"PASS: {len(cells)} safe notebook cells; source/export hashes; real prompt rendering; five-step handoffs; caching; 156 offline evaluation rows; blinded evaluator inputs; human-sheet expansion; invalid-quote detection.")


if __name__ == "__main__":
    main()
