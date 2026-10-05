"""Opus (the teacher's recipe, translator.py) with the reference glossary as one more
input, on the first 3 benchmark test papers. No other change.

    .venv/bin/python glossary/opus_glossary.py              # rewrite (Opus subscription)
    .venv/bin/python training/eval_student.py judge opus-glossary --papers 3
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dpyr
from functai import ai

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rewrite_benchmark"))
sys.path.insert(0, str(ROOT / "training"))
import eval_v3                                                     # noqa: E402
import translator                                                  # noqa: E402
from eval_student import glossary_text                             # noqa: E402

NAME = "opus-glossary"
GLOSSARY_RULE = (" reference_glossary gives correct facts about hard terms, from reference works:"
                 " when you explain one of these terms, base the explanation on it, in your own plain"
                 " words, and add no facts the paper does not need.")


def build():
    """Re-create both programs from translator.py's source with the extra input and rule,
    so the instructions are the teacher's own word for word."""
    import ast
    tree = ast.parse(Path(translator.__file__).read_text())
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("rewrite_opening", "rewrite_section")]
    for d in defs:
        args = [a.arg for a in d.args.args]
        d.args.args.insert(args.index("writing_brief") if d.name == "rewrite_opening" else args.index("reader_habits"),
                           ast.arg(arg="reference_glossary", annotation=ast.Name(id="str", ctx=ast.Load())))
        doc = ast.get_docstring(d, clean=False)
        d.body[0] = ast.Expr(ast.Constant(doc.rstrip() + GLOSSARY_RULE))
    namespace = {"ai": ai, **{k: getattr(translator, k) for k in ("OpeningRewrite", "SectionRewrite", "Term")},
                 "list": list, "str": str}
    exec(compile(ast.fix_missing_locations(ast.Module(body=defs, type_ignores=[])), translator.__file__, "exec"),
         namespace)
    return (namespace["rewrite_opening"].using(**translator.SETTINGS),
            namespace["rewrite_section"].using(**translator.SETTINGS))


def main():
    opening_fn, section_fn = build()
    out = eval_v3.BASE / "rewrites" / NAME / "rewrites"
    out.mkdir(parents=True, exist_ok=True)

    def one_paper(paper):
        pid = paper["paper_id"]
        whole = translator.whole_paper_text(paper)
        opening = translator.call_with_one_retry(
            opening_fn, title=paper.get("title") or "", abstract=paper.get("abstract") or "",
            introduction_first=paper.get("introduction_first") or "", conclusion=paper.get("conclusion") or "",
            original_paper=whole, reference_glossary=glossary_text(pid, whole),
            writing_brief=translator.WRITING_BRIEF).result
        rewritten_opening = "\n\n".join(f"## {s}\n\n{getattr(opening, s)}" for s in translator.OPENING_SECTIONS
                                        if getattr(opening, s))
        rows = [{"paper_id": pid, "section": s, "original": paper[s], "rewrite": getattr(opening, s)}
                for s in translator.OPENING_SECTIONS if paper.get(s)]

        def one_section(s):
            return translator.call_with_one_retry(
                section_fn, section_name=s, original_section=paper[s],
                section_guidance=translator.SECTION_GUIDANCE[s], rewritten_opening=rewritten_opening,
                glossary=opening.glossary, reference_glossary=glossary_text(pid, paper[s]),
                reader_habits=translator.READER_HABITS, writing_brief=translator.WRITING_BRIEF).result.text

        todo = [s for s in translator.OTHER_SECTIONS if paper.get(s)]
        with ThreadPoolExecutor(len(todo)) as pool:
            rows += [{"paper_id": pid, "section": s, "original": paper[s], "rewrite": t}
                     for s, t in zip(todo, pool.map(one_section, todo))]
        dpyr.read(rows).write_parquet(out / translator.file_name(pid))
        print(f"{pid}: done", flush=True)

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(one_paper, eval_v3.test_papers()[:3]))


if __name__ == "__main__":
    main()
