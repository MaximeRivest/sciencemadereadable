"""Operations run only inside the per-document networkless sandbox."""

from pathlib import Path
import json
import os
import socket
import sys

from extract import render_pdf, save


def main():
    operation = sys.argv[1]
    out = Path("/output")
    if operation == "probe":
        refused = False
        try:
            socket.create_connection(("127.0.0.1", 8880), timeout=0.3).close()
        except OSError:
            refused = True
        read_only = True
        if Path("/job").exists():
            try:
                Path("/job/progress.json").write_text("must not be possible")
                read_only = False
            except OSError:
                pass
        links_blocked = True
        if Path("/job/progress.json").exists():
            try:
                os.link("/job/progress.json", "/output/alias")
                links_blocked = False
                Path("/output/alias").unlink()
            except OSError:
                pass
        save(
            out / "probe.json",
            {
                "network_refused": refused,
                "job_control_read_only": read_only,
                "cross_mount_links_blocked": links_blocked,
                "key_absent": "TYPESAFE_API_KEY" not in os.environ
                and "CREDENTIALS_DIRECTORY" not in os.environ,
                "home_secret_hidden": not Path(
                    "/home/maxime/.config/scholarsreadinglist/typesafe.env"
                ).exists(),
                "other_jobs_hidden": not Path("/var/lib/smr-editions/jobs").exists(),
            },
        )
        return
    if operation in {"render", "render-rgb", "thumbs"}:
        data = Path("/input.pdf").read_bytes()
        if len(data) > 25 * 1024 * 1024 or not data.startswith(b"%PDF-"):
            raise ValueError("Please upload a PDF under 25 MiB.")
        only = None
        if operation == "render-rgb" and len(sys.argv) > 2:
            # Controller-supplied page numbers, e.g. "3,4,5"; never parser input.
            if not all(x.isdigit() for x in sys.argv[2].split(",")):
                raise ValueError("Invalid page selection.")
            only = [int(x) for x in sys.argv[2].split(",")]
        pages = render_pdf(
            data,
            out,
            raw_pixels=operation == "render-rgb",
            only_pages=only,
            thumbs=operation == "thumbs",
        )
        save(out / "render.json", {"pages": pages})
        return
    if operation == "export":
        from editions import make_edition, make_epub, make_print, make_pdf

        # The job/control directory is read-only. Work in an isolated copy;
        # even a compromised PDF parser cannot mark the real job as completed.
        import shutil

        work = Path("/tmp/edition-input")
        work.mkdir()
        for name in (
            "original.pdf",
            "labelled.json",
            "sections.json",
            "paper.json",
            "rewrite.json",
            "extraction.json",
            "comparison.json",
            "caption-anchors.json",
        ):
            src = Path("/job") / name
            if src.exists():
                shutil.copyfile(src, work / name)
        edition = make_edition(work, out, regenerate_pdf=True, automatic=True)
        make_epub(edition, out)
        make_print(edition, out)
        make_pdf(edition, out, os.environ.get("CHROMIUM_BIN"))
        return
    raise ValueError("Unknown operation")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        save(Path("/output/failure.json"), {"error": str(e)[:400]})
        raise
