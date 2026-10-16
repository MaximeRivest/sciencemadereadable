"""Networkless per-job parser/export sandbox; no keys, home directory or other jobs."""

from __future__ import annotations
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = ROOT / "reading_engine"


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (240, 250))
    resource.setrlimit(resource.RLIMIT_FSIZE, (300 * 1024 * 1024, 300 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))


def run(operation, source, output, timeout=300):
    bwrap = os.environ.get("SMR_BWRAP") or shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError(
            "The document sandbox is unavailable; refusing to parse this PDF."
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binds = [Path("/nix/store"), HERE, ENGINE]
    for p in ("/run/current-system/sw", "/etc/fonts", "/lib", "/lib64"):
        if Path(p).exists():
            binds.append(Path(p))
    cmd = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/run/user",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "LANG",
        "C.UTF-8",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONPATH",
        str(ENGINE) + ":" + str(HERE),
        "--setenv",
        "PATH",
        "/run/current-system/sw/bin",
        "--setenv",
        "LD_LIBRARY_PATH",
        os.environ.get("LD_LIBRARY_PATH", "/run/current-system/sw/share/nix-ld/lib"),
    ]
    for p in binds:
        cmd += ["--ro-bind", str(p), str(p)]
    # This environment is an explicit allowlist, not an inherited API-key environment.
    for k in ("NIX_LD", "NIX_LD_LIBRARY_PATH", "LD_LIBRARY_PATH", "FONTCONFIG_FILE"):
        if os.environ.get(k):
            cmd += ["--setenv", k, os.environ[k]]
    chromium = os.environ.get("CHROMIUM_BIN")
    if chromium:
        cmd += ["--setenv", "CHROMIUM_BIN", chromium]
    if operation in {"render", "render-rgb"}:
        cmd += [
            "--ro-bind",
            str(Path(source).resolve()),
            "/input.pdf",
            "--bind",
            str(output),
            "/output",
        ]
    elif operation == "export":
        cmd += [
            "--ro-bind",
            str(Path(source).resolve()),
            "/job",
            "--bind",
            str(output),
            "/output",
        ]
    elif operation == "probe":
        cmd += ["--bind", str(output), "/output"]
        if source:
            cmd += ["--ro-bind", str(Path(source).resolve()), "/job"]
    else:
        raise ValueError("Unknown sandbox operation")
    # Resolve the release-directory symlink, but retain the venv's executable
    # entry point (resolving the final symlink would lose its site-packages).
    python = str(Path(sys.executable).parent.resolve() / Path(sys.executable).name)
    cmd += ["--chdir", "/tmp", python, str(HERE / "pdf_ops.py"), operation]
    # Never capture unbounded parser output in the controller's memory, or
    # write a diagnostic through a path supplied by the sandbox (symlink risk).
    with tempfile.TemporaryFile() as diagnostic:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=diagnostic,
            timeout=timeout,
            env={
                k: v
                for k, v in os.environ.items()
                if k in {"PATH", "NIX_LD", "NIX_LD_LIBRARY_PATH", "LD_LIBRARY_PATH"}
            },
            preexec_fn=limits,
            close_fds=True,
        )
        diagnostic.seek(max(0, diagnostic.tell() - 8000))
        detail = diagnostic.read(8000).decode("utf-8", "replace")
    count = total = 0
    for p in output.rglob("*"):
        count += 1
        if p.is_symlink() or (not p.is_file() and not p.is_dir()):
            raise RuntimeError("Unexpected output from the document sandbox.")
        if p.is_file():
            info = p.stat()
            if info.st_nlink != 1:
                raise RuntimeError("Hard-linked sandbox output is not allowed.")
            total += info.st_size
        if (
            count > 1000
            or total > (400 if operation == "render-rgb" else 300) * 1024 * 1024
        ):
            raise ValueError(
                "This document produced too much export data. Try a shorter paper."
            )
    if result.returncode:
        message = output / "failure.json"
        if message.is_file() and message.stat().st_size <= 4096:
            try:
                data = json.loads(message.read_text())
                if isinstance(data.get("error"), str):
                    raise ValueError(data["error"][:400])
            except (json.JSONDecodeError, AttributeError):
                pass
        print("Isolated process failed:", detail[-2000:], file=sys.stderr)
        raise RuntimeError(
            "The isolated document process could not finish. Try a smaller PDF."
        )
    return result


def render(data, run_dir):
    staging = Path(run_dir) / "render-sandbox"
    run("render", Path(run_dir) / "source.pdf", staging)
    result = json.loads((staging / "render.json").read_text())
    if not 1 <= result["pages"] <= 20:
        raise ValueError("The PDF page count exceeds the limit.")
    for name in ("original.pdf", "rendered.pdf", "native-diagnostic.json"):
        shutil.copyfile(staging / name, Path(run_dir) / name)
    shutil.copytree(staging / "pages", Path(run_dir) / "pages", dirs_exist_ok=True)
    return result["pages"]
