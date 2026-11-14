"""Subprocesses never receive a shell or a free-form command."""

import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("MLOPS_ROOT", Path(__file__).resolve().parents[2])).resolve()
TIMEOUT = 30


def command(args: list[str], timeout: int = TIMEOUT) -> str:
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        # Raw stderr can contain remote URLs and credentials. Keep it out of logs/responses.
        raise RuntimeError(f"{args[0]} {args[1]} failed (exit {result.returncode})")
    return result.stdout.strip()
