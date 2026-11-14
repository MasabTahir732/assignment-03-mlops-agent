"""Subprocesses never receive a shell or a free-form command."""

import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("MLOPS_ROOT", Path(__file__).resolve().parents[2])).resolve()
TIMEOUT = 30


def command(args: list[str], timeout: int = TIMEOUT) -> str:
    env = os.environ.copy()
    if args[1:3] == ["-m", "dvc"]:
        env.setdefault("DVC_SITE_CACHE_DIR", str(ROOT / ".dvc/tmp/site-cache"))
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if result.returncode:
        # Raw stderr can contain remote URLs and credentials. Keep it out of logs/responses.
        raise RuntimeError(f"{args[0]} {args[1]} failed (exit {result.returncode})")
    return result.stdout.strip()
