"""Read real DVC pointers and Git history; no arbitrary subprocess tool."""

import json
import shlex
import sys

import yaml

from agent.tools.common import ROOT, command


def tracked() -> dict:
    paths = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in ROOT.glob("data/*.dvc"))
    return {"datasets": paths}


def current() -> dict:
    pointer = yaml.safe_load((ROOT / "data/iris.csv.dvc").read_text())
    data_commit = command(["git", "log", "-1", "--format=%H", "--", "data/iris.csv.dvc"])
    return {
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command(["git", "status", "--porcelain"])),
        "dataset": "data/iris.csv",
        "dataset_git_commit": data_commit,
        "dataset_version": command(["git", "describe", "--tags", "--always", data_commit]),
        "dvc_hash": pointer["outs"][0]["md5"],
        "pointer": "data/iris.csv.dvc",
        "data_status": status(),
    }


def status() -> dict:
    return json.loads(
        command(
            [
                sys.executable,
                "-m",
                "dvc",
                "status",
                "data/iris.csv.dvc",
                "--json",
            ]
        )
    )


def compare(old: str, new: str = "HEAD") -> dict:
    # Resolve revisions to immutable commits before passing them to DVC.
    refs = [command(["git", "rev-parse", "--verify", f"{r}^{{commit}}"]) for r in (old, new)]
    difference = command([sys.executable, "-m", "dvc", "diff", *refs, "--json"])
    pointers = {}
    for label, ref in zip((old, new), refs):
        pointers[label] = yaml.safe_load(
            command(
                [
                    "git",
                    "show",
                    f"{ref}:data/iris.csv.dvc",
                ]
            )
        )["outs"][0]
    return {"old": refs[0], "new": refs[1], "diff": json.loads(difference), "pointers": pointers}


def reproduce(revision: str) -> dict:
    commit = command(["git", "rev-parse", "--verify", f"{revision}^{{commit}}"])
    return {
        "commit": commit,
        "instructions": [
            f"git worktree add ../iris-reproduce-{commit[:7]} {shlex.quote(commit)}",
            f"cd ../iris-reproduce-{commit[:7]}",
            "python -m dvc pull data/iris.csv.dvc",
        ],
        "note": "Configure the original absolute local DVC remote in the new worktree first. "
        "A separate worktree preserves the current dataset.",
    }
