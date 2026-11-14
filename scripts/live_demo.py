"""Run real integration checks. No service responses are mocked in this script."""

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from agent.orchestrator import ask
from agent.tools import airflow_tools as af
from agent.tools.common import ROOT
from scripts.capture_evidence import SCENARIOS
from training.train import generate

EVIDENCE = ROOT / "reports/live-evidence"


def run(args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args[:3]} (exit {result.returncode})")
    return result.stdout.strip()


def save(name, value):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / name).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def prepare():
    if not (ROOT / ".env").exists():
        text = (ROOT / ".env.example").read_text()
        for key in ("AIRFLOW_DB_PASSWORD", "AIRFLOW_SECRET_KEY", "AIRFLOW_JWT_SECRET"):
            text += f"\n{key}={secrets.token_hex(24)}"
        (ROOT / ".env").write_text(text + "\n")
    for folder in ("reports", "models"):
        path = ROOT / folder
        path.mkdir(exist_ok=True)
        path.chmod(0o777)  # Local bind-mount shared with Airflow's non-root UID 50000.
    os.environ["DVC_SITE_CACHE_DIR"] = str(ROOT / ".dvc/tmp/site-cache")
    run(
        [
            sys.executable,
            "-m",
            "dvc",
            "remote",
            "add",
            "--local",
            "--force",
            "-d",
            "local-store",
            str(ROOT / ".dvc-remote"),
        ]
    )
    original = (ROOT / "data/iris.csv.dvc").read_text()
    for version in ("v1", "v2"):
        generate(version)
        run([sys.executable, "-m", "dvc", "add", "data/iris.csv"])
        run([sys.executable, "-m", "dvc", "push", "data/iris.csv.dvc"])
    if (ROOT / "data/iris.csv.dvc").read_text() != original:
        raise RuntimeError("Regenerated v2 does not match its committed DVC pointer")
    print("Prepared both tiny dataset versions and local demo directories")


def authenticate():
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            passwords = json.loads(
                run(
                    [
                        "docker",
                        "compose",
                        "exec",
                        "-T",
                        "airflow",
                        "cat",
                        "/opt/airflow/simple_auth_manager_passwords.json.generated",
                    ]
                )
            )
            os.environ["AIRFLOW_USERNAME"] = "admin"
            os.environ["AIRFLOW_PASSWORD"] = passwords["admin"]
            af.dags()
            return
        except (RuntimeError, ValueError, KeyError, requests.RequestException):
            time.sleep(3)
    raise TimeoutError("Airflow authentication or startup did not become ready")


def wait_run(run_id, expected):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        runs = af.runs().get("dag_runs", [])
        found = next((r for r in runs if r["dag_run_id"] == run_id), None)
        if found and found["state"] in ("success", "failed"):
            if found["state"] != expected:
                save("unexpected-failure.json", af.diagnose(run_id))
                raise RuntimeError(f"DAG {run_id} ended {found['state']}; expected {expected}")
            return found
        time.sleep(3)
    raise TimeoutError(f"DAG did not finish within ten minutes: {run_id}")


def pipeline():
    authenticate()
    af._request("PATCH", "/dags/iris_training", {"is_paused": False})
    response = ask("Trigger the model training pipeline", confirmed=True)
    trigger = next((e.data for e in response.evidence if e.status == "ok"), None)
    if not trigger:
        raise RuntimeError("Real training trigger failed")
    run_id = trigger["dag_run_id"]
    completed = wait_run(run_id, "success")
    tasks = af.tasks(run_id)
    save(
        "pipeline.json",
        {
            "user": "Trigger the model training pipeline",
            "submission": response.model_dump(),
            "completed_run": completed,
            "tasks": tasks,
        },
    )
    if not all(t["state"] == "success" for t in tasks.get("task_instances", [])):
        raise RuntimeError("Not all training tasks succeeded")
    print(f"Actual Airflow run succeeded: {run_id}")


def failures():
    authenticate()
    investigations = []
    for failure, task_id in (("validation", "validate_data"), ("quality", "quality_gate")):
        previous = hashlib.sha256((ROOT / "models/model.joblib").read_bytes()).hexdigest()
        trigger = af.trigger(confirmed=True, inject_failure=failure)
        completed = wait_run(trigger["dag_run_id"], "failed")
        diagnosis = af.diagnose(trigger["dag_run_id"])
        failed = [t for t in diagnosis["failed_tasks"] if t["state"] == "failed"]
        if not any(t["task_id"] == task_id for t in failed):
            raise RuntimeError(f"Unexpected failed task for {failure}")
        unchanged = (
            previous == hashlib.sha256((ROOT / "models/model.joblib").read_bytes()).hexdigest()
        )
        if not unchanged:
            raise RuntimeError("Failed workflow changed the approved model")
        investigations.append(
            {
                "injection": failure,
                "run": completed,
                "diagnosis": diagnosis,
                "approved_model_unchanged": unchanged,
                "root_cause": "Injected missing-target validation error"
                if failure == "validation"
                else "Injected threshold 1.01 exceeds possible F1",
                "resolution": "Remove failure injection and rerun with the original validation/gate",
            }
        )
    recovered = af.trigger(confirmed=True)
    recovery = wait_run(recovered["dag_run_id"], "success")
    save("failure-investigations.json", {"investigations": investigations, "recovery": recovery})
    print("Both injected failures diagnosed; recovery DAG succeeded")


def capture(mode):
    authenticate()
    conversations = []
    for name, prompt in SCENARIOS:
        response = ask(prompt, mode)
        conversations.append({"scenario": name, "user": prompt, "response": response.model_dump()})
        save(
            "conversations-llm.json" if mode == "llm" else "conversations.json",
            {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "mode": mode,
                "conversations": conversations,
            },
        )
        if not response.evidence or any(e.status != "ok" for e in response.evidence):
            raise RuntimeError(f"Actual service/tool failure in scenario {name}")
    integration = conversations[-1]["response"]
    tools = {e["tool"] for e in integration["evidence"]}
    if not {"mlflow_best", "dvc_current", "kube_diagnose", "airflow_runs"}.issubset(tools):
        raise RuntimeError("Integrated response did not inspect all four services")
    save(
        "kubernetes.json",
        json.loads(
            run(["kubectl", "-n", "mlops-demo", "get", "pods,deployments,services", "-o", "json"])
        ),
    )
    save("dvc-comparison.json", ask("Compare dataset v1 and v2").model_dump())
    save("inference.json", requests.get("http://localhost:8001/metadata", timeout=15).json())
    print(f"Captured all five conversations with actual tool evidence ({mode})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "pipeline", "failures", "capture"])
    parser.add_argument("--confirm", action="store_true", help="Authorize demo workflow triggers")
    parser.add_argument("--mode", choices=["rules", "llm"], default="rules")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.stage in ("pipeline", "failures") and not args.confirm:
        parser.error("Workflow writes require --confirm")
    if args.stage == "capture":
        capture(args.mode)
    else:
        {"prepare": prepare, "pipeline": pipeline, "failures": failures}[args.stage]()


if __name__ == "__main__":
    main()
