"""Airflow 3 public API, authenticated with a JWT (or SimpleAuth credentials)."""

import os
from datetime import datetime, timezone
from urllib.parse import quote
from uuid import uuid4

import requests


def _request(method: str, path: str, payload=None):
    base = os.getenv("AIRFLOW_URL", "http://localhost:8080").rstrip("/")
    token = os.getenv("AIRFLOW_TOKEN")
    if not token:
        username, password = os.getenv("AIRFLOW_USERNAME"), os.getenv("AIRFLOW_PASSWORD")
        if not username or not password:
            raise RuntimeError("Set AIRFLOW_TOKEN or AIRFLOW_USERNAME/AIRFLOW_PASSWORD")
        auth = requests.post(
            f"{base}/auth/token", json={"username": username, "password": password}, timeout=15
        )
        auth.raise_for_status()
        token = auth.json()["access_token"]
    response = requests.request(
        method,
        f"{base}/api/v2{path}",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def dags() -> dict:
    return _request("GET", "/dags?limit=20")


def runs(dag_id: str = "iris_training") -> dict:
    return _request("GET", f"/dags/{dag_id}/dagRuns?limit=5&order_by=-logical_date")


def trigger(
    dag_id: str = "iris_training", confirmed: bool = False, inject_failure: str = "none"
) -> dict:
    if dag_id != "iris_training" or not confirmed:
        raise PermissionError("Triggering iris_training requires explicit --confirm")
    return _request(
        "POST",
        f"/dags/{dag_id}/dagRuns",
        {
            "dag_run_id": f"agent__{uuid4().hex}",
            "logical_date": datetime.now(timezone.utc).isoformat(),
            "conf": {"inject_failure": inject_failure},
        },
    )


def tasks(dag_run_id: str, dag_id: str = "iris_training") -> dict:
    return _request("GET", f"/dags/{dag_id}/dagRuns/{quote(dag_run_id, safe='')}/taskInstances")


def diagnose(dag_run_id: str, dag_id: str = "iris_training") -> dict:
    result = tasks(dag_run_id, dag_id)
    failed = [
        t
        for t in result.get("task_instances", [])
        if t.get("state") in ("failed", "upstream_failed")
    ]
    diagnostics = []
    for task in failed:
        entry = {
            "task_id": task["task_id"],
            "state": task["state"],
            "try_number": task.get("try_number"),
            "start_date": task.get("start_date"),
            "end_date": task.get("end_date"),
        }
        if task["state"] == "failed" and task.get("try_number", 0) > 0:
            path = (
                f"/dags/{dag_id}/dagRuns/{quote(dag_run_id, safe='')}/taskInstances/"
                f"{quote(task['task_id'], safe='')}/logs/{task['try_number']}"
            )
            try:
                entry["logs"] = _request("GET", path)
            except requests.RequestException:
                entry["logs_unavailable"] = True
        diagnostics.append(entry)
    return {
        "failed_tasks": diagnostics,
        "tasks": result,
        "note": "Use the task logs for root cause; upstream_failed is a consequence.",
    }
