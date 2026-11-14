"""All experiment claims come from the supported MLflow client."""

import math
import os

from mlflow import MlflowClient


def client():
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    return MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def experiments() -> dict:
    return {
        "experiments": [
            {"id": e.experiment_id, "name": e.name}
            for e in client().search_experiments(max_results=50)
        ]
    }


def _run(run) -> dict:
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }


def runs(limit: int = 5) -> dict:
    c = client()
    exp = c.get_experiment_by_name("iris-demo")
    if not exp:
        return {"runs": [], "note": "iris-demo has no recorded experiments yet"}
    return {
        "runs": [
            _run(r)
            for r in c.search_runs(
                [exp.experiment_id],
                max_results=limit,
                order_by=["attributes.start_time DESC"],
            )
        ]
    }


def details(run_id: str) -> dict:
    c = client()
    result = _run(c.get_run(run_id))
    result["artifacts"] = [{"path": a.path, "is_dir": a.is_dir} for a in c.list_artifacts(run_id)]
    return result


def best(metric: str = "f1", limit: int = 5, mode: str = "max") -> dict:
    recent = runs(limit)["runs"]
    eligible = [
        r
        for r in recent
        if r["status"] == "FINISHED"
        and metric in r["metrics"]
        and math.isfinite(r["metrics"][metric])
    ]
    selected = (max if mode == "max" else min)(
        eligible,
        key=lambda r: r["metrics"][metric],
        default=None,
    )
    return {
        "metric": metric,
        "mode": mode,
        "candidate": selected,
        "compared": recent,
        "reason": f"{mode} recorded {metric} among the latest {limit} runs; "
        "candidate selection alone does not establish production readiness",
    }


def compare(run_ids: list[str]) -> dict:
    from agent.schemas import RunInput

    return {"runs": [details(RunInput(run_id=r).run_id) for r in run_ids]}


def registry() -> dict:
    return {
        "versions": [
            {"name": v.name, "version": v.version, "run_id": v.run_id, "status": v.status}
            for v in client().search_model_versions("name = 'iris-classifier'")
        ]
    }
