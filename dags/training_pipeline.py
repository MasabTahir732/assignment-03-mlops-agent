"""Airflow 3 TaskFlow DAG; all five stages are separately visible in the UI."""

import logging
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, get_current_context, task

from agent.tools.common import ROOT
from training import train as pipeline


@dag(
    dag_id="iris_training",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["assignment", "tiny-data"],
    default_args={"retries": 1, "retry_delay": timedelta(seconds=10)},
    doc_md="Validate → preprocess → train → evaluate/MLflow → quality gate. No auto-deployment.",
)
def iris_training():
    @task
    def validate_data():
        context = get_current_context()
        conf = context["dag_run"].conf or {}
        failure = conf.get("inject_failure", "none")
        if failure not in ("none", "validation", "quality"):
            raise ValueError("Unsupported failure injection")
        logging.info("Validating tiny dataset; failure mode=%s", failure)
        pipeline.validate(failure)
        # Stable per-run folder; retries reuse it, different runs cannot overwrite it.
        import hashlib

        key = hashlib.sha256(context["run_id"].encode()).hexdigest()[:16]
        return {"work_dir": str(ROOT / "reports" / key), "inject_failure": failure}

    @task
    def preprocess_data(info):
        logging.info("Deterministic stratified split")
        pipeline.preprocess(info["work_dir"])
        return info

    @task
    def train_model(info):
        logging.info("Training RandomForest: 50 trees, max_depth=3")
        pipeline.train(info["work_dir"])
        return info

    @task
    def evaluate_and_log(info):
        logging.info("Recording metrics, artifacts and DVC provenance in MLflow")
        pipeline.evaluate(info["work_dir"], inject_failure=info["inject_failure"])
        return info

    @task
    def quality_gate(info):
        logging.info("Quality gate: macro F1 >= 0.90")
        pipeline.quality_gate(info["work_dir"])

    quality_gate(evaluate_and_log(train_model(preprocess_data(validate_data()))))


iris_training()
