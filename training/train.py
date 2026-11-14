"""Real DVC provenance, deterministic splits, MLflow metrics and promotion gate."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from agent.tools.common import ROOT, command

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
MIN_F1 = 0.90


def generate(version: str):
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=FEATURES)
    frame["target"] = iris.target
    # v1=120 stratified rows, v2=all 150. Both fit in a few KB.
    if version == "v1":
        frame = frame.groupby("target", group_keys=False).head(40)
    path = ROOT / "data/iris.csv"
    path.parent.mkdir(exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    print(f"Generated {version}: {len(frame)} rows, {path.stat().st_size} bytes")


def validate(inject_failure: str = "none") -> dict:
    if inject_failure == "validation":
        raise ValueError("INJECTED_VALIDATION_FAILURE: required target column is missing")
    frame = pd.read_csv(ROOT / "data/iris.csv")
    if list(frame.columns) != FEATURES + ["target"]:
        raise ValueError("Unexpected dataset schema")
    if len(frame) < 90 or frame.isna().any().any():
        raise ValueError("Dataset needs >=90 complete rows")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite dataset values")
    if not set(frame.target.unique()).issubset({0, 1, 2}) or frame.target.nunique() != 3:
        raise ValueError("Expected three Iris classes")
    if (frame[FEATURES] <= 0).any().any():
        raise ValueError("Iris measurements must be positive")
    pointer = yaml.safe_load((ROOT / "data/iris.csv.dvc").read_text())["outs"][0]
    actual = hashlib.md5((ROOT / "data/iris.csv").read_bytes()).hexdigest()
    if actual != pointer["md5"]:
        raise ValueError("Dataset differs from DVC pointer; run dvc add and commit first")
    # Tracked changes invalidate a clean immutable provenance claim.
    dirty = command(["git", "status", "--porcelain", "--untracked-files=no"])
    if dirty:
        raise ValueError("Commit tracked changes before training for reproducible provenance")
    return {"rows": len(frame), "hash": actual}


def preprocess(work_dir: str):
    dest = Path(work_dir)
    dest.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "data/iris.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame.target,
        test_size=0.25,
        stratify=frame.target,
        random_state=42,
    )
    joblib.dump((x_train, x_test, y_train, y_test), dest / "split.joblib")


def train(work_dir: str, trees: int = 50, max_depth: int = 3):
    dest = Path(work_dir)
    x_train, _, y_train, _ = joblib.load(dest / "split.joblib")
    model = RandomForestClassifier(n_estimators=trees, max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)
    joblib.dump(model, dest / "candidate.joblib")


def evaluate(
    work_dir: str, trees: int = 50, max_depth: int = 3, inject_failure: str = "none"
) -> dict:
    dest = Path(work_dir)
    _, x_test, _, y_test = joblib.load(dest / "split.joblib")
    model = joblib.load(dest / "candidate.joblib")
    predictions = model.predict(x_test)
    metrics = {
        "f1": float(f1_score(y_test, predictions, average="macro")),
        "accuracy": float(accuracy_score(y_test, predictions)),
    }
    threshold = 1.01 if inject_failure == "quality" else MIN_F1
    passed = metrics["f1"] >= threshold
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("iris-demo")
    pointer = yaml.safe_load((ROOT / "data/iris.csv.dvc").read_text())["outs"][0]
    commit = command(["git", "rev-parse", "HEAD"])
    tags = command(["git", "tag", "--points-at", "HEAD"]).splitlines()
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "trees": trees,
                "max_depth": max_depth,
                "random_state": 42,
                "test_size": 0.25,
                "quality_threshold": threshold,
            }
        )
        mlflow.set_tags(
            {
                "dataset_hash": pointer["md5"],
                "dataset_git_commit": commit,
                "dataset_version": ",".join(tags) or commit[:7],
                "dataset_pointer": "data/iris.csv.dvc",
                "quality_gate": "passed" if passed else "failed",
            }
        )
        mlflow.log_metrics(metrics)
        metadata = {
            "run_id": run.info.run_id,
            "dataset_hash": pointer["md5"],
            "dataset_git_commit": commit,
            "metrics": metrics,
            "quality_passed": passed,
            "classes": ["setosa", "versicolor", "virginica"],
        }
        (dest / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (dest / "metrics.json").write_text(json.dumps(metrics, indent=2))
        mlflow.log_artifact(str(dest / "candidate.joblib"))
        mlflow.log_artifact(str(dest / "metadata.json"))
        mlflow.log_artifact(str(ROOT / "data/iris.csv.dvc"), "provenance")
        # Registry gets candidates only after they pass the gate.
        if passed:
            from mlflow import MlflowClient

            client = MlflowClient()
            if not client.search_registered_models(filter_string="name = 'iris-classifier'"):
                client.create_registered_model("iris-classifier")
            client.create_model_version(
                "iris-classifier", f"runs:/{run.info.run_id}/candidate.joblib", run.info.run_id
            )
    return metadata


def quality_gate(work_dir: str):
    dest = Path(work_dir)
    metadata = json.loads((dest / "metadata.json").read_text())
    if not metadata["quality_passed"]:
        raise ValueError("QUALITY_GATE_FAILED: candidate blocked; deployed model is unchanged")
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    # Inference loads one atomic bundle, avoiding mixed model/metadata during promotion.
    bundle = {"model": joblib.load(dest / "candidate.joblib"), "metadata": metadata}
    temp = model_dir / "model.joblib.tmp"
    joblib.dump(bundle, temp)
    temp.replace(model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", choices=["v1", "v2"])
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--inject-failure", choices=["none", "validation", "quality"], default="none"
    )
    args = parser.parse_args()
    if args.generate:
        generate(args.generate)
        return
    from uuid import uuid4

    work = str(ROOT / "reports" / uuid4().hex)
    validate(args.inject_failure)
    preprocess(work)
    train(work, args.trees, args.max_depth)
    evaluate(work, args.trees, args.max_depth, args.inject_failure)
    quality_gate(work)


if __name__ == "__main__":
    main()
