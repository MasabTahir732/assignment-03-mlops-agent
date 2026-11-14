import hashlib
import json
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from api.main import app
from training import train as t


@pytest.fixture
def small_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "ROOT", tmp_path)
    t.generate("v2")
    path = tmp_path / "data/iris.csv"
    pointer = {"outs": [{"md5": hashlib.md5(path.read_bytes()).hexdigest(), "path": "iris.csv"}]}
    (tmp_path / "data/iris.csv.dvc").write_text(yaml.safe_dump(pointer))
    monkeypatch.setattr(t, "command", lambda args: "")
    return tmp_path


def test_tiny_dataset_and_validation(small_repo):
    result = t.validate()
    assert result["rows"] == 150
    assert (small_repo / "data/iris.csv").stat().st_size < 5000


def test_changed_dataset_rejected(small_repo):
    path = small_repo / "data/iris.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "sepal_length"] = 6.3
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differs from DVC"):
        t.validate()


def test_validation_failure_stops_work(small_repo):
    with pytest.raises(ValueError, match="INJECTED_VALIDATION_FAILURE"):
        t.validate("validation")
    assert not (small_repo / "models").exists()


def test_dirty_git_provenance_rejected(small_repo, monkeypatch):
    monkeypatch.setattr(t, "command", lambda args: " M training/train.py")
    with pytest.raises(ValueError, match="Commit tracked changes"):
        t.validate()


def test_failed_gate_preserves_previous_model(small_repo):
    models = small_repo / "models"
    models.mkdir()
    (models / "model.joblib").write_bytes(b"previous approved model")
    work = small_repo / "work"
    work.mkdir()
    (work / "metadata.json").write_text(json.dumps({"quality_passed": False}))
    with pytest.raises(ValueError, match="QUALITY_GATE_FAILED"):
        t.quality_gate(str(work))
    assert (models / "model.joblib").read_bytes() == b"previous approved model"


def test_inference_unready_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.get("/metadata").status_code == 503


def test_real_training_bundle_and_prediction(small_repo, monkeypatch):
    work = str(small_repo / "work")
    t.preprocess(work)
    t.train(work)
    model = joblib.load(small_repo / "work/candidate.joblib")
    bundle = {
        "model": model,
        "metadata": {
            "quality_passed": True,
            "run_id": "real-test-run",
            "classes": ["setosa", "versicolor", "virginica"],
        },
    }
    path = small_repo / "bundle.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        result = client.post(
            "/predict",
            json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        )
        assert result.status_code == 200
        assert result.json() == {"class_id": 0, "class_name": "setosa", "run_id": "real-test-run"}
        assert client.post("/predict", json={"sepal_length": -1}).status_code == 422


def test_failed_quality_evaluation_is_logged_not_registered(small_repo, monkeypatch):
    work = str(small_repo / "work")
    t.preprocess(work)
    t.train(work)
    context = Mock()
    context.__enter__ = Mock(return_value=Mock(info=Mock(run_id="test-run")))
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(t.mlflow, "start_run", lambda: context)
    for name in (
        "set_tracking_uri",
        "set_experiment",
        "log_params",
        "set_tags",
        "log_metrics",
        "log_artifact",
    ):
        monkeypatch.setattr(t.mlflow, name, Mock())
    monkeypatch.setattr(t, "command", lambda args: "a" * 40)
    result = t.evaluate(work, inject_failure="quality")
    assert not result["quality_passed"]
    assert result["metrics"]["f1"] <= 1.0
    assert t.mlflow.set_tags.call_args.args[0]["quality_gate"] == "failed"
    assert t.mlflow.log_artifact.call_count == 3
