import json
import subprocess
from unittest.mock import Mock

import pytest
import requests

from agent import orchestrator as o
from agent import schemas as s
from agent.tools import airflow_tools as af
from agent.tools import kubernetes_tools as kube
from agent.tools import mlflow_tools as mf


def register(monkeypatch, name, function, schema=s.EmptyInput):
    monkeypatch.setitem(o.TOOLS, name, (function, schema, "test tool"))


def candidate():
    return {
        "run_id": "a" * 32,
        "metrics": {"f1": 0.97},
        "params": {},
        "tags": {
            "dataset_hash": "abc",
            "dataset_git_commit": "b" * 40,
            "dataset_version": "v2",
            "quality_gate": "passed",
        },
    }


def test_successful_selection_and_real_execution(monkeypatch):
    called = Mock(return_value={"candidate": None, "compared": []})
    register(monkeypatch, "mlflow_best", called, s.BestInput)
    response = o.ask("Find the best model by f1")
    called.assert_called_once_with(metric="f1", limit=5, mode="max")
    assert response.evidence[0].status == "ok"
    assert "No finished" in " ".join(response.uncertainties)


def test_integration_correlates_four_systems_without_training(monkeypatch):
    register(monkeypatch, "mlflow_best", lambda **kw: {"candidate": candidate()}, s.BestInput)
    register(monkeypatch, "mlflow_details", lambda **kw: candidate(), s.RunInput)
    register(monkeypatch, "dvc_current", lambda: {"dvc_hash": "abc"})
    register(
        monkeypatch, "dvc_reproduce", lambda **kw: {"commit": kw["revision"]}, s.ReproduceInput
    )
    register(monkeypatch, "kube_diagnose", lambda: {"healthy": True, "deployments": []})
    register(
        monkeypatch,
        "airflow_runs",
        lambda **kw: {"dag_runs": [{"dag_run_id": "real-run", "state": "success"}]},
        s.DagInput,
    )
    mutation = Mock(side_effect=AssertionError("Must not trigger"))
    register(monkeypatch, "airflow_trigger", mutation, s.TriggerInput)
    response = o.ask("Find the best recent model, dataset version, deployment and retraining need")
    assert {"mlflow_best", "dvc_current", "kube_diagnose", "airflow_runs"}.issubset(
        {r.tool for r in response.evidence}
    )
    assert any("matches current pointer: True" in fact for fact in response.facts)
    assert any("No retraining requirement" in fact for fact in response.recommendations)
    mutation.assert_not_called()


def test_unavailable_tool_does_not_invent_health(monkeypatch):
    register(monkeypatch, "kube_diagnose", Mock(side_effect=OSError("missing kubectl")))
    response = o.ask("Why is my prediction API unavailable?")
    assert not response.facts
    assert "no operational result is known" in response.uncertainties[0]
    assert response.evidence[0].data is None


@pytest.mark.parametrize("failure", [requests.Timeout(), subprocess.TimeoutExpired("dvc", 30)])
def test_tool_timeout(monkeypatch, failure):
    register(monkeypatch, "dvc_status", Mock(side_effect=failure))
    result = o.execute("dvc_status", {}, "request-1")
    assert result.status == "error"
    assert "timed out" in result.error


def test_service_failure_redacts_exception(monkeypatch, caplog):
    register(monkeypatch, "dvc_status", Mock(side_effect=RuntimeError("secret-token-123")))
    with caplog.at_level("INFO", logger="mlops.tools"):
        result = o.execute("dvc_status", {}, "r123")
    assert "secret-token-123" not in result.model_dump_json() + caplog.text
    event = json.loads(caplog.records[-1].message)
    assert event.keys() == {"tool", "status", "duration_ms", "request_id"}
    assert event["request_id"] == "r123" and event["status"] == "error"


def test_invalid_request_and_extra_arguments_rejected():
    assert o.ask("").uncertainties
    assert o.ask("do something mysterious").uncertainties
    assert o.execute("dvc_current", {"shell": "delete"}, "r").status == "error"
    assert o.execute("unknown", {}, "r").status == "error"


def test_unauthorized_kubernetes_mutations_rejected(monkeypatch):
    command = Mock()
    monkeypatch.setattr(kube, "command", command)
    monkeypatch.setenv("KUBE_NAMESPACE", "mlops-demo")
    with pytest.raises(PermissionError):
        kube.restart(confirmed=False)
    with pytest.raises(PermissionError):
        kube.restart(namespace="kube-system", confirmed=True)
    command.assert_not_called()


def test_model_cannot_self_confirm(monkeypatch):
    command = Mock()
    monkeypatch.setattr(kube, "command", command)
    result = o.execute("kube_restart", {"confirmed": True}, "r", confirmed=False)
    assert result.status == "error"
    command.assert_not_called()


def test_confirmed_restart_scoped_to_namespace(monkeypatch):
    command = Mock(return_value="restarted")
    monkeypatch.setattr(kube, "command", command)
    monkeypatch.setenv("KUBE_NAMESPACE", "mlops-demo")
    result = o.execute("kube_restart", {}, "r", confirmed=True)
    assert result.status == "ok"
    args = command.call_args.args[0]
    assert args[args.index("-n") + 1] == "mlops-demo"
    assert args[-1] == "deployment/iris-api"


def test_airflow_confirmation_and_authenticated_v3_api(monkeypatch):
    request = Mock()
    response = request.return_value
    response.json.return_value = {"dag_run_id": "queued-run", "state": "queued"}
    monkeypatch.setattr(af.requests, "request", request)
    monkeypatch.setenv("AIRFLOW_TOKEN", "fake-test-token")
    with pytest.raises(PermissionError):
        af.trigger()
    request.assert_not_called()
    result = af.trigger(confirmed=True)
    assert result["state"] == "queued"
    assert "/api/v2/dags/iris_training/dagRuns" in request.call_args.args[1]
    assert request.call_args.kwargs["headers"]["Authorization"] == "Bearer fake-test-token"


def test_best_run_excludes_failed_and_missing_metrics(monkeypatch):
    monkeypatch.setattr(
        mf,
        "runs",
        lambda limit: {
            "runs": [
                {"run_id": "1", "status": "FAILED", "metrics": {"f1": 1.0}},
                {"run_id": "2", "status": "FINISHED", "metrics": {}},
                {"run_id": "3", "status": "FINISHED", "metrics": {"f1": 0.8}},
                {"run_id": "4", "status": "FINISHED", "metrics": {"f1": 0.9}},
            ]
        },
    )
    assert mf.best()["candidate"]["run_id"] == "4"
    assert mf.best(mode="min")["candidate"]["run_id"] == "3"


def test_failed_gate_recommends_investigation_not_automatic_trigger():
    run = candidate()
    run["tags"]["quality_gate"] = "failed"
    evidence = [
        s.ToolResult(
            tool="mlflow_best", status="ok", data={"candidate": run}, request_id="r", duration_ms=1
        )
    ]
    _, _, recommendations = o.summarize(evidence)
    assert any("Retraining is recommended" in line for line in recommendations)


def test_ollama_structured_tool_selection(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "message": {"tool_calls": [{"function": {"name": "dvc_tracked", "arguments": {}}}]}
    }
    monkeypatch.setattr(o.requests, "post", Mock(return_value=response))
    register(monkeypatch, "dvc_tracked", lambda: {"datasets": ["data/iris.csv.dvc"]})
    result = o.ask("List datasets", mode="llm")
    assert result.evidence[0].data["datasets"] == ["data/iris.csv.dvc"]


def test_ollama_cannot_turn_advice_into_mutation(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "message": {
            "tool_calls": [
                {"function": {"name": "airflow_trigger", "arguments": {"confirmed": True}}}
            ]
        }
    }
    monkeypatch.setattr(o.requests, "post", Mock(return_value=response))
    mutation = Mock()
    register(monkeypatch, "airflow_trigger", mutation, s.TriggerInput)
    result = o.ask("Should we retrain?", mode="llm", confirmed=True)
    assert not result.evidence and result.uncertainties
    mutation.assert_not_called()


def test_diagnosis_uses_pod_states_events_and_logs(monkeypatch):
    monkeypatch.setattr(
        kube,
        "pods",
        lambda: {
            "items": [
                {
                    "metadata": {"name": "iris-api-1"},
                    "status": {
                        "phase": "Running",
                        "containerStatuses": [
                            {
                                "ready": False,
                                "restartCount": 3,
                                "state": {"waiting": {"reason": "CrashLoopBackOff"}},
                            }
                        ],
                    },
                }
            ]
        },
    )
    monkeypatch.setattr(
        kube,
        "deployments",
        lambda: {
            "items": [
                {
                    "metadata": {"name": "iris-api"},
                    "spec": {
                        "replicas": 2,
                        "template": {"spec": {"containers": [{"image": "mlops-iris:1.0.0"}]}},
                    },
                    "status": {"availableReplicas": 0},
                }
            ]
        },
    )
    monkeypatch.setattr(kube, "services", lambda: {"items": [{"metadata": {"name": "iris-api"}}]})
    monkeypatch.setattr(kube, "logs", lambda pod: {"pod": pod, "logs": "model missing"})
    monkeypatch.setattr(
        kube,
        "command",
        lambda args: json.dumps(
            {"items": [{"involvedObject": {"name": "iris-api-1"}, "reason": "Unhealthy"}]}
        ),
    )
    result = kube.diagnose()
    assert not result["healthy"]
    assert result["pods"][0]["restarts"] == 3
    assert result["events"][0]["reason"] == "Unhealthy"
    assert result["logs"][0]["logs"] == "model missing"
