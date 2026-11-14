from unittest.mock import Mock

import requests

from agent import orchestrator as o
from agent.planners import gemini_plan


def test_gemini_selects_structured_tools_without_exposing_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-secret")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    response = Mock()
    response.json.return_value = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "dvc_tracked", "args": {}}}]}}
        ]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr(o.requests, "post", post)
    monkeypatch.setitem(
        o.TOOLS,
        "dvc_tracked",
        (lambda: {"datasets": ["data/iris.csv.dvc"]}, o.s.EmptyInput, "datasets"),
    )
    result = o.ask("What datasets do we have?", mode="llm")
    assert result.evidence[0].tool == "dvc_tracked"
    assert result.evidence[0].status == "ok"
    assert "fake-test-secret" not in result.model_dump_json()
    assert "fake-test-secret" not in post.call_args.args[0]
    assert post.call_args.kwargs["headers"]["x-goog-api-key"] == "fake-test-secret"
    declarations = post.call_args.kwargs["json"]["tools"][0]["functionDeclarations"]
    assert declarations[0]["parametersJsonSchema"]["type"] == "object"


def test_gemini_rejects_unrequested_write(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-secret")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    response = Mock()
    response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "kube_restart", "args": {"confirmed": True}}}
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(o.requests, "post", Mock(return_value=response))
    result = o.ask("Should we retrain the model?", mode="llm", confirmed=True)
    assert not result.evidence
    assert result.uncertainties


def test_gemini_missing_key_fails_before_network(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    post = Mock()
    monkeypatch.setattr(o.requests, "post", post)
    import pytest

    with pytest.raises(RuntimeError, match="not configured"):
        gemini_plan("List data", [])
    post.assert_not_called()


def test_project_access_error_is_helpful_and_secret_safe(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-secret")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    response = Mock(status_code=403)
    response.raise_for_status.side_effect = requests.HTTPError("unsafe key fake-test-secret")
    monkeypatch.setattr(o.requests, "post", Mock(return_value=response))
    result = o.ask("Which datasets are tracked?", mode="llm")
    assert "project access denied (403)" in result.uncertainties[0]
    assert "fake-test-secret" not in result.model_dump_json()
    assert not result.evidence
