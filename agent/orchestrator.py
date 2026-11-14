"""Structured tools, guarded execution, and evidence-based cross-system reasoning."""

import json
import logging
import os
import re
import subprocess
import time
from uuid import uuid4

import requests
from pydantic import ValidationError

from agent import schemas as s
from agent.planners import SYSTEM, PlannerUnavailable, gemini_plan
from agent.tools import airflow_tools as af
from agent.tools import dvc_tools as dvc
from agent.tools import kubernetes_tools as kube
from agent.tools import mlflow_tools as mf

LOGGER = logging.getLogger("mlops.tools")
TOOLS = {
    "dvc_tracked": (dvc.tracked, s.EmptyInput, "List DVC dataset pointers"),
    "dvc_current": (dvc.current, s.EmptyInput, "Current dataset hash, Git commit and dirty status"),
    "dvc_status": (dvc.status, s.EmptyInput, "Detect changed tracked data"),
    "dvc_compare": (dvc.compare, s.DvcCompareInput, "Compare two Git dataset revisions"),
    "dvc_reproduce": (dvc.reproduce, s.ReproduceInput, "Instructions to reproduce a Git revision"),
    "mlflow_experiments": (mf.experiments, s.EmptyInput, "List experiments"),
    "mlflow_runs": (mf.runs, s.RunsInput, "Recent iris-demo training runs"),
    "mlflow_details": (mf.details, s.RunInput, "Run metrics, provenance, parameters, artifacts"),
    "mlflow_best": (mf.best, s.BestInput, "Compare recent runs and select a metric candidate"),
    "mlflow_compare": (mf.compare, s.CompareRunsInput, "Compare specified runs"),
    "mlflow_registry": (mf.registry, s.EmptyInput, "Registered model versions"),
    "airflow_dags": (af.dags, s.EmptyInput, "List Airflow DAGs"),
    "airflow_runs": (af.runs, s.DagInput, "Recent training workflow status"),
    "airflow_trigger": (
        af.trigger,
        s.TriggerInput,
        "Trigger approved training DAG; confirmation needed",
    ),
    "airflow_tasks": (af.tasks, s.TaskInput, "DAG task statuses and execution information"),
    "airflow_diagnose": (af.diagnose, s.TaskInput, "Failed tasks and available task logs"),
    "kube_pods": (kube.pods, s.EmptyInput, "Application Pods"),
    "kube_deployments": (kube.deployments, s.EmptyInput, "Deployment versions and replica status"),
    "kube_services": (kube.services, s.EmptyInput, "Application Services"),
    "kube_logs": (kube.logs, s.PodInput, "Logs for an application Pod"),
    "kube_diagnose": (kube.diagnose, s.EmptyInput, "Diagnose application health, events and logs"),
    "kube_restart": (kube.restart, s.RestartInput, "Restart deployment; confirmation needed"),
}
MUTATIONS = {"airflow_trigger", "kube_restart"}


def tool_schemas():
    schemas = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": schema.model_json_schema(),
            },
        }
        for name, (_, schema, desc) in TOOLS.items()
    ]
    # Single-value enums are understood more consistently by small tool-calling models.
    for tool in schemas:
        for field in tool["function"]["parameters"].get("properties", {}).values():
            if "const" in field:
                field["enum"] = [field.pop("const")]
    return schemas


def execute(name: str, arguments: dict, request_id: str, confirmed: bool = False):
    started = time.monotonic()
    data, error = None, None
    try:
        if name not in TOOLS:
            raise ValueError("Unknown tool")
        function, schema, _ = TOOLS[name]
        arguments = dict(arguments)
        if name in MUTATIONS:
            # Only the CLI's trusted --confirm grants consent. A model cannot grant it.
            arguments["confirmed"] = confirmed
        validated = schema(**arguments)
        data = function(**validated.model_dump())
    except subprocess.TimeoutExpired:
        error = "Tool timed out; no operational result is known."
    except requests.Timeout:
        error = "Service timed out; no operational result is known."
    except PermissionError as exc:
        error = str(exc)
    except (ValueError, ValidationError):
        error = "Invalid tool name or arguments. Inspect the tool schema."
    except Exception as exc:
        # Exception text may contain credentials, tokens, or URLs. Do not echo it.
        error = f"Tool unavailable ({type(exc).__name__}); no operational result is known."
    elapsed = round((time.monotonic() - started) * 1000, 2)
    result = s.ToolResult(
        tool=name,
        status="error" if error else "ok",
        data=data,
        error=error,
        request_id=request_id,
        duration_ms=elapsed,
    )
    LOGGER.info(
        json.dumps(
            {
                "tool": name,
                "status": result.status,
                "duration_ms": elapsed,
                "request_id": request_id,
            }
        )
    )
    return result


def rule_plan(prompt: str) -> list[tuple[str, dict]]:
    text = prompt.lower()
    # This scenario is always read-only, including when it contains 'retraining'.
    if ("best" in text or "recent model" in text) and any(
        w in text for w in ("deploy", "retrain", "kubernetes", "running")
    ):
        return [
            ("mlflow_best", {}),
            ("dvc_current", {}),
            ("kube_diagnose", {}),
            ("airflow_runs", {}),
        ]
    if "restart" in text:
        return [("kube_restart", {})]
    if any(w in text for w in ("trigger", "start training", "run pipeline")):
        failure = (
            "validation"
            if "validation failure" in text
            else ("quality" if "quality failure" in text else "none")
        )
        return [("airflow_trigger", {"inject_failure": failure})]
    if any(w in text for w in ("pod", "deploy", "unavailable", "kubernetes")):
        return [("kube_diagnose", {})]
    if "dag" in text or "pipeline" in text or "airflow" in text:
        return [("airflow_runs", {})]
    ids = re.findall(r"\b[a-fA-F0-9]{32}\b", prompt)
    if ids and "compare" in text and len(ids) >= 2:
        return [("mlflow_compare", {"run_ids": ids})]
    if ids:
        return [("mlflow_details", {"run_id": ids[0]})]
    if "best" in text or "f1" in text or "accuracy" in text:
        return [("mlflow_best", {"metric": "accuracy" if "accuracy" in text else "f1"})]
    if "registry" in text:
        return [("mlflow_registry", {})]
    if "experiment" in text or "training runs" in text:
        return [("mlflow_experiments", {}), ("mlflow_runs", {})]
    if "compare" in text and ("dataset" in text or "data" in text):
        refs = re.findall(r"\bv\d+(?:\.\d+)*\b", text)
        if len(refs) == 2:
            return [("dvc_compare", {"old": refs[0], "new": refs[1]})]
        raise ValueError("Supply two revision tags, for example compare dataset v1 and v2")
    if "reproduce" in text:
        refs = re.findall(r"\bv\d+(?:\.\d+)*\b", text)
        if not refs:
            raise ValueError("Supply a revision tag, for example reproduce v1")
        return [("dvc_reproduce", {"revision": refs[0]})]
    if any(w in text for w in ("dataset", "dvc", "data version", "data changed")):
        return [("dvc_tracked", {}), ("dvc_current", {})]
    raise ValueError(
        "Use a dataset, experiment, pipeline or deployment request; see README examples"
    )


def ollama_plan(prompt: str) -> list[tuple[str, dict]]:
    schemas = tool_schemas()
    choices = [
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": [t["function"]["name"]]},
                "arguments": t["function"]["parameters"],
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for t in schemas
    ]
    plan_schema = {
        "type": "object",
        "properties": {
            "calls": {"type": "array", "items": {"anyOf": choices}, "minItems": 0, "maxItems": 12}
        },
        "required": ["calls"],
        "additionalProperties": False,
    }
    response = requests.post(
        os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat",
        json={
            "model": os.getenv("OLLAMA_MODEL", "qwen3:1.7b"),
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2048},
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM + '\nReturn JSON {"calls":[{"name":"TOOL_NAME",'
                    '"arguments":{}}]}. Choose the smallest sufficient tool plan. '
                    "For a relevant request, include at least one call. "
                    "Tool registry: " + json.dumps([t["function"] for t in schemas]),
                },
                {"role": "user", "content": prompt},
            ],
            # Constrained JSON makes small CPU models obey the same function schemas.
            # Tool selection and arguments still come from the actual LLM, not rules.
            "format": plan_schema,
        },
        timeout=90,
    )
    response.raise_for_status()
    message = response.json()["message"]
    if message.get("tool_calls"):
        return [(c["function"]["name"], c["function"]["arguments"]) for c in message["tool_calls"]]
    calls = json.loads(message["content"])["calls"]
    return [(c["name"], c["arguments"]) for c in calls]


def llm_plan(prompt: str) -> list[tuple[str, dict]]:
    provider = os.getenv("LLM_PROVIDER") or (
        "gemini" if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") else "ollama"
    )
    planning_prompt = prompt
    for _ in range(3):
        plan = (
            gemini_plan(planning_prompt, tool_schemas())
            if provider == "gemini"
            else ollama_plan(planning_prompt)
        )
        if not plan or len(plan) > 12:
            raise ValueError("LLM did not return a bounded structured tool plan")
        invalid = []
        # Validate the plan before any operation. Repair inputs through the same LLM,
        # never by inventing results or silently substituting the rules planner.
        for name, arguments in plan:
            if name in MUTATIONS:
                explicit = (name == "kube_restart" and "restart" in prompt.lower()) or (
                    name == "airflow_trigger"
                    and any(
                        w in prompt.lower() for w in ("trigger", "start training", "run pipeline")
                    )
                )
                if not explicit:
                    raise PermissionError("LLM proposed an unrequested mutation")
            if name not in TOOLS:
                invalid.append({"tool": name, "error": "Tool is not in the registry"})
                continue
            try:
                TOOLS[name][1].model_validate(arguments)
            except ValidationError as exc:
                invalid.append(
                    {
                        "tool": name,
                        "errors": [
                            {"field": list(e["loc"]), "message": e["msg"]}
                            for e in exc.errors(include_input=False)
                        ],
                    }
                )
        if not invalid:
            return plan
        planning_prompt = (
            prompt
            + "\nYour previous structured calls were rejected before execution: "
            + json.dumps(invalid)
            + ". Return corrected calls using only the provided tool schemas. "
            "No tools have executed yet. Do not invent any operational results."
        )
    raise ValueError("LLM could not produce valid tool arguments after three planning attempts")


def summarize(evidence: list[s.ToolResult]):
    facts, unknown, recommendations = [], [], []
    successful = {e.tool: e.data for e in evidence if e.status == "ok"}
    for result in evidence:
        if result.status == "error":
            unknown.append(f"{result.tool}: {result.error}")
        else:
            facts.append(f"{result.tool} returned actual tool evidence (see evidence).")
    selection = successful.get("mlflow_best", {})
    candidate = selection.get("candidate") or successful.get("mlflow_details")
    if candidate:
        facts.append(f"Candidate {candidate['run_id']} recorded metrics {candidate['metrics']}.")
        tags = candidate.get("tags", {})
        for key in ("dataset_hash", "dataset_git_commit", "dataset_version", "quality_gate"):
            if key in tags:
                facts.append(f"Training provenance {key}={tags[key]}.")
            else:
                unknown.append(f"Candidate has no recorded {key}.")
        current = successful.get("dvc_current")
        if current and tags.get("dataset_hash"):
            same = tags["dataset_hash"] == current["dvc_hash"]
            facts.append(f"Candidate DVC hash matches current pointer: {same}.")
            if not same:
                recommendations.append(
                    "Dataset pointer changed; assess the new data before "
                    "deciding whether retraining is justified."
                )
        if tags.get("quality_gate") == "failed":
            recommendations.append(
                "Retraining is recommended: the selected run has a recorded "
                "failed quality gate. Investigate data and parameters first."
            )
        elif "mlflow_best" in successful:
            recommendations.append(
                "No retraining requirement is established by these results. "
                "No freshness policy or degradation evidence was supplied."
            )
        deployment = successful.get("kube_diagnose", {})
        if deployment:
            unknown.append(
                "Image availability does not prove the deployed model matches the "
                "candidate. Verify /metadata run_id against the selected run."
            )
    elif "mlflow_best" in successful:
        unknown.append("No finished recent run has the requested metric; no best model is known.")
    health = successful.get("kube_diagnose")
    if health is not None:
        facts.append(
            f"Kubernetes application healthy={health['healthy']}; versions={health['deployments']}."
        )
        if not health["healthy"]:
            recommendations.append(
                "Investigate returned Pod states, probe events and logs. "
                "Deployment failure alone is not evidence for retraining."
            )
    flows = successful.get("airflow_runs", {}).get("dag_runs", [])
    if flows:
        latest = flows[0]
        facts.append(f"Latest workflow {latest.get('dag_run_id')}: {latest.get('state')}.")
        if latest.get("state") == "failed":
            recommendations.append("Inspect failed Airflow tasks and logs before retriggering.")
    if "airflow_trigger" in successful:
        facts.append("Training was submitted. Submission does not establish execution success.")
        recommendations.append(
            "Poll airflow_runs, then airflow_tasks/airflow_diagnose for the run ID."
        )
    if any(e.status == "error" for e in evidence):
        recommendations.append("Restore unavailable tools and repeat the read-only request.")
    return facts, unknown, recommendations


def ask(prompt: str, mode: str = "rules", confirmed: bool = False) -> s.AgentResponse:
    request_id = uuid4().hex
    try:
        if not prompt.strip() or len(prompt) > 4000:
            raise ValueError("Request must contain 1 to 4000 characters")
        plan = llm_plan(prompt) if mode == "llm" else rule_plan(prompt)
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, (ValueError, PlannerUnavailable))
            else "Planner unavailable or request rejected"
        )
        return s.AgentResponse(
            request_id=request_id,
            mode=mode,
            facts=[],
            uncertainties=[message],
            recommendations=[],
            evidence=[],
        )
    evidence = [execute(n, a, request_id, confirmed) for n, a in plan]
    # Follow recorded provenance, not a guessed dataset version.
    best = next(
        (e.data.get("candidate") for e in evidence if e.tool == "mlflow_best" and e.status == "ok"),
        None,
    )
    if best:
        evidence.append(execute("mlflow_details", {"run_id": best["run_id"]}, request_id))
        revision = best.get("tags", {}).get("dataset_git_commit")
        if revision:
            evidence.append(execute("dvc_reproduce", {"revision": revision}, request_id))
    facts, unknown, recommendations = summarize(evidence)
    return s.AgentResponse(
        request_id=request_id,
        mode=mode,
        facts=facts,
        uncertainties=unknown,
        recommendations=recommendations,
        evidence=evidence,
    )
