"""Namespace and label scoped kubectl with an explicit mutation gate."""

import json
import os

from agent.tools.common import command


def _namespace():
    return os.getenv("KUBE_NAMESPACE", "mlops-demo")


def _get(resource: str):
    return json.loads(
        command(
            [
                "kubectl",
                "--request-timeout=15s",
                "-n",
                _namespace(),
                "get",
                resource,
                "-l",
                "app=iris-api",
                "-o",
                "json",
            ]
        )
    )


def pods() -> dict:
    return _get("pods")


def deployments() -> dict:
    return _get("deployments")


def services() -> dict:
    return _get("services")


def logs(pod: str, lines: int = 80) -> dict:
    allowed = {p["metadata"]["name"] for p in pods().get("items", [])}
    if pod not in allowed:
        raise PermissionError("Pod is outside the approved application")
    return {
        "pod": pod,
        "logs": command(
            [
                "kubectl",
                "--request-timeout=15s",
                "-n",
                _namespace(),
                "logs",
                pod,
                "--tail",
                str(lines),
            ]
        ),
    }


def diagnose() -> dict:
    pod_data, deployment_data, service_data = pods(), deployments(), services()
    names = {p["metadata"]["name"] for p in pod_data.get("items", [])}
    events = json.loads(
        command(
            ["kubectl", "--request-timeout=15s", "-n", _namespace(), "get", "events", "-o", "json"]
        )
    )
    app_events = [
        e
        for e in events.get("items", [])
        if e.get("involvedObject", {}).get("name") in names | {"iris-api"}
    ]
    health = []
    for p in pod_data.get("items", []):
        states = p.get("status", {}).get("containerStatuses", [])
        health.append(
            {
                "pod": p["metadata"]["name"],
                "phase": p.get("status", {}).get("phase"),
                "ready": bool(states) and all(s.get("ready") for s in states),
                "restarts": sum(s.get("restartCount", 0) for s in states),
                "container_states": states,
            }
        )
    log_evidence = []
    for p in health[:4]:
        if not p["ready"] or p["restarts"]:
            try:
                log_evidence.append(logs(p["pod"]))
            except (RuntimeError, OSError):
                log_evidence.append({"pod": p["pod"], "logs_unavailable": True})
    versions = [
        {
            "deployment": d["metadata"]["name"],
            "images": [c["image"] for c in d["spec"]["template"]["spec"]["containers"]],
            "desired": d["spec"].get("replicas", 1),
            "available": d.get("status", {}).get("availableReplicas", 0),
        }
        for d in deployment_data.get("items", [])
    ]
    healthy = (
        bool(health)
        and all(p["ready"] for p in health)
        and bool(versions)
        and all(v["available"] >= v["desired"] for v in versions)
        and bool(service_data.get("items"))
    )
    return {
        "namespace": _namespace(),
        "healthy": healthy,
        "pods": health,
        "deployments": versions,
        "services": service_data,
        "events": app_events,
        "logs": log_evidence,
        "scope": "Kubernetes readiness and availability; no external traffic test",
    }


def restart(namespace: str = "mlops-demo", confirmed: bool = False) -> dict:
    if not confirmed or namespace != _namespace():
        raise PermissionError("Restart requires explicit confirmation and the approved namespace")
    return {
        "output": command(
            [
                "kubectl",
                "--request-timeout=15s",
                "-n",
                _namespace(),
                "rollout",
                "restart",
                "deployment/iris-api",
            ]
        )
    }
