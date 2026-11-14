"""Gemini structured function calls. Only planning is delegated to the cloud model."""

import os

import requests

SYSTEM = (
    "Select tools to answer the user. You are a planner only. Never invent tool results. "
    "For best model + dataset + deployment + retraining requests call mlflow_best, "
    "dvc_current, kube_diagnose, airflow_runs. For metric comparisons use mlflow_best. "
    "Never trigger training from a recommendation request. Never grant confirmation. "
    "Select mutations only when explicitly requested. Available workflow: iris_training; "
    "namespace: mlops-demo. Do not guess run IDs or Pod names; inspect lists first."
)


def gemini_plan(prompt: str, schemas: list[dict]) -> list[tuple[str, dict]]:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Gemini API key is not configured in .env")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    declarations = [
        {
            "name": x["function"]["name"],
            "description": x["function"]["description"],
            "parametersJsonSchema": x["function"]["parameters"],
        }
        for x in schemas
    ]
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"functionDeclarations": declarations}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
        },
        timeout=45,
    )
    response.raise_for_status()
    parts = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    calls = [part["functionCall"] for part in parts if "functionCall" in part]
    return [(call["name"], call.get("args", {})) for call in calls]
