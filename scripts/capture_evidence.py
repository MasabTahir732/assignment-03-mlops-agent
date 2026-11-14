"""Save actual conversations. Failed/unavailable services remain marked as errors."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from agent.orchestrator import ask, execute
from agent.tools.common import ROOT

SCENARIOS = [
    ("dataset", "Which dataset version is current?"),
    ("selection", "Compare the recent runs and find the best model based on f1"),
    ("pipeline", "Check the training pipeline status"),
    ("deployment", "Why is my prediction API unavailable?"),
    (
        "integration",
        "Find the best recent model, identify the dataset version used to train it, "
        "check deployment status, and explain whether retraining is necessary.",
    ),
]


def main():
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="screenshots/evidence.json")
    parser.add_argument("--mode", choices=["rules", "llm"], default="rules")
    parser.add_argument("--run-id", help="Append actual Airflow task diagnostics for this run ID")
    args = parser.parse_args()
    conversations = [
        {"scenario": name, "user": prompt, "response": ask(prompt, args.mode).model_dump()}
        for name, prompt in SCENARIOS
    ]
    result = {"captured_at": datetime.now(timezone.utc).isoformat(), "conversations": conversations}
    if args.run_id:
        result["airflow_diagnostics"] = execute(
            "airflow_diagnose", {"dag_run_id": args.run_id}, "evidence-capture"
        ).model_dump()
    path = Path(args.output)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    errors = sum(e["status"] == "error" for c in conversations for e in c["response"]["evidence"])
    print(f"Saved {len(conversations)} conversations to {path}; unavailable tool calls={errors}")


if __name__ == "__main__":
    main()
