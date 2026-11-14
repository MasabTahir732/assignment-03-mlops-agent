# Five-scenario demonstration checklist

Use an activated virtual environment from the repository root. Run setup, start
MLflow/Airflow, unpause the DAG, train at least two models, build the inference image
and deploy it as described in README. Commit tracked changes before training.
Use `--mode llm` for AI tool selection; omit it for the deterministic walkthrough.

| Scenario | Agent request / command | Evidence to show |
| --- | --- | --- |
| 1. Dataset provenance | `python -m agent.main 'Compare dataset v1 and v2'` | Two distinct DVC hashes and immutable Git commits; small sizes |
| Reproduction | `powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Revision v1` | Actual DVC pull, 120 rows, matching v1 MD5; current v2 still 150 rows |
| 2. Model selection | `python -m agent.main 'Compare recent runs and recommend the best model by f1'` | Recorded run IDs, actual F1/accuracy, selection explanation, provenance tags |
| 3. Pipeline execution | `python -m agent.main 'Trigger model training pipeline' --confirm` | Actual submitted run ID; poll pipeline status; show all five task statuses in Airflow |
| 4. Deployment diagnosis | `python -m agent.main 'Why is my prediction API unavailable?'` | Actual two-replica availability, image version, Pod readiness, restart counts, events/logs |
| 5. Integration | `python -m agent.main 'Find the best recent model, identify the dataset version used to train it, check deployment status, and explain whether retraining is necessary.' --mode llm` | MLflow + recorded DVC commit + Kubernetes + Airflow responses; facts, unknowns, recommendations |

For scenario 3, a submitted/queued run is not success: show the completed run and
task-instance details. For scenario 4, inspect `/metadata` after port-forwarding to
verify the actual deployed model run ID; do not infer it from the image tag.

Demonstrate the two failure injections from README, diagnose the *first failed
task*, and distinguish it from tasks blocked by upstream failure. Explain why a
deployment problem usually calls for infrastructure recovery rather than retraining.

Export the five actual conversations with `python -m scripts.capture_evidence
--mode llm --run-id ACTUAL_RUN_ID`. Save screenshots only from running services.
For an unexpected failure, check connectivity/authentication first, then task logs,
DVC/Git consistency, quality gate, and Pod events/probes. State uncertainty when a
service is unreachable.
