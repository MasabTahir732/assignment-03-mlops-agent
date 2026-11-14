Store **actual** UI screenshots here after running the demo:

The `live/` directory already contains actual exported evidence from the successful
four-service demonstration. The assignment permits exported evidence in place of
UI screenshots. `architecture.png` is a generated architecture diagram, not a
screenshot of a running service.

- `01-dvc.png`: `dvc diff v1 v2` plus reproduced v1 hash / 120 rows.
- `02-mlflow.png`: runs, F1 comparison, dataset tags, registered candidate.
- `03-airflow.png`: five-stage successful DAG and individual task statuses.
- `04-kubernetes.png`: two ready replicas, Service, events/log diagnosis.
- `05-agent.png`: integrated request with evidence from all four technologies.
- `06-failures.png`: validation failure and failed quality gate, with task logs.

`python -m scripts.capture_evidence` exports real JSON conversations here. Errors in
that file mean a service was unavailable; they are not successful demonstration evidence.
Review exported logs for sensitive information before committing evidence.
