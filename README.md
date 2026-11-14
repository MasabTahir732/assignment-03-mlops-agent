# Assignment 03 — MLOps AI Agent

An evidence-based command-line assistant for **DVC, MLflow, Airflow 3 and Kubernetes**.
It answers dataset, experiment, pipeline and deployment questions, then combines all
four systems for the fifth integration requirement. The directory structure follows
the assignment PDF. No paid API key is required.

**Submission report:** [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) includes the complete
architecture, permissions, five tested conversations, actual exported evidence,
two failure investigations, limitations and improvements. The first
[real infrastructure workflow](https://github.com/MasabTahir732/assignment-03-mlops-agent/actions/runs/43591743449)
passed with all four services running. Evidence is committed under `screenshots/live/`.

**Tiny data:** Iris v1 has 120 rows; v2 has 150 rows. Each CSV is under 5 KB. Both
versions are generated from scikit-learn's bundled Iris data; no dataset download.
The DVC remote is a local ignored folder. DVC versions real CSV contents, not dummy data.

## 1. Install and prepare the repository (Windows PowerShell)

Use Python **3.12**, Git and PowerShell. Docker Desktop with Linux containers and
Minikube + a working kubectl are needed for the full infrastructure demo. Allow
roughly 6–8 GB RAM for Docker/Airflow/Minikube; the dataset itself is tiny.

```powershell
Set-Location 'D:\SEM 7\assignment-03-mlops-agent'
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
python -m pytest -q
ruff check .
```

If activation is restricted, use `.\.venv\Scripts\python.exe` in place of `python`.
Setup creates `.env` with random local service secrets, initializes Git/DVC for a
new project, creates commits/tags `v1` and `v2`, and pushes both small objects to
`.dvc-remote/`. On a fresh GitHub clone it regenerates the two cache objects without
changing the committed v2 pointer. Never commit `.env`, cache files, or credentials.
Git needs a configured name/email for the initial commits.

Linux/macOS alternative: `python3.12 -m venv .venv`, activate it, install
`pip install -e '.[dev]'`, initialize Git/DVC and use the versioning commands below.
Replace PowerShell paths and environment assignment syntax with your shell's equivalents.

## 2. Start MLflow and create real training runs

**Easy local option:** in a separate PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_mlflow.ps1
```

In the activated project terminal:

```powershell
python -m training.train --trees 10 --max-depth 1
python -m training.train --trees 50 --max-depth 3
python -m training.train --trees 80 --max-depth 5
python -m agent.main 'Compare recent training runs and find the best model by f1'
python -m agent.main --tool mlflow_registry
```

Open <http://localhost:5000>. Each run records F1, accuracy, parameters, DVC hash,
Git commit, dataset tag, candidate artifact and gate status. The quality gate is
macro F1 >= 0.90. Passing candidates enter the registry; a failing candidate is
logged for diagnosis and cannot replace `models/model.joblib`.
The shallow `--max-depth 1` candidate deliberately illustrates a failed gate on
this split (F1 about 0.866); continue to the stronger candidates (F1 about 0.949).

Commit tracked source changes **before training**. Training refuses dirty tracked
files or CSV contents that disagree with their DVC pointer. Generated models,
reports and MLflow database files are ignored by Git.

Optional DVC pipeline execution (with MLflow running): `python -m dvc repro`.
This creates `dvc.lock`; commit it after running so subsequent provenance stays clean.

## 3. Run Airflow 3 and MLflow with Docker

Stop the local MLflow terminal first to free port 5000. Compose provides its own
persistent MLflow store, PostgreSQL metadata database and an Airflow standalone
container (API server, DAG processor, scheduler and local task execution).
Runs from the separate local SQLite MLflow database are not migrated automatically.

```powershell
docker compose up -d --build
docker compose logs airflow
```

Open <http://localhost:8080>. SimpleAuth creates the `admin` user; read the generated
password in `docker compose logs airflow` or:

```powershell
docker compose exec airflow cat /opt/airflow/simple_auth_manager_passwords.json.generated
```

Put that password in the ignored `.env` as `AIRFLOW_PASSWORD=...` (no quotes needed).
The CLI exchanges it for a JWT using `/auth/token`; operations use `/api/v2`.
Alternatively set `AIRFLOW_TOKEN` to a valid JWT. Environment variables override `.env`.
Unpause `iris_training` in the Airflow UI before triggering it.

```powershell
python -m agent.main --tool airflow_dags
python -m agent.main 'Trigger the model training pipeline' --confirm
python -m agent.main 'Check training pipeline status'
python -m agent.main --tool airflow_tasks --args '{"dag_run_id":"COPY_ACTUAL_RUN_ID"}'
python -m agent.main --tool airflow_diagnose --args '{"dag_run_id":"COPY_ACTUAL_RUN_ID"}'
```

PowerShell 5 may strip quotes from native JSON arguments. If that happens, use
PowerShell 7 or the Airflow UI for task inspection; natural-language commands above
do not require JSON. `--list-tools` prints every validated JSON schema.

The five DAG tasks are `validate_data → preprocess_data → train_model →
evaluate_and_log → quality_gate`, with one retry and a ten-second retry delay.
The agent reports a trigger as **submitted**, then reads actual run/task statuses.
Validation failure prevents training; quality failure prevents model promotion.
The DAG does not automatically deploy or change Kubernetes state.

## 4. Inference and Kubernetes (two replicas)

First obtain at least one passing model from local training or the DAG. Models are
saved to the shared `models/` directory. Local API check:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000/docs>. Try `/predict` with:

```json
{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
```

In another terminal, deploy to your local Minikube cluster:

```powershell
minikube start --cpus=2 --memory=3072
docker build -t mlops-iris:1.0.0 .
minikube image load mlops-iris:1.0.0
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/rbac.yaml
kubectl -n mlops-demo rollout status deployment/iris-api --timeout=180s
kubectl -n mlops-demo get pods,deployments,services
python -m agent.main 'Why is my prediction API unavailable?'
kubectl -n mlops-demo port-forward service/iris-api 8001:80
```

Visit <http://localhost:8001/metadata> to see the actual loaded run ID and dataset
hash; compare these with the recommended MLflow candidate. A running image tag does
not establish that the best candidate is deployed. To deploy a newly trained model,
build/load a **new image tag**, update `deployment.yaml` and explicitly apply it.

The manifests include two replicas, a ClusterIP Service, liveness/readiness probes,
resource requests/limits, a ConfigMap and a read-only RBAC role. The application is
alive without a model but **not ready** until a quality-approved model bundle loads.

Only the approved namespace (default `mlops-demo`) and `iris-api` application are
accessible through agent tools. A restart requires explicit `--confirm`; runtime
credentials must separately grant permission to patch the Deployment. The supplied
reader role intentionally cannot restart it. No shell/exec/delete tools are exposed.

```powershell
python -m agent.main 'Restart the prediction deployment'
# Rejected. To explicitly request and authorize the restart:
python -m agent.main 'Restart the prediction deployment' --confirm
```

## 5. Conversational multi-tool agent

For Gemini, follow the beginner [GEMINI_SETUP.md](GEMINI_SETUP.md) guide. Add your
key only to `.env`, then use `python -m agent.main 'Which datasets are tracked?'
--mode llm --provider gemini`. Gemini is optional; no specific LLM vendor is required.
The CLI automatically uses LLM mode when a Gemini key is configured.

Explicit `--mode rules` selects tools for the documented demonstration phrases
without downloading a language model. Its limits are explicit: it is a deterministic
router, not an LLM. For the **AI agent demonstration**, install Ollama separately,
then use its tool-calling API with a local model (add `--provider ollama`):

```powershell
ollama pull qwen3:1.7b
python -m agent.main 'Which datasets are tracked by DVC?' --mode llm --provider ollama
python -m agent.main 'Find the best recent model, identify the dataset version used to train it, check deployment status, and explain whether retraining is necessary.' --mode llm --provider ollama
```

Ollama must be running at `OLLAMA_URL`; `OLLAMA_MODEL` is configurable. This optional
model download is much larger than the dataset. Tool schemas constrain its inputs,
the executor enforces permissions, and the final response is rendered from actual
evidence rather than free-form model assertions. LLM failure is reported explicitly.
You can use rules mode for a lightweight repeatable walkthrough of the same tools.

On this Windows computer, Docker is installed **inside Ubuntu WSL**. Use these
commands from the project terminal to start the local model without a separate
Windows Ollama installation:

```powershell
wsl --cd 'D:\SEM 7\assignment-03-mlops-agent' -u root -e docker compose --profile llm up -d ollama
wsl --cd 'D:\SEM 7\assignment-03-mlops-agent' -u root -e docker compose exec -T ollama ollama pull qwen3:1.7b
python -m agent.main 'Which datasets are tracked?' --mode llm --provider ollama
```

The optional Compose profile uses a small CPU-only Ollama image; the model download
is approximately 1.4 GB and is kept in a persistent volume. Iris data remains under
3 KB. For the other Docker commands in this README, use the same `wsl --cd ... -u
root -e` prefix if Docker is only available inside WSL.

The mandatory scenario queries MLflow, follows the candidate's recorded dataset
commit, inspects DVC and Kubernetes, checks Airflow, and separates facts from
uncertainties and recommendations. It never triggers retraining. A changed dataset
or unavailable service alone does not justify retraining. A recorded failed gate
supports a recommendation to investigate and retrain; no freshness or degradation
policy is assumed.

## Dataset versioning and reproducible checkout

```powershell
python -m agent.main 'Which dataset version is current?'
python -m agent.main 'Compare dataset v1 and v2'
python -m agent.main 'Reproduce v1'
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1 -Revision v1
```

The reproduction script uses an isolated Git worktree under ignored `reports/`,
configures the original absolute DVC remote, pulls v1 and prints its MD5 and 120-row
count. The active v2 dataset remains unchanged. Both Git tags and DVC content must
be retained; GitHub stores pointers, while setup regenerates the two tiny objects
for a fresh clone. For a shared remote, configure DVC storage and credentials outside Git.

## Two deliberate failures for the defense

1. **Validation:** `python -m agent.main 'Trigger training pipeline with validation failure' --confirm`.
   After the retry, `validate_data` fails; downstream tasks are `upstream_failed`.
   Inspect its logs with `airflow_diagnose`. The injected missing-target error is the
   root cause. Retrigger without the injection to resolve it.
2. **Quality:** `python -m agent.main 'Trigger training pipeline with quality failure' --confirm`.
   Validation through evaluation succeeds; the forced F1 threshold of 1.01 makes
   `quality_gate` fail. MLflow records the measured score and `quality_gate=failed`.
   The existing approved model is preserved; no failed candidate enters the registry.
   Retrigger normally to use the ordinary 0.90 threshold.

For a local failure demonstration, run `python -m training.train --inject-failure validation`
and `python -m training.train --inject-failure quality` with MLflow running.

Optional Kubernetes failure: after an explicit operator decision, set ConfigMap
`MODEL_PATH` to `/app/models/missing.joblib` and restart the Deployment. Pod events
show failing readiness probes and `/health/ready` returns 503; the agent reports
unready replicas. Reapply the provided ConfigMap and explicitly restart to recover.

## Demonstration, evidence and submission

Follow [DEMO.md](DEMO.md) for the five scenarios. Architecture, tool permissions,
failure investigations and limitations are in [architecture.md](architecture.md).
Run the evidence export **after** starting services and completing a real DAG run:

```powershell
python -m scripts.capture_evidence --mode llm --run-id COPY_ACTUAL_RUN_ID
```

This saves five real conversations and available task diagnostics to
`screenshots/evidence.json`. Keep failed-service evidence labeled as such. Capture
the actual UI screens listed in [screenshots/README.md](screenshots/README.md).
Never manufacture screenshots, successful run statuses or cluster results.

CI runs Ruff and pytest on pushes/pull requests with external services mocked.
Tests cover tool selection/execution, multi-tool reasoning, failures/timeouts,
invalid requests, secret-safe logs, denied mutations, data validation, promotion
blocking and real inference against a trained model.

A separate manual **Live infrastructure demonstration** GitHub Actions workflow
creates a real Docker/Kind environment, executes the five-stage Airflow DAG, injects
and recovers from both failures, deploys two API replicas and exports actual service
evidence as the `actual-mlops-evidence` artifact. Its rules-mode conversation capture
is distinct from the optional Gemini live-language demonstration. Ordinary CI stays
mocked. Download the artifact and copy the reviewed JSON into `screenshots/live/`.

When ready to submit, create an empty GitHub repository, then:

```powershell
git add .
git commit -m 'Add verified demo evidence and final report'
git remote add origin https://github.com/YOUR_USERNAME/assignment-03-mlops-agent.git
git branch -M main
git push -u origin main
git push origin v1 v2
```

Review evidence for secrets before committing. Docker/Kubernetes execution and the
optional Ollama model require their runtimes; passing mocked tests is not proof of
a live deployment. `docker compose stop` and `minikube stop` pause the local demo
without deleting its data.

## API references

- [Airflow 3 public API and JWT authentication](https://airflow.apache.org/docs/apache-airflow/stable/security/api.html)
- [SimpleAuth token exchange](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/auth-manager/simple/token.html)
- [MLflow Python client](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
