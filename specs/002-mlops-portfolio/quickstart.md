# Quickstart: MLOps Learning Portfolio

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-22

This guide walks through three ways to run the pipeline, in learning order:
1. **Local via DVC** — fastest, no containers needed
2. **Airflow DAG** — local orchestration with UI
3. **Kubeflow Pipelines** — container-native on Kubernetes

---

## Prerequisites

- Python 3.12 (`uv` installed)
- Docker Desktop (with Kubernetes enabled for step 3)
- Git

---

## One-Time Setup

### 1. Install dependencies

```bash
uv add dvc great-expectations kfp
uv sync
```

### 2. Initialise DVC

```bash
dvc init
dvc remote add -d local data/dvc-remote
git add .dvc .dvcignore
git commit -m "chore: init DVC with local remote"
```

### 3. Track the raw dataset

```bash
dvc add data/raw/CGR_Crash_Data.csv
git add data/raw/CGR_Crash_Data.csv.dvc data/raw/.gitignore
git commit -m "data: track raw crash CSV with DVC"
dvc push
```

### 4. Initialise Great Expectations

```bash
uv run python -c "import great_expectations as gx; gx.get_context(mode='file', project_root_dir='great_expectations')"
```

---

## Path 1: Run the Pipeline via DVC

This is the simplest path — no containers, no scheduler, just the pipeline.

```bash
# Run all stages (first run — all stages execute)
uv run dvc repro

# Re-run after a parameter change
# Edit params.yaml → change model.n_select from 3 to 5
uv run dvc repro   # only train + evaluate + register re-run

# Re-run after a data change
# Replace data/raw/CGR_Crash_Data.csv → run dvc add again
dvc add data/raw/CGR_Crash_Data.csv
uv run dvc repro   # all stages re-run

# Check which stages are cached (nothing to do)
uv run dvc status

# Reproduce a past run (git commit abc123 with its data version)
git checkout abc123
dvc pull
uv run dvc repro
```

**Verify**: `models/best_ml_model.pkl` and `docs/evaluation_report.json` exist.
Open `great_expectations/uncommitted/data_docs/index.html` in a browser to see the
validation report.

---

## Path 2: Run via Airflow

```bash
# Start Airflow (standalone mode)
cd airflow
uv run airflow standalone
# Open http://localhost:8080 (user: admin, password printed in terminal)
```

1. In the Airflow UI, find the DAG `crash_severity_pipeline`
2. Toggle it **On**
3. Click **Trigger DAG** (▶ button)
4. Watch each task turn green in the Graph view
5. Click any task → **Log** to see its output

**If a task fails**:
1. Fix the issue (e.g., correct the data quality problem)
2. Click the failed task in the UI
3. Click **Clear** → **Yes**
4. The task re-runs; upstream completed tasks are skipped

```bash
# View MLflow tracking UI (run in a separate terminal)
uv run mlflow ui
# Open http://localhost:5000
```

---

## Path 3: Run via Kubeflow Pipelines

### One-Time Kubernetes Setup

```bash
# Enable Kubernetes in Docker Desktop:
# Settings → Kubernetes → Enable Kubernetes → Apply & Restart

# Verify cluster is running
kubectl cluster-info

# Install Kubeflow Pipelines standalone (~5 minutes)
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources/base?ref=2.2.0"
kubectl wait --for=condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=2.2.0"

# Wait for pods to be ready (~3 minutes)
kubectl -n kubeflow wait --for=condition=Ready pods --all --timeout=300s

# Access the KFP UI
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80
# Open http://localhost:8888
```

### Build the Docker Image

```bash
# Build the stage image (run from repo root)
docker build -f docker/Dockerfile -t mlops-portfolio:latest .

# Verify any stage runs in the container
docker run --rm mlops-portfolio:latest python -m src.ingest.run
```

### Compile and Submit the Pipeline

```bash
# Compile the KFP pipeline to YAML
uv run python pipelines/kubeflow/pipeline.py
# Produces: pipelines/kubeflow/pipeline.yaml

# Upload and run via the KFP UI:
# 1. Open http://localhost:8888
# 2. Pipelines → Upload pipeline → select pipeline.yaml
# 3. Runs → Create run → select pipeline → Start
```

**Verify**: In the KFP UI, all six steps show green checkmarks. In the MLflow UI
(`http://localhost:5000`), the run appears with tag `orchestrator=kubeflow`.

---

## Verify the Registered Model

```bash
# Load the champion model from the registry (works regardless of which orchestrator ran)
uv run python - <<'EOF'
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/crash-severity@champion")
print("Model loaded:", type(model))
EOF
```

Expected output: `Model loaded: <class 'mlflow.pyfunc.PyFuncModel'>`

---

## Cheat Sheet

| Action | Command |
|---|---|
| Run full pipeline | `uv run dvc repro` |
| Check pipeline status | `uv run dvc status` |
| Push artifacts to remote | `dvc push` |
| Pull artifacts from remote | `dvc pull` |
| View MLflow UI | `uv run mlflow ui` |
| Start Airflow | `cd airflow && uv run airflow standalone` |
| Port-forward KFP UI | `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80` |
| Build Docker image | `docker build -f docker/Dockerfile -t mlops-portfolio:latest .` |
| Load registered model | `mlflow.pyfunc.load_model("models:/crash-severity@champion")` |
