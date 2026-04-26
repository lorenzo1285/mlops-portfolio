# Quickstart: MLOps Learning Portfolio

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-22 | **Updated**: 2026-04-26

This guide walks through two ways to run the 10-stage pipeline, in learning order:
1. **Local via DVC** — fastest, no containers needed
2. **Kubeflow Pipelines** — container-native on Kubernetes

**Pipeline stages**: `validate → ingest → featurize → train_vae → encode → train_ml → train_dl → evaluate → tune → register`

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

**Verify**: The following artifacts exist after a successful run:
- `models/vae_encoder.pth` + `models/vae_decoder.pth` — trained VAE
- `data/processed/Z_train_augmented.npy`, `Z_val.npy`, `Z_test.npy` — latent vectors
- `models/best_ml_model.pkl` — best XGBoost seed
- `models/mlp_model.pth` — best MLP seed
- `docs/ab_test_comparison.json` — A/B test result with p-value and winner
- `docs/evaluation_report.json` — constitutional gate PASS/FAIL

Open `great_expectations/gx/uncommitted/data_docs/index.html` in a browser to see the
validation report. Open the MLflow UI (`uv run mlflow ui`) to inspect the
`crash-severity-vae` ELBO convergence curve.

---

## Path 2: Run via Kubeflow Pipelines

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

**Verify**: In the KFP UI, all 10 steps show green checkmarks with correct dependency
arrows. In the MLflow UI (`http://localhost:5000`), runs appear under `crash-severity-vae`,
`crash-severity-ml`, `crash-severity-dl` tagged `orchestrator=kubeflow`.

---

## Verify the Registered Model

```bash
# Load the champion model from the registry (works regardless of which orchestrator ran)
uv run python - <<'EOF'
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/crash-severity@champion")
print("Model loaded:", type(model))
# Model produces 3-class predictions: 0=PDO, 1=Injury, 2=Fatal
EOF
```

Expected output: `Model loaded: <class 'mlflow.pyfunc.PyFuncModel'>`

## Inspect the VAE ELBO Curve

```bash
# Open MLflow UI and navigate to crash-severity-vae experiment
uv run mlflow ui
# http://localhost:5000 → Experiments → crash-severity-vae → select run → Metrics → vae_elbo
```

The ELBO curve should decrease monotonically over at least 80% of training epochs.
A diverging ELBO (increasing) means β may be too high — run `dvc repro tune` to search.

---

## Cheat Sheet

| Action | Command |
|---|---|
| Run full 10-stage pipeline | `uv run dvc repro` |
| Run single stage | `uv run dvc repro <stage>` e.g. `train_vae` |
| Check pipeline status | `uv run dvc status` |
| Push artifacts to remote | `dvc push` |
| Pull artifacts from remote | `dvc pull` |
| View MLflow UI | `uv run mlflow ui` |
| View VAE ELBO curve | MLflow UI → `crash-severity-vae` → Metrics → `vae_elbo` |
| Port-forward KFP UI | `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80` |
| Port-forward Katib UI | `kubectl port-forward -n kubeflow svc/katib-ui 8080:80` |
| Build Docker image | `docker build -f docker/Dockerfile -t mlops-portfolio:latest .` |
| Compile KFP pipeline | `uv run python pipelines/kubeflow/pipeline.py` |
| Load registered model | `mlflow.pyfunc.load_model("models:/crash-severity@champion")` |
