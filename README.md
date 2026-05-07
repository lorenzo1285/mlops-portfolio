# MLOps Portfolio — Crash Severity Classification

> **Work in progress.** Core pipeline (stages 1–9) is functional end-to-end. GMM integration (evaluation, registration, HPO dispatch) is in active development on branch `003-gmm-classifier`.

A production-style MLOps pipeline on the City of Grand Rapids crash dataset (74,309 records, 142 features). The goal is to classify crash severity into three classes — **PDO** (property damage only), **Injury**, and **Fatal** — while demonstrating the full MLOps toolchain from raw data to a versioned champion model.

---

## Pipeline

```
validate → ingest → featurize → ┬─ train_vae ─┐
                                 └─ augment   ─┤
                                               ▼
                                            encode → ┬─ train_ml  ─┐
                                                      ├─ train_dl  ─┤
                                                      └─ train_gmm ─┤
                                                                     ▼
                                                                 evaluate → tune → register
```

| # | Stage | Description | Status |
|---|-------|-------------|--------|
| 1 | `validate` | Great Expectations suite on raw CSV; halts pipeline on failure | ✅ Done |
| 2 | `ingest` | Copy raw CSV to `data/processed/raw.csv` | ✅ Done |
| 3 | `featurize` | 3-way split (70/15/15); cyclical encoding (HOUR, MONTH); ordinal/categorical encoding; sample-complexity gate | ✅ Done |
| 4 | `train_vae` | Denoising β-VAE with KL annealing → 8-dim latent space (unsupervised) | ✅ Done |
| 5 | `augment` | CTGAN augmentation on Fatal-class training rows; X_val/X_test never touched | ✅ Done |
| 6 | `encode` | Frozen encoder projects all splits to Z-vectors | ✅ Done |
| 7 | `train_ml` | XGBoost classifier on Z-space; N-seed A/B; MLflow tracking | ✅ Done |
| 8 | `train_dl` | Shallow MLP on Z-space; class-weighted cross-entropy; early stopping | ✅ Done |
| 9 | `train_gmm` | Per-class Gaussian Mixture MAP classifier; fatal-prior boost | 🔄 In progress |
| 10 | `evaluate` | Welch's t-test A/B(/C) on macro F1; constitutional gates (F1 > 0.35, fatal recall > 0.50) | 🔄 In progress |
| 11 | `tune` | Optuna HPO (β, fatal ratio, focal loss γ); writes best params to `params.yaml` | ✅ Done |
| 12 | `register` | Promote champion to MLflow registry with `pyfunc` wrapper; write `registry_receipt.json` | ✅ Done |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Pipeline & data versioning | DVC |
| Experiment tracking & registry | MLflow |
| Data quality | Great Expectations v1 |
| Orchestration | Kubeflow Pipelines v2, Apache Airflow |
| HPO | Optuna, Katib |
| Models | XGBoost, PyTorch (MLP), scikit-learn (GMM) |
| Augmentation | CTGAN / TVAE (SDV) |
| Infrastructure | Kubernetes (Docker Desktop), Docker |
| Language & tooling | Python 3.12, uv |

---

## Dataset

**CGR Crash Data** — City of Grand Rapids open data portal  
74,309 crash records · 142 raw features · 3-class target (PDO / Injury / Fatal)  
Class imbalance: ~78% PDO, ~20% Injury, ~2% Fatal

---

## Quick Start

```bash
# Restore pipeline artifacts from DVC remote
uv run dvc pull

# Run full pipeline
uv run dvc repro

# Run a single stage
uv run dvc repro train_gmm

# Launch MLflow UI
uv run mlflow ui   # http://localhost:5000

# Run tests
uv run python -m pytest tests/

# Compile KFP pipeline YAML
python pipelines/kubeflow/pipeline.py
```

---

## Architecture Principles

- **Deep modules** — each stage is one class (`constructor + 1 public method`); `run.py` handles I/O only
- **3-way split discipline** — val set used only for HPO/early-stopping; test set strictly reserved for final A/B evaluation
- **Three-mechanism imbalance handling** — class weights + CTGAN augmentation + KL annealing (no SMOTE/ADASYN)
- **Constitution v3.4.0** — 18 non-negotiable ML integrity principles (`.specify/memory/constitution.md`)
- **TDD** — boundary tests on real pipeline artifacts; no synthetic fixtures

---

## Project Structure

```
src/
  validate/     featurize/    train_vae/    augment/
  encode/       train_ml/     train_dl/     train_gmm/
  evaluate/     tune/         register/
  config.py     metrics.py
pipelines/kubeflow/   # KFP v2 pipeline
k8s/                  # PVC, Katib experiment YAML
great_expectations/   # GE v1 file context + utils
specs/                # Feature specs, plans, task lists
data/                 # DVC-tracked (not in git)
models/               # DVC-tracked (not in git)
```
