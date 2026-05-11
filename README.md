# MLOps Portfolio — Crash Severity Classification

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![DVC](https://img.shields.io/badge/data%20versioning-DVC-945DD6)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/experiment%20tracking-MLflow-0194E2)](https://mlflow.org)
[![Great Expectations](https://img.shields.io/badge/data%20quality-Great%20Expectations-FF6B35)](https://greatexpectations.io)
[![Kubeflow](https://img.shields.io/badge/orchestration-Kubeflow%20Pipelines-326CE5)](https://kubeflow.org)
[![Optuna](https://img.shields.io/badge/HPO-Optuna-2D7DD2)](https://optuna.org)
[![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9)](https://docs.astral.sh/uv)

A **production-style MLOps pipeline** built on the City of Grand Rapids crash dataset (74,309 records · 142 raw features). The task is 3-class crash severity classification — **PDO** (property damage only), **Injury**, and **Fatal** — with severe class imbalance (~78 / ~20 / ~2 %). The project demonstrates the full ML lifecycle: data contract → feature engineering → generative augmentation → latent-space compression → multi-model A/B testing → HPO → champion registration → drift monitoring.

---

## Results

| Model | Macro F1 (10-seed mean) | 95 % CI | Fatal Recall | A/B result |
|-------|------------------------|---------|--------------|------------|
| **MLP (champion)** | **0.315** | [0.301, 0.329] | 0.510 | winner |
| XGBoost | 0.238 | — | 0.762 | eliminated |
| GMM | 0.243 | — | 0.667 | eliminated |

- Champion registered at `models:/crash-severity@champion` (v3, MLflow Registry)
- All 3 pairwise comparisons statistically significant — Bonferroni-corrected Welch's t-test (α/3 per pair, 10 seeds each)
- Constitutional gates passed: macro F1 > 0.30 ✅ · fatal recall > 0.50 ✅
- Latent space drift check (MMD): 0.062 vs threshold 0.825 — no drift ✅

> **Why does XGBoost have higher fatal recall but lose the A/B test?**
> Macro F1 is the primary metric (equal weight to all three classes). XGBoost's fatal recall advantage comes at the cost of collapsing Injury predictions — macro F1 penalises that. The constitution accepts a PDO sacrifice over fatal misclassification, but not an Injury erasure.

---

## Pipeline Architecture

```
validate → ingest → featurize → ┬─ train_vae ─┐
                                 └─ augment   ─┘
                                               ▼
                                            encode → ┬─ train_ml  ─┐
                                                      ├─ train_dl  ─┤
                                                      └─ train_gmm ─┘
                                                                     ▼
                                                                 evaluate → tune → register
```

12 DVC stages · fully reproducible from raw CSV · parallelism where safe (VAE+augment; ML+DL+GMM training)

| # | Stage | What it does |
|---|-------|-------------|
| 1 | `validate` | GE v1 suite checks schema, ranges, null rates on raw CSV; halts on failure |
| 2 | `ingest` | Gates on `.validation_passed` sentinel; copies CSV to `data/processed/` |
| 3 | `featurize` | 60/20/20 split; cyclical encoding (HOUR, MONTH, DAYOFWEEK); 4-group ColumnTransformer; sample-complexity gate (N/p ≥ 3) |
| 4 | `train_vae` | Denoising β-VAE with KL annealing (β ramp 0→0.153, 29 warm-up epochs); 16-dim latent space; trained unsupervised on all rows |
| 5 | `augment` | CTGAN (TVAE) on Fatal-class X_train rows → Fatal fraction 2% → 15%; X_val/X_test never touched |
| 6 | `encode` | Frozen VAE encoder (μ path) projects all splits to Z-vectors; builds MMD drift reference from unaugmented X_train |
| 7 | `train_ml` | XGBoost on Z_train_augmented; 10 seeds; class weights; MLflow autolog disabled |
| 8 | `train_dl` | Shallow MLP (Z→64→3) with focal loss (γ=2.73); 10 seeds; early stopping on val loss |
| 9 | `train_gmm` | Per-class GMM (full covariance, 2 Fatal components); fatal prior boost; MAP prediction |
| 10 | `evaluate` | 3-way Welch's t-test A/B/C; Bonferroni correction; asserts constitutional gates |
| 11 | `tune` | Optuna HPO: β, fatal ratio, focal γ; writes best params to `params.yaml`; invalidates downstream DVC cache |
| 12 | `register` | Promotes champion to MLflow registry; writes `models/registry_receipt.json` |

---

## Key Design Decisions

### Latent space as a feature hub

Raw tabular features (26 columns, mixed types) are compressed through a **denoising β-VAE** to a 16-dimensional latent space before any classifier sees the data. This does three things: it forces the representation to disentangle structure from noise (denoising objective), regularises the geometry (KL term), and gives all three classifiers (XGBoost, MLP, GMM) an identical, well-scaled input regardless of their native assumptions.

### Three-mechanism imbalance handling — nothing more

Fatal crashes are ~2 % of the dataset. The pipeline uses exactly three mechanisms:

1. **Runtime class weights** — inverse-frequency weights fed to XGBoost and MLP loss functions
2. **CTGAN augmentation** — synthetic Fatal rows generated by TVAE and appended to X_train only (target 15% Fatal fraction); X_val and X_test are never touched
3. **KL annealing** — prevents posterior collapse in the VAE (β ramps from 0 → 0.153 over 29 epochs), which would otherwise squash the rare Fatal representation into the majority cluster

No SMOTE, ADASYN, or interpolation. The constitution prohibits them (principle III).

### Rigorous A/B testing over point estimates

Every model is trained on 10 different random seeds. The A/B evaluation uses **Welch's t-test** (unequal variance) on the 10-seed F1 distributions, with **Bonferroni correction** (α/3 per pair) for 3-way comparison. Selecting a winner on a single training run is not acceptable.

### Constitution v3.4.0 — 18 ML integrity principles

The project is governed by a written constitution (`.specify/memory/constitution.md`). Key principles:

- **I** — No post-crash columns as model inputs (leakage prevention)
- **II** — Val set used only for HPO/early-stopping; test set strictly reserved for final A/B evaluation
- **IV** — Sample-complexity gate: featurize halts if N_train/n_params < 3
- **VI** — Macro F1 is the primary metric; gates: F1 > 0.30, fatal recall > 0.50
- **XIV** — Deep modules: constructor + 1 public method; `run.py` handles I/O only; business logic is never in entry points
- **XV** — TDD: red→green→refactor; boundary tests on real pipeline artifacts
- **XVIII** — Real data fixtures only in tests; no `np.random.randn` phantoms

### Drift monitoring in latent space

After encoding, the pipeline computes an **MMD (Maximum Mean Discrepancy)** test between the latent representation of new data and a reference distribution built from the unaugmented training set. The reference is stored as a DVC-tracked artifact (`models/drift_reference.npz`). Drift is advisory — it never halts the pipeline — but it surfaces as an MLflow metric and a `docs/drift_report.json` artifact for inspection.

---

## Tech Stack

| Category | Tool | Why |
|----------|------|-----|
| Pipeline & reproducibility | **DVC** | Content-addressed artifact cache; stage DAG with deps/outs/params; `dvc repro` is the single source of truth for pipeline execution |
| Experiment tracking & registry | **MLflow** | Per-seed metric logging; model registry with alias (`@champion`); explicit logging only (autolog disabled) |
| Data quality | **Great Expectations v1** | Column contracts defined in `params.yaml`; suite regenerated programmatically; Data Docs rendered on every run |
| Orchestration | **Kubeflow Pipelines v2** | Each DVC stage is a KFP component; pipeline compiled to YAML; runs on Docker Desktop Kubernetes with shared hostPath PVC |
| HPO | **Optuna** | Median pruner; Welch's t-test score function; search space: β ∈ [0.05, 2.0], fatal ratio ∈ {0.10, 0.15, 0.20}, focal γ ∈ [1.0, 4.0] |
| Generative augmentation | **CTGAN / TVAE (SDV)** | Mode-specific synthetic Fatal rows; evaluated on downstream classifier gain, not standalone fidelity |
| Deep learning | **PyTorch** | β-VAE (encoder 256→128→64→16); MLP classifier (16→64→3); focal loss |
| Gradient boosting | **XGBoost** | Z-space classifier; scale_pos_weight via class weights |
| Probabilistic | **scikit-learn GMM** | Per-class full-covariance mixture; MAP prediction with fatal prior boost |
| Dependency management | **uv** | Deterministic lockfile; sub-second environment resolution |

---

## Project Structure

```
src/
├── validate/       # GE suite orchestration
├── ingest/         # Raw CSV gating + copy
├── featurize/      # ColumnTransformer + FeatureSelector
├── train_vae/      # DenoisingBetaVAE + DVAETrainer
├── augment/        # CTGANAugmenter
├── encode/         # LatentEncoder + drift reference build
├── train_ml/       # MLTrainer (XGBoost)
├── train_dl/       # DLTrainer (MLP, focal loss)
├── train_gmm/      # GMMTrainer (per-class GMM)
├── evaluate/       # ABEvaluator (3-way Welch's t-test)
├── tune/           # HyperparamTuner (Optuna)
├── register/       # ModelRegistrar (MLflow registry)
├── drift/          # DriftDetector (MMD + RBF kernel)
├── config.py       # Typed dataclasses for all params
└── metrics.py      # Shared eval helpers

pipelines/kubeflow/ # KFP v2 pipeline definition
k8s/                # PVC manifest, Katib experiment YAML
great_expectations/ # GE v1 file context + utility layer
specs/              # Feature specs, plans, task lists, constitution
data/               # DVC-tracked (not in git)
models/             # DVC-tracked (not in git)
docs/               # Pipeline output reports (drift, evaluation, A/B)
```

Each stage follows the same pattern: `<stage>/run.py` (thin entry point — reads env vars, opens files, calls MLflow) + `<stage>/<module>.py` (business-logic class with `constructor + 1 public method`). No file I/O or env var reads inside business-logic classes.

---

## Quick Start

```bash
# 1. Install dependencies (uv required — https://docs.astral.sh/uv/getting-started/installation)
uv sync

# 2. Restore DVC-tracked artifacts from remote
uv run dvc pull

# 3. Run the full 12-stage pipeline
uv run dvc repro

# 4. Run a single stage
uv run dvc repro encode

# 5. Check what DVC would re-run (dry run)
uv run dvc status

# 6. Launch MLflow experiment tracking UI
uv run mlflow ui   # → http://localhost:5000

# 7. Run tests
uv run python -m pytest tests/ -v

# 8. Compile KFP pipeline (requires kubectl + KFP SDK)
python pipelines/kubeflow/pipeline.py
```

---

## Dataset

**CGR Crash Data** — City of Grand Rapids Open Data Portal  
74,309 crash records · 142 raw features · 26 features selected  
Target: `CRASHSEVER` (0 = PDO, 1 = Injury, 2 = Fatal)  
Class distribution: ~78 % PDO · ~20 % Injury · ~2 % Fatal  
Split: 60 % train / 20 % val / 20 % test (stratified, `random_state=42`)

Forbidden features (post-crash leakage — constitution I): `NUMOFKILL`, `NUMOFINJ`, `NUMOFUNINJ`

---

## Reproducing from scratch

```bash
# Delete all cached artifacts
rm -rf data/processed/ models/ docs/

# Pull raw data only
uv run dvc pull data/raw/CGR_Crash_Data.csv

# Re-run full pipeline end-to-end (~20–40 min depending on hardware)
uv run dvc repro
```

DVC will re-execute only stages whose inputs have changed. The pipeline is fully deterministic given fixed `random_state` values in `params.yaml`.
