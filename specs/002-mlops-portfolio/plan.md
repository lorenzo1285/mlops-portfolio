# Implementation Plan: MLOps Learning Portfolio — VAE-Based Crash Severity Pipeline

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-26 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-mlops-portfolio/spec.md`

---

## Summary

Build a 10-stage ML pipeline on the CGR crash dataset demonstrating the full MLOps
toolchain. The architectural foundation is a Denoising β-VAE with KL annealing that
learns a compressed latent representation of crash records (unsupervised, no labels).
Both classifiers — XGBoost and a shallow PyTorch MLP — operate on 8-dimensional Z
vectors produced by the frozen encoder. CTGAN augmentation in a dedicated `augment`
stage handles the extreme Fatal class imbalance by generating synthetic Fatal rows in
X-space before encoding. A statistical A/B test (Welch's t-test, N=10 seeds each)
compares the two classifiers on the same held-out Z_test set. Katib searches the β
hyperparameter. Kubeflow Pipelines is the sole orchestrator.

---

## Technical Context

**Language/Version**: Python 3.12, managed with `uv`
**Primary Dependencies**:
- PyTorch 2.x — β-VAE encoder/decoder + MLP classifier
- XGBoost 2.x — multi-class classifier on Z vectors
- DVC 3.x — 10-stage pipeline DAG, artifact versioning
- Great Expectations 1.x — data validation (GE v1 API)
- MLflow 3.x — experiment tracking + model registry + `mlflow.evaluate()`
- Kubeflow Pipelines SDK v2 — 10 `@dsl.component` definitions
- Katib (Kubernetes CRD) — β HPO via `vae_experiment.yaml`
- scikit-learn — `ColumnTransformer` preprocessing pipeline + `compute_class_weight`
- scipy — Welch's t-test + Cohen's d

**Storage**: Local DVC remote (`data/dvc-remote`); MLflow local (`mlruns/`); numpy `.npy`
arrays for Z vectors; `.pth` checkpoint for VAE encoder/decoder; `.pkl` for XGBoost

**Testing**: pytest, red→green→refactor TDD cycle (Constitution XV); boundary tests only

**Target Platform**: Windows 11 + Docker Desktop Kubernetes (local dev); container-portable

**Performance Goals**: Classifier inference < 30 s from registry load; VAE training
convergent within configured epoch budget; full pipeline completable on one machine

**Constraints**: Sequential stage execution (no RAM contention); Docker Desktop
Kubernetes resource limits; single DVC remote (local filesystem)

**Scale/Scope**: 74,309 rows; ~50-80 preprocessed features → 32-dim Z vectors; N=10
seeds per classifier; 5 β values for Katib

---

## Constitution Check

*Constitution v3.3.0 — checked 2026-04-29*

| Gate | Status | Detail |
|---|---|---|
| No post-crash columns as model inputs | **PASS** | `featurize` governs feature selection; VAE trains on preprocessed X (pre-crash only) |
| `samples_per_param_ratio ≥ 3.0` logged by featurize | **PASS** | MLP on Z(32): ~2,307 params; ratio ≈ 22.5 × — easily above 3.0 |
| Preprocessing fit on train split only | **PASS** | `ColumnTransformer` fits on X_train; VAE uses all X (Principle II unsupervised exception) |
| Test set never used during HPO search | **PASS** | Katib trials evaluate fitness on `Z_val` — spec `eout_macro_f1` in tune context means val; `Z_test` reserved for `evaluate` stage only |
| Class imbalance strategy documented | **PASS** | Runtime-computed class weights + CTGAN augmentation on X_train only + encode → Z (Principle III v3.3.0) |
| MLflow tracking present in plan | **PASS** | FR-004–FR-010; experiments: vae, ml, dl, tune |
| Macro F1 > 0.45 AND fatal recall > 0.30 gates | **PASS** | FR-009, SC-005; Principle VI v3.1.0 |
| DVC pipeline stages defined in `dvc.yaml` | **PASS** | FR-002: 10 stages with explicit deps/outs/params |
| GE expectation suite before train stage | **PASS** | validate → ingest → featurize → train_vae |
| KFP pipeline — all 10 stages as `@dsl.component` | **PASS** | FR-015; `pipelines/kubeflow/pipeline.py` |
| Each stage executable as Docker container | **PASS** | FR-014; Principle X |
| No notebooks in pipeline tasks | **PASS** | Principle XI |
| All spec terms in `UBIQUITOUS_LANGUAGE.md` | **ACTION BEFORE TASKS** | New terms (β-VAE, ELBO, LSA, Z_train_augmented, latent_dim, PDO, fatal class) must be added via `/ubiquitous-language` before `/speckit.tasks` |
| Grill-me pass completed and spec updated | **PASS** | Completed 2026-04-26 |

**Design constraint (tune stage)**: Katib trials compute fitness on `Z_val`. The stage
contract for `tune` must explicitly state: `Z_TEST_PATH` is passed through to the final
post-tune `evaluate` call only; the Katib metric collector reads `val_macro_f1` from
stdout, not `test_macro_f1`.

---

## Project Structure

### Documentation (this feature)

```text
specs/002-mlops-portfolio/
├── plan.md              ← this file
├── research.md          ← Phase 0 (updated with VAE decisions 11–15)
├── data-model.md        ← Phase 1 (rewritten for 10-stage VAE pipeline)
├── quickstart.md        ← Phase 1 (updated for 10-stage pipeline)
├── contracts/
│   └── stage-interface.md   ← Phase 1 (added train_vae + encode; updated train_ml/dl/evaluate)
├── checklists/
│   └── requirements.md
└── tasks.md             ← Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
src/
├── config.py               # add VAEConfig, EncodeConfig; update ModelConfig for multi-class
├── metrics.py              # add per-class P/R/F1 matrix helper
├── ingest/
│   ├── run.py
│   └── ingester.py         # ✅ complete
├── validate/
│   ├── run.py
│   └── validator.py        # ✅ complete
├── featurize/
│   ├── run.py
│   ├── featurizer.py       # update: 3-class target encoding (PDO=0, Injury=1, Fatal=2)
│   └── selector.py
├── train_vae/              # NEW stage
│   ├── run.py
│   └── vae_trainer.py      # DVAETrainer: encoder + decoder + ELBO loss + KL annealing + MLflow logging
├── augment/                # NEW stage (parallel with train_vae)
│   ├── run.py
│   └── augmenter.py        # CTGANAugmenter: fit TVAE on Fatal rows, generate synthetic X_train rows
├── encode/                 # NEW stage (depends on train_vae + augment)
│   ├── run.py
│   └── encoder.py          # LatentEncoder: freeze VAE encoder, project X_train_augmented + X_val/test → Z splits
├── train_ml/
│   ├── run.py              # update: XGBoost config
│   └── trainer.py          # REWRITE: XGBoost multi-class on Z vectors (remove PyCaret)
├── train_dl/
│   ├── run.py              # update: Z-vector input, 3-class output
│   ├── trainer.py          # REWRITE: shallow MLP on Z(8→64→3); cross-entropy + class weights
│   └── pyfunc.py           # update: 3-class MLP wrapper
├── evaluate/
│   ├── run.py
│   └── evaluator.py        # update: multi-class gates; per-class matrix; fatal recall gate
├── tune/
│   ├── run.py
│   └── tuner.py            # update: submit vae_experiment.yaml; fitness = val_macro_f1
├── register/
│   ├── run.py
│   └── registrar.py
└── tune/
    └── trial.py            # update: retrain VAE + encode + winner classifier; log val_macro_f1

great_expectations/gx/       # unchanged (GE layer)
k8s/
├── pvc.yaml                 # unchanged
└── katib/
    ├── vae_experiment.yaml  # NEW: β search space [0.5,1.0,2.0,4.0,8.0]
    ├── ml_experiment.yaml   # DELETE or repurpose
    └── dl_experiment.yaml   # DELETE or repurpose

pipelines/kubeflow/
└── pipeline.py              # update: add train_vae + encode components (10 total)

tests/
├── test_ingest.py           # ✅ complete
├── test_validate.py         # ✅ complete
├── test_featurize.py        # update: 3-class target
├── test_train_vae.py        # NEW: ELBO convergence, encoder output shape
├── test_encode.py           # REVISED: Z shape from X_train_augmented; no LSA assertions
├── test_train_ml.py         # REWRITE: XGBoost multi-class on Z
├── test_train_dl.py         # REWRITE: shallow MLP on Z(8), 3-class output, class weights
├── test_evaluate.py         # update: multi-class gates
├── test_tune.py             # update: β search
└── test_register.py
```

**params.yaml additions/changes**:

```yaml
# NEW / UPDATED sections
vae:
  encoder_dims: [256, 128, 64]
  latent_dim: 8               # fixed — not tunable
  beta_start: 0.0             # KL annealing start (prevents posterior collapse)
  beta_max: 0.5               # KL annealing ceiling; tuned by Katib
  warmup_epochs: 15           # epochs to ramp beta_start → beta_max
  dropout_p: 0.15             # neural inpainting corruption rate
  epochs: 200
  patience: 20
  batch_size: 512
  lr: 0.0005
  experiment_name: crash-severity-vae

augment:
  tvae_epochs: 500            # CTGAN/TVAE epochs to fit Fatal distribution
  target_fatal_ratio: 0.05   # augment Fatal class to 5% of X_train_augmented
  random_state: 42

# encode section removed — no LSA parameters needed; encode is a pure projection pass

# MODIFIED sections
model:
  n_classes: 3                # was binary
  macro_f1_threshold: 0.45    # was 0.55
  fatal_recall_threshold: 0.30  # new — was minority_recall_threshold: 0.40
  # remove: class_weight_neg, class_weight_pos (now computed at runtime from train split)

dl:
  input_dim: 8                # Z-vector dimensionality (matches vae.latent_dim)
  hidden_dim: 64
  dropout_p: 0.1
  epochs: 100
  patience: 10
  batch_size: 256
  lr: 0.001
  experiment_name: crash-severity-dl

mlflow:
  experiment_name_ml: crash-severity-ml
  experiment_name_dl: crash-severity-dl
  experiment_name_vae: crash-severity-vae
  experiment_name_tune: crash-severity-tune
  model_name: crash-severity
  tracking_uri: mlruns/
```

---

## Scope of Changes

### New files (create from scratch)

| File | Purpose |
|---|---|
| `src/train_vae/run.py` | Entry point: reads VAEConfig, calls DVAETrainer, logs to MLflow |
| `src/train_vae/vae_trainer.py` | DVAETrainer class: encoder + decoder, ELBO loss, training loop |
| `src/encode/run.py` | Entry point: reads EncodeConfig, calls LatentEncoder, writes Z arrays |
| `src/encode/encoder.py` | LatentEncoder class: freeze encoder, produce Z splits, LSA augmentation |
| `k8s/katib/vae_experiment.yaml` | Katib Experiment CRD — β search space [0.5,1.0,2.0,4.0,8.0] |
| `tests/test_train_vae.py` | Boundary tests for DVAETrainer |
| `tests/test_encode.py` | Boundary tests for LatentEncoder + LSA |

### Modified files (targeted changes)

| File | Change |
|---|---|
| `src/config.py` | Add `VAEConfig`, `EncodeConfig`; update `ModelConfig` (n_classes=3, new thresholds) |
| `src/metrics.py` | Add `per_class_matrix()` helper returning JSON dict |
| `src/featurize/featurizer.py` | Target encoding: `PDO→0`, `Injury→1`, `Fatal→2` from CRASHSEVER |
| `src/train_ml/trainer.py` | Replace PyCaret with `xgboost.XGBClassifier` multi-class on Z input |
| `src/train_ml/run.py` | Update config: reads Z paths, writes pkl, no PyCaret setup() |
| `src/train_dl/trainer.py` | Remove EvoTorch NAS; MLP: `Linear(32,64)→ReLU→Dropout→Linear(64,3)` |
| `src/train_dl/pyfunc.py` | Update wrapper for 3-class output |
| `src/train_dl/run.py` | Update config: Z input, 3-class loss (CrossEntropyLoss) |
| `src/evaluate/evaluator.py` | Update gates (F1>0.45, fatal recall>0.30); add per-class matrix |
| `src/evaluate/run.py` | Update config: new gate thresholds from params |
| `src/tune/tuner.py` | Submit `vae_experiment.yaml`; read `val_macro_f1` from Katib result |
| `src/tune/trial.py` | Retrain VAE with β + encode + winner classifier; stdout: `val_macro_f1=<value>` |
| `params.yaml` | Add vae.*, encode.*; update model.* for multi-class; update dl.* |
| `dvc.yaml` | Add `train_vae` + `encode` stages; update `train_ml`/`train_dl` deps |
| `pipelines/kubeflow/pipeline.py` | Add `train_vae_stage` + `encode_stage` components |
| `CLAUDE.md` | Update architecture table (2 new stages), pipeline description, DL section |
| `tests/test_train_ml.py` | Rewrite for XGBoost multi-class on Z input |
| `tests/test_train_dl.py` | Rewrite for MLP on Z(32), 3-class output, no NAS |
| `tests/test_evaluate.py` | Update for multi-class gates |
| `tests/test_tune.py` | Update for β search, val_macro_f1 fitness |

### Deleted files

| File | Reason |
|---|---|
| `k8s/katib/ml_experiment.yaml` | Replaced by `vae_experiment.yaml` (β HPO, not ML HPO) |
| `k8s/katib/dl_experiment.yaml` | Replaced by `vae_experiment.yaml` |
| EvoTorch NAS code in `src/train_dl/` | NAS removed from architecture |

### DVC stage additions

```yaml
# dvc.yaml additions

train_vae:
  cmd: python -m src.train_vae.run
  deps:
    - src/train_vae/
    - data/processed/X_train.npy
    - data/processed/X_val.npy
    - data/processed/X_test.npy
  params:
    - params.yaml:
        - vae
        - mlflow
  outs:
    - models/vae_encoder.pth
    - models/vae_decoder.pth

encode:
  cmd: python -m src.encode.run
  deps:
    - src/encode/
    - models/vae_encoder.pth
    - data/processed/X_train.npy
    - data/processed/y_train.npy
    - data/processed/X_val.npy
    - data/processed/X_test.npy
  params:
    - params.yaml:
        - encode
        - vae.latent_dim
  outs:
    - data/processed/Z_train_augmented.npy
    - data/processed/Z_val.npy
    - data/processed/Z_test.npy
    - data/processed/y_train_augmented.npy
```

---

## Complexity Tracking

No constitution violations. No complexity justifications required.

---

## Phase 0 Research Summary

See `research.md` — Decisions 11–15 added:
- Decision 11: Denoising β-VAE architecture (encoder dims, ELBO loss, neural inpainting)
- Decision 12: LSA (Latent-Space Augmentation) approach
- Decision 13: XGBoost multi-class (replaces PyCaret)
- Decision 14: Multi-class target encoding
- Decision 15: Updated `params.yaml` structure for VAE pipeline

---

## Phase 1 Design Summary

- `data-model.md` — fully rewritten for 10-stage VAE pipeline
- `contracts/stage-interface.md` — added `train_vae` + `encode` stage contracts; updated `train_ml`, `train_dl`, `evaluate` for new architecture
- `quickstart.md` — updated for 10-stage pipeline, new artifact paths
- Agent context: run `.specify/scripts/powershell/update-agent-context.ps1 -AgentType claude`

---

## Pre-Tasks Actions Required

Before `/speckit.tasks`:

1. **Update `UBIQUITOUS_LANGUAGE.md`** via `/ubiquitous-language` with new terms:
   - β-VAE / Denoising β-VAE / DVAE
   - ELBO (Evidence Lower Bound)
   - Latent Space / Latent Vector (Z)
   - Latent-Space Augmentation (LSA)
   - Z_train_augmented / Z_val / Z_test
   - latent_dim
   - PDO (Property Damage Only)
   - Fatal class
   - Neural inpainting

2. **Verify no glossary ambiguities** before proceeding.
