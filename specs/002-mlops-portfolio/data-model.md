# Data Model: MLOps Learning Portfolio — VAE-Based Crash Severity Pipeline

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-22 | **Updated**: 2026-04-28

---

## Entities

### 1. Pipeline Stage

A named, self-contained executable unit. Each stage has a deterministic relationship
between inputs (deps) and outputs (outs). Running the same stage twice with the same
deps and params MUST produce byte-identical outputs (idempotency required for DVC caching).

| Field | Type | Description |
|---|---|---|
| `name` | string | Stage identifier: `validate`, `ingest`, `featurize`, `train_vae`, `encode`, `train_ml`, `train_dl`, `evaluate`, `tune`, `register` |
| `cmd` | string | Shell command to execute (e.g., `python -m src.train_vae.run`) |
| `deps` | list[path] | Input files/dirs this stage reads; change triggers re-run |
| `outs` | list[path] | Output files/dirs this stage writes; DVC-tracked |
| `params` | list[key] | Keys from `params.yaml` this stage reads; change triggers re-run |

**Stage DAG** (`train_ml` and `train_dl` are parallel — both depend on `encode`):

```
validate  ← runs on data/raw/CGR_Crash_Data.csv (GE gate before any data is committed)
  └── ingest  ← only runs after validate sentinel exists
        └── featurize
              ├── train_vae  ← unsupervised; trains on X_train+X_val+X_test (no Y); KL annealing
              └── augment    ← CTGAN on X_train Fatal rows → X_train_augmented (parallel with train_vae)
                    └── (both feed) → encode  ← frozen encoder projects X_train_augmented → Z splits
                                          ├── train_ml ─┐
                                          └── train_dl ─┴── evaluate
                                                              └── tune
                                                                    └── register
```

`train_ml` and `train_dl` have no dependency on each other and MAY run concurrently
(`dvc repro --run-cache` or as parallel KFP steps).

---

### 2. DVC-Tracked Artifact

A file or directory whose content is versioned by DVC. Pointer committed to git;
content stored in DVC cache and remote.

| Field | Type | Description |
|---|---|---|
| `path` | string | Relative path from repo root (e.g., `data/raw/CGR_Crash_Data.csv`) |
| `md5` | string | Content hash stored in `.dvc` pointer file |
| `size` | int | File size in bytes |
| `remote` | string | DVC remote name (default: `local`) |

**Tracked artifacts in this pipeline**:

| Path | Tracked by | Stage |
|---|---|---|
| `data/raw/CGR_Crash_Data.csv` | `dvc add` (manual) | — (input) |
| `data/processed/.validation_passed` | `dvc.yaml outs` | validate (sentinel — gates ingest) |
| `data/processed/raw.csv` | `dvc.yaml outs` | ingest (only after sentinel exists) |
| `data/processed/X_train.npy` | `dvc.yaml outs` | featurize |
| `data/processed/X_val.npy` | `dvc.yaml outs` | featurize |
| `data/processed/X_test.npy` | `dvc.yaml outs` | featurize |
| `data/processed/y_train.npy` | `dvc.yaml outs` | featurize |
| `data/processed/y_val.npy` | `dvc.yaml outs` | featurize |
| `data/processed/y_test.npy` | `dvc.yaml outs` | featurize |
| `models/preprocessing_pipeline.joblib` | `dvc.yaml outs` | featurize |
| `models/vae_encoder.pth` | `dvc.yaml outs` | train_vae |
| `models/vae_decoder.pth` | `dvc.yaml outs` | train_vae |
| `data/processed/X_train_augmented.npy` | `dvc.yaml outs` | augment |
| `data/processed/y_train_augmented.npy` | `dvc.yaml outs` | augment |
| `data/processed/Z_train_augmented.npy` | `dvc.yaml outs` | encode |
| `data/processed/Z_val.npy` | `dvc.yaml outs` | encode |
| `data/processed/Z_test.npy` | `dvc.yaml outs` | encode |
| `models/best_ml_model.pkl` | `dvc.yaml outs` | train_ml |
| `models/mlp_model.pth` | `dvc.yaml outs` | train_dl |
| `docs/evaluation_report.json` | `dvc.yaml outs` | evaluate |
| `docs/ab_test_comparison.json` | `dvc.yaml outs` | evaluate |
| `models/registry_receipt.json` | `dvc.yaml outs` | register |

---

### 3. Expectation Suite

A versioned, committed set of data quality rules for the crash dataset. Defined once,
run at every pipeline execution against `data/raw/CGR_Crash_Data.csv` **before ingest**.
Results determine whether the pipeline proceeds — if any expectation fails, the sentinel
is not written and ingest never runs.

The suite is built programmatically by `GEContextBuilder` from `params.yaml`
`validation.columns` — it is never edited by hand. The JSON is committed to
`great_expectations/gx/expectations/crash_data_suite.json` as a DVC-tracked artefact.
Execution follows the three-class workflow: `GEContextBuilder.build()` creates the suite
→ `GEManager` binds the asset+suite and registers the `BatchDefinition` in context
→ `GECheckpointRunner.run(df)` fires a GE v1 `Checkpoint` with `UpdateDataDocsAction`
(renders HTML) and `StoreValidationResultAction` (audit trail).

| Field | Type | Description |
|---|---|---|
| `name` | string | `crash_data_suite` |
| `expectations` | list[Expectation] | Individual quality rules (see below) |
| `meta` | dict | GE metadata (asset, datasource) |

**Expectation types generated**:

| GE Class | Trigger | Column examples |
|---|---|---|
| `ExpectColumnValuesToNotBeNull` | always (all columns) | HOUR, CRASHSEVER, DRIVER1AGE |
| `ExpectColumnValuesToBeBetween` | when `min` or `max` set | HOUR (0–23), SPEEDLIMIT (5–70), DRIVER1AGE (14–100) |
| `ExpectColumnValuesToBeInSet` | when `allowed_values` set | DAYOFWEEK, WEATHER, SURFCOND, CRASHTYPE |

---

### 4. VAE Model

The trained Denoising β-VAE. Encoder and decoder weights saved as separate `.pth`
artifacts. The encoder alone is used by the `encode` stage; both are needed for Katib
trials that retrain the VAE from scratch.

| Field | Type | Description |
|---|---|---|
| `encoder_path` | string | `models/vae_encoder.pth` |
| `decoder_path` | string | `models/vae_decoder.pth` |
| `latent_dim` | int | 8 (fixed; not tunable; set in `params.yaml vae.latent_dim`) |
| `encoder_dims` | list[int] | `[256, 128, 64]` (configurable via `params.yaml vae.encoder_dims`) |
| `beta_start` | float | KL annealing start value (0.0); ramps to `beta_max` over `warmup_epochs` |
| `beta_max` | float | KL annealing max value (0.5); tuned by Katib |
| `warmup_epochs` | int | Epochs over which β ramps from `beta_start` to `beta_max` (15) |
| `best_epoch` | int | Epoch at which lowest validation ELBO was achieved |
| `vae_elbo` | float | Best validation ELBO (logged to MLflow) |
| `kl_beta` | float (per epoch) | Current β value logged per epoch — confirms annealing schedule |
| `mlflow_run_id` | string | MLflow run ID in `crash-severity-vae` experiment |

---

### 5. Latent Vector (Z)

A 32-dimensional compressed representation of one crash record produced by passing
preprocessed features through the frozen VAE encoder. The input to both classifiers.

| Field | Type | Description |
|---|---|---|
| `Z_train_augmented` | ndarray (N_aug × 8) | Training Z vectors — encoded from CTGAN-augmented X_train |
| `Z_val` | ndarray (N_val × 8) | Validation Z vectors — never augmented |
| `Z_test` | ndarray (N_test × 8) | Test Z vectors — never augmented |
| `y_train_augmented` | ndarray (N_aug,) | Labels aligned with Z_train_augmented (0/1/2); passed through from augment stage |

---

### 6. CTGAN Augmentation

Synthetic Fatal-class feature rows generated by fitting CTGAN/TVAE on real Fatal rows
of `X_train`. Applied only to `X_train` in the `augment` stage; `X_val` and `X_test`
are never touched. The augmented `X_train_augmented` is then encoded through the frozen
VAE encoder to produce `Z_train_augmented`.

| Field | Type | Description |
|---|---|---|
| `n_real_fatal` | int | Count of real Fatal rows in y_train before augmentation |
| `n_synthetic` | int | Count of CTGAN-generated Fatal rows added |
| `target_fatal_ratio` | float | Target fatal fraction of X_train_augmented (default 0.05) |
| `X_train_augmented` | ndarray (N_aug × n_features) | Original X_train + synthetic Fatal rows |
| `y_train_augmented` | ndarray (N_aug,) | Labels: original y_train + 2 for each synthetic row |

---

### 7a. ML Experiment Run (XGBoost)

Produced by the `train_ml` stage. MLflow experiment: `crash-severity-ml`.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | MLflow UUID |
| `experiment_name` | string | `crash-severity-ml` |
| `params` | dict | XGBoost hyperparameters (logged via `mlflow.log_params`) |
| `metrics.ein_macro_f1` | float | In-sample macro F1 (Z_train_augmented) |
| `metrics.eout_macro_f1` | float | Test-set macro F1 (Z_test) |
| `metrics.eout_fatal_recall` | float | Test-set recall on Fatal class (Z_test) |
| `metrics.generalisation_gap` | float | `eout_macro_f1 − ein_macro_f1` |
| `artifact.per_class_matrix.json` | dict | Per-class P/R/F1 for PDO / Injury / Fatal |
| `artifact_uri` | string | Path to `.pkl` model |
| `tags.seed` | string | Random seed for this run |
| `tags.model_type` | string | `xgboost` |
| `tags.orchestrator` | string | `dvc` / `kubeflow` |

### 7b. DL Experiment Run (PyTorch MLP on Z)

Produced by the `train_dl` stage. MLflow experiment: `crash-severity-dl`.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | MLflow UUID |
| `experiment_name` | string | `crash-severity-dl` |
| `metrics.ein_loss` | float (per epoch) | Training CrossEntropy loss |
| `metrics.eout_loss` | float (per epoch) | Validation CrossEntropy loss |
| `metrics.gap_f1` | float (per epoch) | `eout_macro_f1 − ein_macro_f1` |
| `metrics.eout_macro_f1` | float | Final test-set macro F1 |
| `metrics.eout_fatal_recall` | float | Final test-set recall on Fatal class |
| `artifact.per_class_matrix.json` | dict | Per-class P/R/F1 for PDO / Injury / Fatal |
| `artifact_uri` | string | Path to `.pth` checkpoint |
| `tags.seed` | string | Random seed for this run |
| `tags.model_type` | string | `pytorch-mlp` |
| `tags.architecture` | string | `8-64-3-dropout0.1` |
| `tags.orchestrator` | string | `dvc` / `kubeflow` |

### 7c. VAE Training Run

Produced by the `train_vae` stage. MLflow experiment: `crash-severity-vae`.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | MLflow UUID |
| `experiment_name` | string | `crash-severity-vae` |
| `metrics.vae_elbo` | float (per epoch) | ELBO = reconstruction_loss + β × KL |
| `metrics.vae_reconstruction_loss` | float (per epoch) | MSE reconstruction loss |
| `metrics.vae_kl_loss` | float (per epoch) | KL divergence term |
| `params.beta` | float | β value used for this run |
| `params.encoder_dims` | string | e.g., `"[256, 128, 64]"` |
| `params.latent_dim` | int | 32 |
| `params.best_epoch` | int | Epoch of best validation ELBO |
| `tags.orchestrator` | string | `dvc` / `kubeflow` |

### 7d. A/B Test Result

Produced by the `evaluate` stage. Stored as `docs/ab_test_comparison.json`.
Matches the `EvaluationResult` dataclass in `src/evaluate/evaluator.py`.

| Field | Type | Description |
|---|---|---|
| `winner` | string | `"ml"` or `"dl"` |
| `p_value` | float | Welch's t-test p-value |
| `cohens_d` | float | Effect size |
| `ml_mean_f1` | float | Mean eout_macro_f1 across ML seeds |
| `dl_mean_f1` | float | Mean eout_macro_f1 across DL seeds |
| `ml_ci_low` | float | 95% CI lower bound — ML |
| `ml_ci_high` | float | 95% CI upper bound — ML |
| `dl_ci_low` | float | 95% CI lower bound — DL |
| `dl_ci_high` | float | 95% CI upper bound — DL |
| `significant` | bool | `True` if p_value < alpha |
| `gates_passed` | bool | `True` if winner macro F1 > 0.45 AND fatal recall > 0.30 |
| `winner_ml_per_class` | dict | Per-class P/R/F1 matrix for ML winner |
| `winner_dl_per_class` | dict | Per-class P/R/F1 matrix for DL winner |

---

### 8. Registered Model

A promoted model artifact in the MLflow Model Registry. Independent of the training
run ID and local filesystem path. Matches the `RegistryReceipt` dataclass in
`src/register/registrar.py`.

| Field | Type | Description |
|---|---|---|
| `model_name` | string | `crash-severity` |
| `version` | string | Auto-incremented on each registration |
| `alias` | string | `champion` (replaces deprecated `Production` stage in MLflow 3.x) |
| `run_id` | string | MLflow run ID of the promoted artifact |
| `winner` | string | `"ml"` or `"dl"` — which model family was promoted |

**Load pattern**: `mlflow.pyfunc.load_model("models:/crash-severity@champion")`

---

### 9. KFP Pipeline

The Kubeflow Pipelines representation. Ten containerised components with the same
logical dependency structure as the DVC pipeline DAG.

| Field | Type | Description |
|---|---|---|
| `pipeline_name` | string | `crash-severity-pipeline` |
| `components` | list[Component] | One `@dsl.component` per stage (10 total) |
| `base_image` | string | `mlops-portfolio:latest` (local Docker image) |
| `pipeline_yaml` | string | `pipelines/kubeflow/pipeline.yaml` (compiled output) |

**Component interface** (all ten components share this pattern):

| Parameter | Direction | Type | Description |
|---|---|---|---|
| `input_path` | input | str | Path to stage's primary input artifact |
| `output_path` | input | str | Path where stage writes its output |
| `params_path` | input | str | Path to `params.yaml` |
| `mlflow_uri` | input | str | MLflow tracking URI (train_vae, train_ml, train_dl, evaluate, register only) |

---

## State Transitions

### Pipeline Execution State

```
PENDING → RUNNING → SUCCESS
                 ↘ FAILED (validate failure halts all downstream stages)
                 ↘ FAILED (encode: < min_fatal_samples → halt with exit 1)
                 ↘ FAILED (evaluate: gates_passed=False → register blocked)
```

### Model Lifecycle

```
[VAE Training Run] → encoder/decoder artifacts saved → used by encode stage
[Classifier Run]   → registered → version=N, alias=None
                               → alias="champion" (promoted by register stage)
                               → alias="champion" overwritten (next better model)
```

### Augmentation State

```
X_train (real only) → CTGAN augment stage → X_train_augmented (real + synthetic Fatal rows)
X_val   (unchanged, never augmented — constitution III v3.3.0)
X_test  (unchanged, never augmented — constitution III v3.3.0)

X_train_augmented → encode (frozen VAE encoder) → Z_train_augmented
X_val             → encode                       → Z_val   (unchanged row count)
X_test            → encode                       → Z_test  (unchanged row count)
```
