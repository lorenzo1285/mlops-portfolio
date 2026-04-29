# Stage Interface Contracts

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-22 | **Updated**: 2026-04-28

Each pipeline stage MUST honour this interface so it runs identically under DVC
and Kubeflow. All paths are passed via environment variables — no hardcoded
paths anywhere in `src/`.

---

## Environment Variable Interface (all stages)

Every stage script reads configuration exclusively from environment variables.
Default values are loaded from `params.yaml` if an env var is not set, using:

```python
import yaml, os
params = yaml.safe_load(open(os.getenv("PARAMS_PATH", "params.yaml")))
```

---

## Stage Contracts

### Stage: `validate`

**Script**: `src/validate/run.py`
**Purpose**: Run GE expectation suite; write Data Docs; write sentinel; exit non-zero on failure.

| Env Var | Default | Description |
|---|---|---|
| `INPUT_PATH` | `data/raw/CGR_Crash_Data.csv` | CSV to validate |
| `SENTINEL_PATH` | `data/processed/.validation_passed` | Sentinel file written on success |
| `DATA_DOCS_PATH` | `great_expectations/gx/uncommitted/data_docs/` | Output HTML dir |
| `PARAMS_PATH` | `params.yaml` | Parameters file (reads `great_expectations.*`) |
| `GE_ROOT` | `great_expectations/gx/` | GE context root directory |

**Outputs**: `SENTINEL_PATH` (empty file), `DATA_DOCS_PATH/` (HTML Data Docs)
**Exit codes**: `0` = all expectations pass, `1` = one or more expectations fail

**Contract**: A non-zero exit MUST propagate as a pipeline failure. Ingest is gated on
`SENTINEL_PATH` existing — if validate fails, ingest never runs.

---

### Stage: `ingest`

**Script**: `src/ingest/run.py`
**Purpose**: Copy raw CSV to processed dir, validate it is readable, log row count.

| Env Var | Default | Description |
|---|---|---|
| `INPUT_PATH` | `data/raw/CGR_Crash_Data.csv` | Source raw CSV |
| `OUTPUT_PATH` | `data/processed/raw.csv` | Destination copy |
| `SENTINEL_PATH` | `data/processed/.validation_passed` | Must exist (validate gate) |
| `PARAMS_PATH` | `params.yaml` | Parameters file |

**Outputs**: `OUTPUT_PATH` (CSV, same schema as input)
**Exit codes**: `0` = success, `1` = input file not found, unreadable, or sentinel missing

---

### Stage: `featurize`

**Script**: `src/featurize/run.py`
**Purpose**: Select pre-crash features, encode CRASHSEVER to 3-class int target,
3-way split (70/15/15), fit+save pipeline.

| Env Var | Default | Description |
|---|---|---|
| `INPUT_PATH` | `data/processed/raw.csv` | Validated CSV |
| `OUTPUT_DIR` | `data/processed/` | Dir for numpy arrays |
| `PIPELINE_PATH` | `models/preprocessing_pipeline.joblib` | Fitted pipeline output |
| `PARAMS_PATH` | `params.yaml` | Reads `data.*`, `features.*`, `feature_selection.*` |

**Outputs**:
- `OUTPUT_DIR/X_train.npy`, `X_val.npy`, `X_test.npy`
- `OUTPUT_DIR/y_train.npy`, `y_val.npy`, `y_test.npy` (int arrays: 0=PDO, 1=Injury, 2=Fatal)
- `PIPELINE_PATH` (fitted joblib pipeline)

**Exit codes**: `0` = success, `1` = input not found or < 5% rows dropped threshold exceeded

---

### Stage: `train_vae`

**Script**: `src/train_vae/run.py`
**Purpose**: Train Denoising β-VAE on full feature matrix (X_train + X_val + X_test,
no labels). Log ELBO per epoch. Save encoder and decoder weights.

| Env Var | Default | Description |
|---|---|---|
| `TRAIN_X_PATH` | `data/processed/X_train.npy` | Training features |
| `VAL_X_PATH` | `data/processed/X_val.npy` | Validation features |
| `TEST_X_PATH` | `data/processed/X_test.npy` | Test features |
| `ENCODER_OUTPUT_PATH` | `models/vae_encoder.pth` | Saved encoder weights |
| `DECODER_OUTPUT_PATH` | `models/vae_decoder.pth` | Saved decoder weights |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads `vae.*`, `mlflow.experiment_name_vae` |

**Outputs**: `ENCODER_OUTPUT_PATH`, `DECODER_OUTPUT_PATH`

**MLflow side-effects**: Single run in `crash-severity-vae` with per-epoch
`vae_elbo`, `vae_reconstruction_loss`, `vae_kl_loss` logged as `step=epoch`.
Best checkpoint (lowest val ELBO) saved at `best_epoch`.

**Exit codes**: `0` = training complete (best checkpoint saved), `1` = training error

**Note**: VAE trains on all X (no Y labels). This is the constitutionally permitted
unsupervised pre-training exception (Principle II). The test set cannot be contaminated
here because Y is never provided. KL annealing applied: `beta_t` ramps from
`vae.beta_start=0.0` to `vae.beta_max` over `vae.warmup_epochs` — prevents posterior
collapse. Metric `kl_beta` logged per epoch.

---

### Stage: `augment`

**Script**: `src/augment/run.py`
**Purpose**: Generate synthetic Fatal-class training rows using CTGAN/TVAE fitted on
real Fatal rows of `X_train`. Produces augmented training arrays for downstream
classifiers. `X_val` and `X_test` are NEVER augmented (constitution III v3.3.0).
**Parallel with**: `train_vae` (both depend only on featurize outputs)

| Env Var | Default | Description |
|---|---|---|
| `X_TRAIN_PATH` | `data/processed/X_train.npy` | Original training features |
| `Y_TRAIN_PATH` | `data/processed/y_train.npy` | Original training labels |
| `X_AUG_OUTPUT` | `data/processed/X_train_augmented.npy` | Augmented training features |
| `Y_AUG_OUTPUT` | `data/processed/y_train_augmented.npy` | Augmented training labels |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads `augment.*`, `mlflow.tracking_uri` |

**Outputs**:
- `X_AUG_OUTPUT` (shape: N_augmented × n_features — original + synthetic Fatal rows)
- `Y_AUG_OUTPUT` (shape: N_augmented — 0/1/2 int labels)

**Exit codes**: `0` = success, `1` = fewer than 10 real Fatal rows in `y_train`
(CTGAN requires a minimum to fit a meaningful Fatal distribution)

**MLflow side-effects**: Logs `n_real_fatal`, `n_synthetic`, `fatal_fraction_after`
to active run.

---

### Stage: `encode`

**Script**: `src/encode/run.py`
**Purpose**: Use frozen VAE encoder to project `X_train_augmented` (CTGAN-augmented)
and original `X_val`, `X_test` into Z latent vectors. Pure projection — no LSA, no
synthetic row injection inside this stage. Augmentation was handled upstream by `augment`.
**Depends on**: `train_vae` (encoder weights) + `augment` (X_train_augmented)

| Env Var | Default | Description |
|---|---|---|
| `ENCODER_PATH` | `models/vae_encoder.pth` | Frozen encoder weights |
| `X_TRAIN_AUG_PATH` | `data/processed/X_train_augmented.npy` | CTGAN-augmented training features |
| `Y_TRAIN_AUG_PATH` | `data/processed/y_train_augmented.npy` | Augmented training labels |
| `X_VAL_PATH` | `data/processed/X_val.npy` | Validation features (original) |
| `X_TEST_PATH` | `data/processed/X_test.npy` | Test features (original) |
| `OUTPUT_DIR` | `data/processed/` | Dir for Z arrays |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads `vae.latent_dim`, `encode.random_state` |

**Outputs**:
- `OUTPUT_DIR/Z_train_augmented.npy` (shape: N_augmented × latent_dim)
- `OUTPUT_DIR/Z_val.npy` (shape: N_val × latent_dim)
- `OUTPUT_DIR/Z_test.npy` (shape: N_test × latent_dim)
- `OUTPUT_DIR/y_train_augmented.npy` (pass-through from input)

**Exit codes**: `0` = success, `1` = encoder load failure or shape mismatch

---

### Stage: `train_ml`

**Script**: `src/train_ml/run.py`
**Purpose**: Train XGBoost multi-class classifier on Z_train_augmented N times
(one per seed); log each as a separate MLflow run; save the best-seed model artifact.
**Parallel with**: `train_dl` (no dependency between them)

| Env Var | Default | Description |
|---|---|---|
| `TRAIN_Z_PATH` | `data/processed/Z_train_augmented.npy` | Training Z vectors (augmented) |
| `TRAIN_Y_PATH` | `data/processed/y_train_augmented.npy` | Training labels |
| `VAL_Z_PATH` | `data/processed/Z_val.npy` | Validation Z vectors |
| `VAL_Y_PATH` | `data/processed/y_val.npy` | Validation labels |
| `TEST_Z_PATH` | `data/processed/Z_test.npy` | Test Z vectors |
| `TEST_Y_PATH` | `data/processed/y_test.npy` | Test labels |
| `MODEL_OUTPUT_PATH` | `models/best_ml_model.pkl` | Best-seed model artifact |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads `model.*`, `mlflow.*`, `ab_test.seeds` |

**Multi-seed loop**:
```
for seed in params.ab_test.seeds:
    set random_state=seed in XGBClassifier
    compute sample_weight from y_train_augmented class distribution
    fit XGBClassifier with early stopping on Z_val
    log MLflow run: tag seed=<seed>, model_type=xgboost
    log: ein_macro_f1, eout_macro_f1, eout_fatal_recall, generalisation_gap
    log artifact: per_class_matrix.json (P/R/F1 for PDO / Injury / Fatal on Z_test)
    track best seed by eout_macro_f1
save best-seed model to MODEL_OUTPUT_PATH
```

**MLflow side-effects**: N runs in `crash-severity-ml`, tagged `seed=<value>`,
`model_type=xgboost`. `mlflow.sklearn.autolog()` MUST be disabled.
**Exit codes**: `0` = all seeds complete, `1` = any seed fails

---

### Stage: `train_dl`

**Script**: `src/train_dl/run.py`
**Purpose**: Train PyTorch MLP classifier on Z_train_augmented N times (one per seed);
log each run; save best.
**Parallel with**: `train_ml` (no dependency between them)

| Env Var | Default | Description |
|---|---|---|
| `TRAIN_Z_PATH` | `data/processed/Z_train_augmented.npy` | Training Z vectors (augmented) |
| `TRAIN_Y_PATH` | `data/processed/y_train_augmented.npy` | Training labels |
| `VAL_Z_PATH` | `data/processed/Z_val.npy` | Validation Z vectors |
| `VAL_Y_PATH` | `data/processed/y_val.npy` | Validation labels |
| `TEST_Z_PATH` | `data/processed/Z_test.npy` | Test Z vectors |
| `TEST_Y_PATH` | `data/processed/y_test.npy` | Test labels |
| `MODEL_OUTPUT_PATH` | `models/mlp_model.pth` | Best-seed `.pth` checkpoint |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads `model.*`, `dl.*`, `mlflow.*`, `ab_test.seeds` |

**MLP architecture**: `Linear(latent_dim, 64) → ReLU → Dropout(dl.dropout_p) → Linear(64, 3)`
(latent_dim = 8 after T102; constitution IV: max 3 hidden layers)
**Loss**: `CrossEntropyLoss(weight=computed_class_weights)` for training;
`CrossEntropyLoss()` (no weight) for validation loss tracking.

**Multi-seed loop**:
```
for seed in params.ab_test.seeds:
    set torch.manual_seed(seed), numpy.random.seed(seed)
    compute class weights from y_train_augmented distribution
    train MLP with early stopping on Z_val (patience from dl.patience)
    log per-epoch ein_loss, eout_loss, gap_f1 to MLflow (step=epoch)
    log final eout_macro_f1, eout_fatal_recall; tag seed=<seed>, model_type=pytorch-mlp
    log artifact: per_class_matrix.json
    track best seed by eout_macro_f1
save best-seed checkpoint to MODEL_OUTPUT_PATH
```

**MLflow side-effects**: N runs in `crash-severity-dl`, tagged `seed=<value>`,
`model_type=pytorch-mlp`, `architecture=32-64-3-dropout0.3`.
**Exit codes**: `0` = all seeds complete, `1` = any seed fails

---

### Stage: `evaluate`

**Script**: `src/evaluate/run.py`
**Purpose**: A/B test ML vs DL using Welch's t-test on macro F1 distributions;
produce per-class comparison; assert constitutional thresholds on winner; write report.

| Env Var | Default | Description |
|---|---|---|
| `ML_MODEL_PATH` | `models/best_ml_model.pkl` | XGBoost ML model |
| `DL_MODEL_PATH` | `models/mlp_model.pth` | PyTorch MLP checkpoint |
| `TEST_Z_PATH` | `data/processed/Z_test.npy` | Shared test Z vectors (same for both models) |
| `TEST_Y_PATH` | `data/processed/y_test.npy` | Shared test labels |
| `REPORT_PATH` | `docs/evaluation_report.json` | JSON winner + gate PASS/FAIL |
| `AB_REPORT_PATH` | `docs/ab_test_comparison.json` | JSON full A/B table |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads `model.macro_f1_threshold`, `model.fatal_recall_threshold`, `ab_test.*` |

**Statistical A/B test procedure**:
1. Query MLflow for all runs in `crash-severity-ml` tagged `model_type=xgboost`
   → collect `eout_macro_f1` scores → list of N floats
2. Query MLflow for all runs in `crash-severity-dl` tagged `model_type=pytorch-mlp`
   → collect `eout_macro_f1` scores → list of N floats
3. Run Welch's t-test (`scipy.stats.ttest_ind(equal_var=False)`)
4. Compute Cohen's d effect size
5. Compute 95% confidence intervals on each mean
6. Declare winner (p < alpha → higher mean wins; p ≥ alpha → ML/XGBoost default)
7. Assert winner's mean macro F1 > `model.macro_f1_threshold` (0.45) AND
   mean fatal recall > `model.fatal_recall_threshold` (0.30)
8. Write JSON reports; log full report as MLflow artifact

**Outputs**:
- `REPORT_PATH` — winner identity, constitutional gate PASS/FAIL
- `AB_REPORT_PATH` — full statistical report: means, stds, CIs, p-value, Cohen's d,
  raw scores per seed for both models, significance decision, per-class P/R/F1 comparison

**Exit codes**:
- `0` = winner (or default) meets constitutional thresholds
- `1` = winner's mean macro F1 ≤ 0.45 OR mean fatal recall ≤ 0.30

---

### Stage: `tune`

**Script**: `src/tune/run.py`
**Purpose**: Submit Katib Experiment CRD to search β over [0.5, 1.0, 2.0, 4.0, 8.0].
Each trial retrains VAE + encodes + trains winner classifier. Fitness evaluated on
Z_val (NOT Z_test). Best β written to `params.yaml`.

| Env Var | Default | Description |
|---|---|---|
| `VAE_EXPERIMENT_PATH` | `k8s/katib/vae_experiment.yaml` | Katib Experiment CRD |
| `REPORT_PATH` | `docs/evaluation_report.json` | Evaluation report (confirms winner) |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads `tune.*`, `mlflow.experiment_name_tune` |

**Trial contract** (`src/tune/trial.py`):
- Accepts `--beta <float>` as CLI argument
- Retrains VAE with candidate β, re-encodes, trains winner classifier on Z_train_augmented
- Evaluates on Z_val (NOT Z_test) — Katib metric collector reads from stdout:
  `val_macro_f1=<float>` (this is the HPO fitness signal, not the final test metric)
- Logs full trial details to MLflow `crash-severity-tune`: tagged `trial=<n>`,
  `beta=<value>`, `winner=<ml|dl>`

**CRITICAL**: Katib's objective metric MUST be `val_macro_f1` (Z_val performance).
Z_test is never seen by any trial. The final test-set evaluation happens only after
the best β is written to `params.yaml` and the full pipeline is re-run via `dvc repro`.

**Outputs**: `params.yaml` updated with `tune.best_params.beta = <best_value>`
**Exit codes**: `0` = best β written, `1` = Katib submission failed or no trials completed

---

### Stage: `register`

**Script**: `src/register/run.py`
**Purpose**: Promote best MLflow run to Model Registry; set `champion` alias.

| Env Var | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `REPORT_PATH` | `docs/evaluation_report.json` | Evaluation report (confirms PASS status) |
| `RECEIPT_PATH` | `models/registry_receipt.json` | DVC-tracked output |
| `PARAMS_PATH` | `params.yaml` | Reads `mlflow.model_name` |

**Outputs**: MLflow Model Registry entry — `models:/crash-severity/<version>` with
alias `champion`. `RECEIPT_PATH` written for DVC tracking.

**Exit codes**: `0` = model registered and alias set, `1` = evaluate stage FAILED
(refuses to register a model that did not pass constitutional quality gates)

---

## Cross-Cutting Interface Rules

1. **All paths are relative to repo root** when running locally or under DVC.
   Under Kubeflow, paths are absolute (mounted volume paths via hostPath PVC at `/app`).

2. **`PARAMS_PATH` is always available**. Stages MUST NOT hardcode any numeric
   threshold or configuration value — always read from `params.yaml`.

3. **`MLFLOW_TRACKING_URI` is required for `train_vae`, `train_ml`, `train_dl`,
   `evaluate`, `tune`, and `register`**. Under Kubeflow, this must be a URI accessible
   from inside the pod — use the PVC-mounted `mlruns/` path at `/app/mlruns/`.

4. **Non-zero exit MUST be propagated** — no stage may catch an exception and exit 0
   on failure. All errors MUST surface to the orchestrator (DVC or KFP).

5. **Stages are idempotent** — running a stage twice with the same inputs and params
   MUST produce identical outputs (required for DVC caching correctness).

6. **Z arrays are NEVER mixed across splits** — `encode` stage produces three separate
   array files; downstream stages MUST read from the explicit path for their split.
   Z_val and Z_test MUST NOT be used for training under any circumstances.
   X_val and X_test MUST NOT be augmented — augmentation is restricted to X_train only
   (constitution III v3.3.0).

7. **Katib fitness uses Z_val, final gates use Z_test** — the `tune/trial.py` script
   prints `val_macro_f1=<value>` (Z_val) for Katib; `evaluate/run.py` gates use Z_test.
   These are two separate numbers; conflating them violates Principle II.
