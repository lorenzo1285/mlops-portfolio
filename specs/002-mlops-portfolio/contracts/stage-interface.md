# Stage Interface Contracts

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-22

Each pipeline stage MUST honour this interface so it runs identically under DVC,
Airflow, and Kubeflow. All paths are passed via environment variables — no hardcoded
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

### Stage: `ingest`

**Script**: `src/ingest/run.py`
**Purpose**: Copy raw CSV to processed dir, validate it is readable, log row count.

| Env Var | Default | Description |
|---|---|---|
| `INPUT_PATH` | `data/raw/CGR_Crash_Data.csv` | Source raw CSV |
| `OUTPUT_PATH` | `data/processed/raw.csv` | Destination copy |
| `PARAMS_PATH` | `params.yaml` | Parameters file |

**Outputs**: `OUTPUT_PATH` (CSV, same schema as input)
**Exit codes**: `0` = success, `1` = input file not found or unreadable

---

### Stage: `validate`

**Script**: `src/validate/run.py`
**Purpose**: Run GE expectation suite; write Data Docs; exit non-zero on failure.

| Env Var | Default | Description |
|---|---|---|
| `INPUT_PATH` | `data/processed/raw.csv` | CSV to validate |
| `DATA_DOCS_PATH` | `great_expectations/uncommitted/data_docs/` | Output HTML dir |
| `PARAMS_PATH` | `params.yaml` | Parameters file (reads `great_expectations.*`) |
| `GE_ROOT` | `great_expectations/` | GE context root directory |

**Outputs**: `DATA_DOCS_PATH/` (HTML Data Docs)
**Exit codes**: `0` = all expectations pass, `1` = one or more expectations fail

**Contract**: A non-zero exit MUST propagate as a pipeline failure in all three
execution contexts (DVC stops, Airflow marks task failed, KFP marks step failed).

---

### Stage: `featurize`

**Script**: `src/featurize/run.py`
**Purpose**: Select pre-crash features, encode, split 80/20, fit+save pipeline.

| Env Var | Default | Description |
|---|---|---|
| `INPUT_PATH` | `data/processed/raw.csv` | Validated CSV |
| `OUTPUT_DIR` | `data/processed/` | Dir for numpy arrays |
| `PIPELINE_PATH` | `models/preprocessing_pipeline.joblib` | Fitted pipeline output |
| `PARAMS_PATH` | `params.yaml` | Reads `data.test_size`, `data.random_state`, `data.sentinel_value` |

**Outputs**:
- `OUTPUT_DIR/X_train.npy`, `OUTPUT_DIR/X_test.npy`
- `OUTPUT_DIR/y_train.npy`, `OUTPUT_DIR/y_test.npy`
- `PIPELINE_PATH` (fitted joblib pipeline)

**Exit codes**: `0` = success, `1` = input not found or < 5% rows dropped threshold exceeded

---

### Stage: `train_ml`

**Script**: `src/train_ml/run.py`
**Purpose**: Train PyCaret model N times (one per seed); log each as a separate MLflow
run; save the best-seed model artifact.
**Parallel with**: `train_dl` (no dependency between them)

| Env Var | Default | Description |
|---|---|---|
| `TRAIN_X_PATH` | `data/processed/X_train.npy` | Training features |
| `TRAIN_Y_PATH` | `data/processed/y_train.npy` | Training labels |
| `TEST_X_PATH` | `data/processed/X_test.npy` | Test features |
| `TEST_Y_PATH` | `data/processed/y_test.npy` | Test labels |
| `PIPELINE_PATH` | `models/preprocessing_pipeline.joblib` | For feature name recovery |
| `MODEL_OUTPUT_PATH` | `models/best_ml_model.pkl` | Best-seed model artifact |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads `model.*`, `mlflow.*`, `ab_test.seeds` |

**Multi-seed loop** (within one DVC stage invocation):
```
for seed in params.ab_test.seeds:
    set random_state=seed in PyCaret setup()
    run compare_models() + tune_model()
    log MLflow run with tag seed=<seed>, model_type=pycaret-ml
    track best seed by eout_macro_f1
save best-seed model to MODEL_OUTPUT_PATH
```

**MLflow side-effects**: N runs in `crash-severity-ml`, each tagged `seed=<value>`,
`model_type=pycaret-ml`, with `ein_macro_f1`, `eout_macro_f1`, `generalisation_gap`.
**Exit codes**: `0` = all seeds complete, `1` = any seed fails

---

### Stage: `train_dl`

**Script**: `src/train_dl/run.py`
**Purpose**: Train PyTorch ShallowMLP N times (one per seed); log each run; save best.
**Parallel with**: `train_ml` (no dependency between them)

| Env Var | Default | Description |
|---|---|---|
| `TRAIN_X_PATH` | `data/processed/X_train.npy` | Training features |
| `TRAIN_Y_PATH` | `data/processed/y_train.npy` | Training labels |
| `TEST_X_PATH` | `data/processed/X_test.npy` | Test features |
| `TEST_Y_PATH` | `data/processed/y_test.npy` | Test labels |
| `MODEL_OUTPUT_PATH` | `models/mlp_model.pth` | Best-seed `.pth` checkpoint |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads `model.*`, `dl.*`, `mlflow.*`, `ab_test.seeds` |

**Multi-seed loop** (within one DVC stage invocation):
```
for seed in params.ab_test.seeds:
    set torch.manual_seed(seed), numpy.random.seed(seed)
    train ShallowMLP with early stopping
    log per-epoch ein_loss, eout_loss, gap_f1 to MLflow (step=epoch)
    log final eout_macro_f1; tag seed=<seed>, model_type=pytorch-mlp
    track best seed by eout_macro_f1
save best-seed checkpoint to MODEL_OUTPUT_PATH
```

**MLflow side-effects**: N runs in `crash-severity-dl`, each tagged `seed=<value>`,
`model_type=pytorch-mlp`, `architecture=128-64-1-dropout0.3`.
**Exit codes**: `0` = all seeds complete, `1` = any seed fails

---

### Stage: `evaluate`

**Script**: `src/evaluate/run.py`
**Purpose**: A/B test ML vs DL using `mlflow.evaluate()` on the shared test set;
produce comparison table; assert constitutional thresholds on the winner; write report.

| Env Var | Default | Description |
|---|---|---|
| `ML_MODEL_PATH` | `models/best_ml_model.pkl` | PyCaret ML model |
| `DL_MODEL_PATH` | `models/mlp_model.pth` | PyTorch DL model checkpoint |
| `PIPELINE_PATH` | `models/preprocessing_pipeline.joblib` | For DL model input prep |
| `TEST_X_PATH` | `data/processed/X_test.npy` | Shared test features (same for both models) |
| `TEST_Y_PATH` | `data/processed/y_test.npy` | Shared test labels |
| `REPORT_PATH` | `docs/evaluation_report.json` | JSON winner + full metrics |
| `AB_REPORT_PATH` | `docs/ab_test_comparison.json` | JSON side-by-side A/B table |
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `PARAMS_PATH` | `params.yaml` | Reads thresholds |

**Statistical A/B test procedure**:
1. Query MLflow for all runs in `crash-severity-ml` tagged `model_type=pycaret-ml`
   → collect `eout_macro_f1` scores → list of N floats
2. Query MLflow for all runs in `crash-severity-dl` tagged `model_type=pytorch-mlp`
   → collect `eout_macro_f1` scores → list of N floats
3. Run **Welch's t-test** (`scipy.stats.ttest_ind(equal_var=False)`) on the two lists
4. Compute **Cohen's d** effect size: `(mean_a - mean_b) / pooled_std`
5. Compute **95% confidence intervals** on each mean: `mean ± 1.96 * std / sqrt(N)`
6. Declare winner:
   - If p < `ab_test.alpha` → winner = higher-mean model
   - If p ≥ `ab_test.alpha` → winner = ML/PyCaret (simpler model default), labelled
     "no significant difference"
7. Assert winner's mean macro F1 > 0.55 AND mean minority recall > 0.40
8. Log full statistical report as MLflow artifact; write JSON to `AB_REPORT_PATH`

**Outputs**:
- `REPORT_PATH` — winner identity, constitutional gate PASS/FAIL
- `AB_REPORT_PATH` — full statistical report: means, stds, CIs, p-value, Cohen's d,
  raw scores per seed for both models, significance decision

**Exit codes**:
- `0` = winner (or default) meets constitutional thresholds
- `1` = winner's mean macro F1 ≤ 0.55 OR mean minority recall ≤ 0.40

---

### Stage: `register`

**Script**: `src/register/run.py`
**Purpose**: Promote best MLflow run to Model Registry; set `champion` alias.

| Env Var | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `mlruns/` | MLflow tracking store |
| `REPORT_PATH` | `docs/evaluation_report.json` | Evaluation report (confirms PASS status) |
| `PARAMS_PATH` | `params.yaml` | Reads `mlflow.model_name`, `mlflow.experiment_name` |

**Outputs**: MLflow Model Registry entry — `models:/crash-severity/<version>` with
alias `champion`. No filesystem file outputs (DVC does not track the registry).

**Exit codes**: `0` = model registered and alias set, `1` = evaluate stage FAILED
(refuses to register a model that did not pass quality gates)

---

## Cross-Cutting Interface Rules

1. **All paths are relative to repo root** when running locally or under DVC.
   Under Kubeflow, paths are absolute (mounted volume paths).

2. **`PARAMS_PATH` is always available**. Stages MUST NOT hardcode any numeric
   threshold or configuration value — always read from `params.yaml`.

3. **`MLFLOW_TRACKING_URI` is required for `train` and `register`**. Under Kubeflow,
   this must be a URI accessible from inside the pod (e.g., a `PersistentVolumeClaim`
   mount or a networked MLflow server).

4. **Non-zero exit MUST be propagated** — no stage may catch an exception and exit 0
   on failure. All errors MUST surface to the orchestrator.

5. **Stages are idempotent** — running a stage twice with the same inputs and params
   MUST produce identical outputs (required for DVC caching correctness).
