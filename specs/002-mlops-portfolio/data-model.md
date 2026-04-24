# Data Model: MLOps Learning Portfolio

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-22

---

## Entities

### 1. Pipeline Stage

A named, self-contained executable unit. Each stage has a deterministic relationship
between inputs (deps) and outputs (outs). Running the same stage twice with the same
deps and params MUST produce byte-identical outputs (idempotency required for DVC caching).

| Field | Type | Description |
|---|---|---|
| `name` | string | Stage identifier: `ingest`, `validate`, `featurize`, `train`, `evaluate`, `register` |
| `cmd` | string | Shell command to execute (e.g., `python -m src.ingest.run`) |
| `deps` | list[path] | Input files/dirs this stage reads; change triggers re-run |
| `outs` | list[path] | Output files/dirs this stage writes; DVC-tracked |
| `params` | list[key] | Keys from `params.yaml` this stage reads; change triggers re-run |

**Stage DAG** (`train_ml` and `train_dl` are parallel — both depend on `featurize`):

```
ingest
  └── validate
        └── featurize
              ├── train_ml ─┐
              └── train_dl ─┴── evaluate
                                    └── register
```

`train_ml` and `train_dl` have no dependency on each other and MAY run concurrently
(`dvc repro --run-cache` or parallel Airflow tasks / KFP parallel steps).

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
| `data/processed/raw.csv` | `dvc.yaml outs` | ingest |
| `data/processed/X_train.npy` | `dvc.yaml outs` | featurize |
| `data/processed/X_test.npy` | `dvc.yaml outs` | featurize |
| `data/processed/y_train.npy` | `dvc.yaml outs` | featurize |
| `data/processed/y_test.npy` | `dvc.yaml outs` | featurize |
| `models/preprocessing_pipeline.joblib` | `dvc.yaml outs` | featurize |
| `models/best_ml_model.pkl` | `dvc.yaml outs` | train_ml |
| `models/mlp_model.pth` | `dvc.yaml outs` | train_dl |
| `docs/evaluation_report.json` | `dvc.yaml outs` | evaluate |
| `docs/ab_test_comparison.json` | `dvc.yaml outs` | evaluate |

---

### 3. Expectation Suite

A versioned, committed set of data quality rules for the crash dataset. Defined once,
run at every pipeline execution. Results determine whether the pipeline proceeds.

| Field | Type | Description |
|---|---|---|
| `name` | string | `crash_data_suite` |
| `expectations` | list[Expectation] | Individual quality rules (see below) |
| `meta` | dict | GE metadata (version, created_by) |

**Expectation rules defined** (minimum set covering constitution VIII's 5 categories):

| Category | Expectation | Column(s) |
|---|---|---|
| Schema | `expect_table_columns_to_match_ordered_list` | All 24 selected feature cols + target |
| Null rate | `expect_column_values_to_not_be_null` (threshold 5%) | HOUR, DAYOFWEEK, CRASHSEVER |
| Numeric range | `expect_column_values_to_be_between` | SPEEDLIMIT (0–120), DRIVER1AGE (15–100, excl. 999) |
| Categorical | `expect_column_values_to_be_in_set` | WEATHER, SURFCOND, LIGHTING, DAYOFWEEK |
| Row count | `expect_table_row_count_to_be_between` | min=1000, max=200000 |

**Validation result fields**:

| Field | Type | Description |
|---|---|---|
| `success` | bool | `True` if all expectations pass; `False` halts pipeline |
| `statistics.evaluated_expectations` | int | Total expectations checked |
| `statistics.successful_expectations` | int | Passing count |
| `results` | list[ExpectationValidationResult] | Per-expectation detail |

---

### 4a. ML Experiment Run (PyCaret)

Produced by the `train_ml` stage. MLflow experiment: `crash-severity-ml`.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | MLflow UUID |
| `experiment_name` | string | `crash-severity-ml` |
| `params` | dict | PyCaret model name, hyperparameters (via autolog) |
| `metrics.ein_macro_f1` | float | In-sample macro F1 |
| `metrics.eout_macro_f1` | float | Test-set macro F1 |
| `metrics.generalisation_gap` | float | `eout − ein` |
| `artifact_uri` | string | Path to `.pkl` model |
| `tags.model_type` | string | `pycaret-ml` |
| `tags.orchestrator` | string | `dvc` / `airflow` / `kubeflow` |

### 4b. DL Experiment Run (PyTorch)

Produced by the `train_dl` stage. MLflow experiment: `crash-severity-dl`.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | MLflow UUID |
| `experiment_name` | string | `crash-severity-dl` |
| `metrics.ein_loss` | float (per epoch) | Training BCE loss |
| `metrics.eout_loss` | float (per epoch) | Validation BCE loss |
| `metrics.gap_f1` | float (per epoch) | `eout_macro_f1 − ein_macro_f1` |
| `metrics.eout_macro_f1` | float | Final test-set macro F1 |
| `artifact_uri` | string | Path to `.pth` model |
| `tags.model_type` | string | `pytorch-mlp` |
| `tags.architecture` | string | `128-64-1-dropout0.3` |
| `tags.orchestrator` | string | `dvc` / `airflow` / `kubeflow` |

### 4c. A/B Test Result

Produced by the `evaluate` stage via `mlflow.evaluate()`. Logged as a run artifact.

| Field | Type | Description |
|---|---|---|
| `model_a` | string | `pycaret-ml` run_id |
| `model_b` | string | `pytorch-mlp` run_id |
| `winner` | string | `model_a` or `model_b` |
| `winner_macro_f1` | float | Winning model's eout_macro_f1 |
| `delta_macro_f1` | float | `winner_f1 − loser_f1` (margin) |
| `model_a_metrics` | dict | Full metric set for ML model |
| `model_b_metrics` | dict | Full metric set for DL model |
| `gates_passed` | bool | True if winner meets F1 > 0.55 AND recall > 0.40 |

### 4. Experiment Run

A single training execution tracked in MLflow. Created automatically by `mlflow.autolog()`.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | MLflow UUID (e.g., `a3f8...`) |
| `experiment_name` | string | `crash-severity-ml` |
| `status` | enum | `RUNNING`, `FINISHED`, `FAILED` |
| `params` | dict | Hyperparameters logged by autolog |
| `metrics.ein_macro_f1` | float | In-sample macro F1 |
| `metrics.eout_macro_f1` | float | Test-set macro F1 |
| `metrics.generalisation_gap` | float | `eout_macro_f1 − ein_macro_f1` |
| `artifact_uri` | string | Path to logged model artifact |
| `tags.orchestrator` | string | `dvc` / `airflow` / `kubeflow` (set manually per run) |

---

### 5. Registered Model

A promoted model artifact in the MLflow Model Registry. Independent of the training
run ID and local filesystem path.

| Field | Type | Description |
|---|---|---|
| `name` | string | `crash-severity` |
| `version` | int | Auto-incremented on each registration |
| `alias` | string | `champion` (replaces deprecated `Production` stage in MLflow 3.x) |
| `source` | string | `runs:/<run_id>/model` — the promoted run's artifact |
| `creation_timestamp` | int | Unix ms |

**Load pattern**: `mlflow.pyfunc.load_model("models:/crash-severity@champion")`

---

### 6. Airflow DAG

The Airflow representation of the pipeline. Six tasks in linear dependency order.

| Field | Type | Description |
|---|---|---|
| `dag_id` | string | `crash_severity_pipeline` |
| `schedule` | string/None | `None` (manual trigger only for portfolio) |
| `tags` | list | `["mlops", "crash-severity"]` |
| `tasks` | list[Task] | `ingest`, `validate`, `featurize`, `train`, `evaluate`, `register` |
| `default_args.retries` | int | `2` |
| `default_args.retry_delay` | timedelta | 5 minutes |

---

### 7. KFP Pipeline

The Kubeflow Pipelines representation. Six containerised components with the same
logical dependency structure as the Airflow DAG.

| Field | Type | Description |
|---|---|---|
| `pipeline_name` | string | `crash-severity-pipeline` |
| `components` | list[Component] | One `@dsl.component` per stage |
| `base_image` | string | `mlops-portfolio:latest` (local Docker image) |
| `pipeline_yaml` | string | `pipelines/kubeflow/pipeline.yaml` (compiled output) |

**Component interface** (all six components share this pattern):

| Parameter | Direction | Type | Description |
|---|---|---|---|
| `input_path` | input | str | Path to stage's primary input artifact |
| `output_path` | input | str | Path where stage writes its output |
| `params_path` | input | str | Path to `params.yaml` |
| `mlflow_uri` | input | str | MLflow tracking URI (only for train/register) |

---

## State Transitions

### Pipeline Execution State

```
PENDING → RUNNING → SUCCESS
                 ↘ FAILED (validate failure halts all downstream stages)
```

### Model Lifecycle

```
[Experiment Run] → registered → version=N, alias=None
                             → alias="champion" (promoted by register stage)
                             → alias="champion" overwritten (next better model)
```
