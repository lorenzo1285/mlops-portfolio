# Data Model: MLOps Learning Portfolio

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-22 | **Updated**: 2026-04-24

---

## Entities

### 1. Pipeline Stage

A named, self-contained executable unit. Each stage has a deterministic relationship
between inputs (deps) and outputs (outs). Running the same stage twice with the same
deps and params MUST produce byte-identical outputs (idempotency required for DVC caching).

| Field | Type | Description |
|---|---|---|
| `name` | string | Stage identifier: `ingest`, `validate`, `featurize`, `train_ml`, `train_dl`, `evaluate`, `tune`, `register` |
| `cmd` | string | Shell command to execute (e.g., `python -m src.ingest.run`) |
| `deps` | list[path] | Input files/dirs this stage reads; change triggers re-run |
| `outs` | list[path] | Output files/dirs this stage writes; DVC-tracked |
| `params` | list[key] | Keys from `params.yaml` this stage reads; change triggers re-run |

**Stage DAG** (`train_ml` and `train_dl` are parallel — both depend on `featurize`):

```
validate  ← runs on data/raw/CGR_Crash_Data.csv (GE gate before any data is committed)
  └── ingest  ← only runs after validate sentinel exists
        └── featurize
              ├── train_ml ─┐
              └── train_dl ─┴── evaluate
                                    └── tune
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
| `data/processed/.validation_passed` | `dvc.yaml outs` | validate (sentinel — gates ingest) |
| `data/processed/raw.csv` | `dvc.yaml outs` | ingest (only after sentinel exists) |
| `data/processed/X_train.npy` | `dvc.yaml outs` | featurize |
| `data/processed/X_val.npy` | `dvc.yaml outs` | featurize |
| `data/processed/X_test.npy` | `dvc.yaml outs` | featurize |
| `data/processed/y_train.npy` | `dvc.yaml outs` | featurize |
| `data/processed/y_val.npy` | `dvc.yaml outs` | featurize |
| `data/processed/y_test.npy` | `dvc.yaml outs` | featurize |
| `models/preprocessing_pipeline.joblib` | `dvc.yaml outs` | featurize |
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

**Expectation types generated** (one set per column in `params.yaml` `validation.columns`):

| GE Class | Trigger | Column examples |
|---|---|---|
| `ExpectColumnValuesToNotBeNull` | always (all columns) | HOUR, CRASHSEVER, DRIVER1AGE |
| `ExpectColumnValuesToBeBetween` | when `min` or `max` set | HOUR (0–23), SPEEDLIMIT (5–70), DRIVER1AGE (14–100) |
| `ExpectColumnValuesToBeInSet` | when `allowed_values` set | DAYOFWEEK, WEATHER, SURFCOND, CRASHTYPE |

`mostly` per expectation: null-tolerance uses `contract.mostly` per column;
range/value-set checks use `_RANGE_MOSTLY=0.99` to tolerate known sentinel values.

**Validation result** (returned by `GEManager.run_validation()` as a `pd.DataFrame`):

| Column | Type | Description |
|---|---|---|
| `Expectation` | str | GE class name (e.g., `ExpectColumnValuesToNotBeNull`) |
| `Column` | str | Column being checked |
| `Rule` | str | Human label from `meta.rule` (e.g., `HOUR_not_null`) |
| `Success` | bool | Whether this expectation passed |
| `Total Rows` | int | Rows evaluated |
| `Bad Rows` | int | Rows violating the expectation |
| `Success %` | str | Formatted pass rate (e.g., `"99.73%"`) |
| `Examples` | list | Up to 5 offending values (from `partial_unexpected_list`) |

---

### 4a. ML Experiment Run (PyCaret)

Produced by the `train_ml` stage. MLflow experiment: `crash-severity-ml`.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | MLflow UUID |
| `experiment_name` | string | `crash-severity-ml` |
| `params` | dict | PyCaret model name, hyperparameters (logged via `mlflow.log_params` — autolog disabled) |
| `metrics.ein_macro_f1` | float | In-sample macro F1 |
| `metrics.eout_macro_f1` | float | Test-set macro F1 |
| `metrics.eout_minority_recall` | float | Test-set recall on minority class |
| `metrics.generalisation_gap` | float | `eout − ein` |
| `artifact_uri` | string | Path to `.pkl` model |
| `tags.seed` | string | Random seed for this run (e.g., `"3"`) |
| `tags.model_type` | string | `pycaret-ml` |
| `tags.orchestrator` | string | `dvc` / `airflow` / `kubeflow` |

### 4b. DL Experiment Run (EvoTorch NAS + Adam)

Produced by the `train_dl` stage. MLflow experiment: `crash-severity-dl`.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | MLflow UUID |
| `experiment_name` | string | `crash-severity-dl` |
| `metrics.ein_loss` | float (per epoch) | Training BCE loss |
| `metrics.eout_loss` | float (per epoch) | Validation BCE loss |
| `metrics.gap_f1` | float (per epoch) | `eout_macro_f1 − ein_macro_f1` |
| `metrics.eout_macro_f1` | float | Final test-set macro F1 |
| `metrics.eout_minority_recall` | float | Final test-set recall on minority class |
| `artifact_uri` | string | Path to `.pth` checkpoint |
| `tags.seed` | string | Random seed for this run (e.g., `"3"`) |
| `tags.model_type` | string | `evotorch-adam-mlp` |
| `tags.arch_hidden_dims` | string | NAS-selected architecture (e.g., `"[128, 64]"`) |
| `tags.orchestrator` | string | `dvc` / `airflow` / `kubeflow` |

### 4c. A/B Test Result

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
| `gates_passed` | bool | `True` if winner F1 > 0.55 AND recall > 0.40 |

---

### 5. Registered Model

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

### 6. Airflow DAG

The Airflow representation of the pipeline. Eight tasks in dependency order
(train_ml and train_dl are parallel after featurize).

| Field | Type | Description |
|---|---|---|
| `dag_id` | string | `crash_severity_pipeline` |
| `schedule` | string/None | `None` (manual trigger only for portfolio) |
| `tags` | list | `["mlops", "crash-severity"]` |
| `tasks` | list[Task] | `ingest`, `validate`, `featurize`, `train_ml`, `train_dl`, `evaluate`, `tune`, `register` |
| `default_args.retries` | int | `2` |
| `default_args.retry_delay` | timedelta | 5 minutes |

---

### 7. KFP Pipeline

The Kubeflow Pipelines representation. Eight containerised components with the same
logical dependency structure as the Airflow DAG.

| Field | Type | Description |
|---|---|---|
| `pipeline_name` | string | `crash-severity-pipeline` |
| `components` | list[Component] | One `@dsl.component` per stage (8 total) |
| `base_image` | string | `mlops-portfolio:latest` (local Docker image) |
| `pipeline_yaml` | string | `pipelines/kubeflow/pipeline.yaml` (compiled output) |

**Component interface** (all eight components share this pattern):

| Parameter | Direction | Type | Description |
|---|---|---|---|
| `input_path` | input | str | Path to stage's primary input artifact |
| `output_path` | input | str | Path where stage writes its output |
| `params_path` | input | str | Path to `params.yaml` |
| `mlflow_uri` | input | str | MLflow tracking URI (only for train/evaluate/register) |

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
