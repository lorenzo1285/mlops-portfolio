# Implementation Plan: MLOps Learning Portfolio — Crash Severity Use Case

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-22 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/002-mlops-portfolio/spec.md`

---

## Summary

Build an eight-stage ML pipeline on the CGR crash dataset:
`ingest → validate → featurize → train_ml → train_dl → evaluate → tune → register`

`train_ml` uses PyCaret (`compare_models` + `tune_model`); `train_dl` uses a PyTorch
shallow MLP (128→64→1). Both are tracked in MLflow. The `evaluate` stage runs an
A/B test via `mlflow.evaluate()` on the shared held-out test set and declares a winner
by macro F1. The winner is registered as `models:/crash-severity@champion`. The full
pipeline is versioned with DVC, quality-gated with Great Expectations, and orchestrated
by both Apache Airflow and Kubeflow Pipelines (Docker Desktop Kubernetes).

---

## Technical Context

**Language/Version**: Python 3.12 (pinned in `.python-version`)
**Package Manager**: uv (existing)
**Primary Dependencies**:
- `dvc>=3.0` — pipeline versioning and caching
- `great-expectations>=1.0` — data validation (v1 API, breaking change from 0.x)
- `mlflow>=3.10.1` — experiment tracking and model registry (already in pyproject.toml)
- `apache-airflow>=2.8.0` — local DAG orchestration (already in pyproject.toml)
- `kfp>=2.0` — Kubeflow Pipelines SDK v2
- `optuna>=3.0` — hyperparameter optimisation (TPE sampler, MLflow integration)
- `optuna-integration[mlflow]` — MLflow callback for per-trial run logging
- `scikit-learn`, `pycaret>=3.3.2` — ML training (already in pyproject.toml)

**Storage**:
- DVC cache: `.dvc/cache/` (local) + DVC remote: `data/dvc-remote/` (local dir)
- MLflow tracking: `mlruns/` (local filesystem)
- Great Expectations Data Docs: `great_expectations/uncommitted/data_docs/`
- Processed artifacts: `data/processed/`, `models/`

**Testing**: pytest — boundary tests per stage's public interface (ENV-var in / artifact out); written test-first (red→green→refactor); no unit tests on internal functions

**Target Platform**: Windows 11 / Docker Desktop / Docker Desktop Kubernetes (single-node)

**Project Type**: ML pipeline / MLOps learning portfolio

**Performance Goals**:
- Full `dvc repro` end-to-end: < 30 minutes (dominated by PyCaret `compare_models`)
- Model load from registry: < 30 seconds (SC-006)
- Data validation: < 60 seconds for 74k rows

**Constraints**:
- Docker Desktop Kubernetes: ~4GB RAM available for Kubeflow pods
- Kubeflow Pipelines **standalone** only (not full Kubeflow) — lighter footprint
- No cloud dependencies — all storage and tracking is local
- Single base Docker image (`python:3.12-slim`) for all stages to reduce build overhead

**Scale/Scope**: 74,309 rows × 142 cols → ~50–80 selected features; 8 pipeline stages;
2 orchestrators; 1 Docker image; 1 DVC remote (local); 3-way split (70/15/15 train/val/test)

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design: all PASS.*

| Gate | Status | Evidence |
|---|---|---|
| No post-crash columns in feature list | ✅ PASS | spec FR-001, constitution I; featurize stage selects pre-crash cols only |
| Pipeline fit only on train | ✅ PASS | featurize stage fits pipeline on `X_train` only (constitution II) |
| Class weights (not SMOTE) | ✅ PASS | `params.yaml` carries `class_weight_pos=2.74` (constitution III) |
| MLflow tracking in plan | ✅ PASS | train_ml uses `mlflow.sklearn.autolog()`; train_dl logs per-epoch; evaluate uses `mlflow.evaluate()` A/B (constitution V) |
| Macro F1 threshold assertion in tasks | ✅ PASS | evaluate stage A/B test asserts F1 > 0.55, recall > 0.40 for winner (constitution VI) |
| DVC pipeline stages in `dvc.yaml` | ✅ PASS | all 6 stages defined with `deps`/`outs`/`cmd` (constitution VII) |
| GE validation before train stage | ✅ PASS | validate stage is a DVC dep of featurize; blocks pipeline on failure (constitution VIII) |
| Airflow DAG and KFP pipeline both defined | ✅ PASS | US4 + US5, FR-006 + FR-007 (constitution IX) |
| Each stage runnable as Docker container | ✅ PASS | single Dockerfile + ENV-var interface (constitution X) |
| No notebook in pipeline tasks | ✅ PASS | all stages in `src/*/run.py`; `notebooks/` is exploration only (constitution XI) |
| `UBIQUITOUS_LANGUAGE.md` exists and covers all spec terms | ⚠️ PENDING | file generated at spec review 2026-04-23; spec Key Entities + stage names canonicalised (constitution XII) |
| No flagged ambiguities unresolved in glossary | ⚠️ PENDING | to be verified after `/ubiquitous-language` run (constitution XII) |
| Grill-me pass completed and spec updated | ⚠️ PENDING | to be completed before implementation begins (constitution XIII) |
| All `src/` code written test-first; boundary tests only | ⚠️ PENDING | TDD tasks added to tasks.md preceding each implementation task (constitution XIV, XV) |
| No shallow modules introduced | ⚠️ PENDING | `src/utils.py` load_params() flagged for review; to be verified at feature close (constitution XIV) |

**Result**: Gates I–XI PASS. Gates XII–XV PENDING — must be resolved before `/speckit.implement` begins.

---

## Project Structure

### Documentation (this feature)

```text
specs/002-mlops-portfolio/
├── plan.md              ✅ this file
├── research.md          ✅ Phase 0 output
├── data-model.md        ✅ Phase 1 output
├── quickstart.md        ✅ Phase 1 output
├── contracts/
│   └── stage-interface.md  ✅ Phase 1 output
└── tasks.md             🔜 Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
mlops-portfolio/
│
├── src/                          # Stage business logic — ONE script per stage
│   ├── config.py                 # Typed dataclass accessors for params.yaml; schema validation
│   ├── metrics.py                # make_eval_dataset() helper for mlflow.evaluate()
│   ├── ingest/
│   │   └── run.py                # Copy raw CSV → data/processed/raw.csv
│   ├── validate/
│   │   └── run.py                # Build GE expectations from params; halt on failure; write Data Docs
│   ├── featurize/
│   │   └── run.py                # Select features.columns from params; encode, split, fit+save pipeline
│   ├── train_ml/
│   │   └── run.py                # PyCaret compare+tune; autolog disabled; mlflow.evaluate(); save .pkl
│   ├── train_dl/
│   │   ├── run.py                # PyTorch ShallowMLP; BCEWithLogitsLoss; early stopping; save .pth
│   │   └── pyfunc.py             # mlflow.pyfunc.PythonModel wrapper for ShallowMLP
│   ├── evaluate/
│   │   └── run.py                # mlflow.evaluate() A/B test; comparison table; assert gates
│   ├── tune/
│   │   └── run.py                # Optuna TPE search on winner; MLflowCallback per trial; write best_params to params.yaml
│   └── register/
│       └── run.py                # Promote winner; write models/registry_receipt.json
│
├── pipelines/
│   └── kubeflow/
│       └── pipeline.py           # KFP v2 pipeline — 7 @dsl.component definitions, each calls dvc repro <stage>
│
├── k8s/
│   └── pvc.yaml                  # hostPath PVC mounting project root at /app for all KFP pods
│
├── docker/
│   └── Dockerfile                # Single image: python:3.12-slim + uv + deps; includes dvc.yaml
│
├── .gitattributes                # Forces LF line endings on *.py to prevent CRLF issues in containers
│
├── great_expectations/           # GE v1 context (generated by `gx init`)
│   ├── great_expectations.yml
│   └── expectations/
│       └── crash_data_suite.json # Committed expectation suite
│
├── data/
│   ├── raw/
│   │   └── CGR_Crash_Data.csv    # DVC-tracked (existing file)
│   ├── processed/                # DVC pipeline outputs
│   │   ├── raw.csv
│   │   ├── X_train.npy
│   │   ├── X_val.npy
│   │   ├── X_test.npy
│   │   ├── y_train.npy
│   │   ├── y_val.npy
│   │   └── y_test.npy
│   └── dvc-remote/               # Local DVC remote storage
│
├── models/                       # DVC-tracked pipeline outputs
│   ├── preprocessing_pipeline.joblib
│   └── best_ml_model.pkl
│
├── docs/
│   ├── data_contract.md          # Column-level data contract: valid ranges, null thresholds, allowed values
│   └── evaluation_report.json    # Metrics output of evaluate stage
│
├── airflow/
│   └── dags/
│       └── crash_ml_pipeline.py  # Airflow TaskFlow DAG — calls src/ stages
│
├── notebooks/                    # Exploration ONLY — not in any pipeline
│   └── eda.ipynb
│
├── dvc.yaml                      # DVC pipeline: 8 stages with deps/outs/params
├── .gitattributes                # *.py text eol=lf — prevents CRLF issues in Linux containers
├── params.yaml                   # All configurable pipeline parameters
├── .dvcignore
├── pyproject.toml                # Updated with dvc, great-expectations, kfp
└── specs/002-mlops-portfolio/
```

**Structure Decision**: Single-project layout. `src/` holds all stage scripts; no
`services/` or `api/` layer needed. `pipelines/` holds only the Kubeflow definition
(Airflow DAGs stay in `airflow/dags/` per existing convention). All DVC outputs are
under `data/processed/` and `models/` — both DVC-tracked, not git-tracked.

---

## Notes

- `train_ml` and `train_dl` are **independent but sequential** — no data dependency between them, but executed one after the other to avoid RAM contention (PyCaret + PyTorch simultaneously exceeds single-machine memory). Independence demonstrated via DAG structure.
- The featurize stage produces a **3-way stratified split** (70% train / 15% val / 15% test) controlled by `params.yaml` keys `data.train_size`, `data.val_size`, `data.test_size`. `X_val`/`y_val` is used exclusively for NAS/HPO fitness and DL early stopping. `X_test`/`y_test` is strictly reserved for the final `evaluate` stage A/B test — it is never seen during training or search (constitution Principle II gate).
- Both Airflow tasks and KFP components call `dvc repro <stage>` — not scripts directly. DVC caching and param tracking apply to all execution contexts.
- KFP uses a single hostPath PVC mounting the project root at `/app`. `params.yaml` is mounted from PVC (overrides image default) so params change without image rebuilds. `dvc.yaml` is baked into the image (pipeline-as-code).
- `MLFLOW_TRACKING_URI` is an absolute path in `params.yaml`. All stages call `mlflow.set_tracking_uri(config.mlflow.tracking_uri)` via `src/config.py` — no relative paths.
- `mlflow.sklearn.autolog()` is **disabled** in `train_ml`. All metrics logged via `mlflow.evaluate()` for consistency across both models.
- `src/config.py` provides typed dataclass accessors for `params.yaml`. No raw dict access in stage scripts.
- `docs/data_contract.md` defines column requirements from EDA; encoded in `params.yaml` under `validation.*`; GE expectations generated programmatically from params.
- `register` stage writes `models/registry_receipt.json` as its DVC `outs`.
- Concurrent orchestrator execution is not supported — one pipeline run at a time.
- The `tune` stage runs Optuna TPE sampler with `MLflowCallback` — each trial = one MLflow run in `crash-severity-tune`. Search space for ML and DL defined in `params.yaml` under `tune.ml_search_space` / `tune.dl_search_space`. Best params written to `params.yaml` under `tune.best_params` after search.
- For PyTorch, Optuna pruning halts unpromising trials early based on per-epoch validation loss, reducing total HPO time significantly.
- PyCaret `compare_models()` ~5–10 min/seed. Use `ab_test.seeds: [0,1,2]` during development.

## Complexity Tracking

> No constitution violations — this section left intentionally empty.
