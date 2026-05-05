# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MLOps Learning Portfolio — Crash Severity Use Case**

A 10-stage ML pipeline on the CGR crash dataset (Grand Rapids, 74,309 rows, 142 cols)
demonstrating the full MLOps toolchain: DVC, Great Expectations, MLflow, Kubeflow
Pipelines, and Katib (HPO).

Pipeline: `validate → ingest → featurize → [train_vae ‖ augment] → encode → [train_ml ‖ train_dl] → evaluate → tune → register`

- `train_vae` and `augment` run in parallel (both depend only on featurize outputs)
- `train_ml` and `train_dl` run in parallel (both depend on encode outputs)
- All other stages are sequential to avoid RAM contention on a single machine
- All stages call `dvc repro <stage>` — both locally and as KFP components
- **Class-based architecture**: each stage has a business-logic class and a thin `run.py` entry point that handles config, I/O, and MLflow logging only. Classes accept all parameters via constructor — no env var reads or file I/O inside them (constitution XIV)
- All stage scripts read config via `src/config.py` typed dataclasses — no raw yaml dicts
- Feature columns, numeric/ordinal/categorical split, cyclical columns all defined in `params.yaml` under `features.*`
- **3-way split (70/15/15)**: `X_train`/`y_train` — fit weights; `X_val`/`y_val` — HPO fitness + early stopping; `X_test`/`y_test` — final A/B test only (never seen during training or search)
- **Cyclical encoding**: HOUR and MONTH replaced by sin/cos pairs in featurize; defined in `features.cyclical_columns`
- **HPO via Katib**: `tune` stage submits a Katib `Experiment` CRD; each trial retrains VAE + encode + winner classifier; β search space in `k8s/katib/vae_experiment.yaml`

## Commands

```bash
# Run a stage script directly
uv run python -m src.<stage>.run

# Add a dependency
uv add <package>

# DVC pipeline
dvc repro                          # run full 8-stage pipeline
dvc repro <stage>                  # run one stage
dvc status                         # check cache state
dvc push                           # sync artifacts to local remote
dvc pull                           # restore artifacts from remote

# MLflow experiment tracking UI
uv run mlflow ui                   # http://localhost:5000

# Airflow workflow orchestration
cd airflow
.\setup.ps1                        # one-time setup
uv run airflow standalone          # http://localhost:8080

# Kubeflow Pipelines + Katib (Docker Desktop Kubernetes)
kubectl apply -f k8s/pvc.yaml                                        # shared hostPath PVC
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80          # KFP UI
kubectl port-forward -n kubeflow svc/katib-ui 8080:80                # Katib UI
python pipelines/kubeflow/pipeline.py                                 # compile pipeline.yaml

# Run tests
uv run pytest tests/
```

## Architecture

### Pipeline Stages (`src/`)

Each stage: `<stage>/run.py` (thin entry point) + `<stage>/<module>.py` (business-logic class).

| Stage | Class module | Purpose |
|-------|-------------|---------|
| validate | `validator.py` → `DataValidator` | GE expectations from params; halt on failure; write Data Docs |
| ingest | `ingester.py` → `Ingester` | Copy raw CSV to `data/processed/raw.csv` |
| featurize | `featurizer.py` → `Featurizer` | 3-way split; 3-group encoding; cyclical HOUR/MONTH; sample complexity gate |
| train_vae | `vae_trainer.py` → `DVAETrainer` | Denoising β-VAE + KL annealing on X_all (unsupervised); save encoder/decoder `.pth` |
| augment | `augmenter.py` → `CTGANAugmenter` | CTGAN on X_train Fatal rows → X_train_augmented; parallel with train_vae |
| encode | `encoder.py` → `LatentEncoder` | Frozen encoder projects X_train_augmented + X_val/test → Z vectors (8-dim) |
| train_ml | `trainer.py` → `MLTrainer` | XGBoost N seeds on Z_train_augmented; autolog disabled; save `.pkl` |
| train_dl | `trainer.py` → `DLTrainer` | Shallow MLP N seeds on Z_train_augmented; cross-entropy + class weights; save `.pth` |
| evaluate | `evaluator.py` → `ABEvaluator` | Welch's t-test on macro F1 distributions; assert constitutional gates |
| tune | `tuner.py` → `HyperparamTuner` | Submit Katib β-HPO Experiment; poll completion; write best β to params.yaml |
| register | `registrar.py` → `ModelRegistrar` | Promote champion to MLflow registry; write `registry_receipt.json` |

### Featurize Stage Details

Four-group `ColumnTransformer` (fitted on X_train only):
- `num` — median impute → StandardScaler (YEAR, SPEEDLIMIT, RDNUMLANES, RDWIDTH, DRIVER1AGE, DRIVER2AGE)
- `cyc` — mode impute → sin/cos pairs (HOUR×2, MONTH×2); period defined in `features.cyclical_columns`
- `ord` — mode impute → OrdinalEncoder(explicit categories, dtype=int) — semantic integer encoding, no scaling (DAYOFWEEK Mon=0…Sun=6)
- `cat` — mode impute → OrdinalEncoder(arbitrary order) (WEATHER, SURFCOND, LIGHTING, etc.)

Optional `FeatureSelector` runs after preprocessing, fits on X_train only. Methods:
- `mutual_info` / `rfe` — supervised; requires y_train
- `correlation` / `vif` — unsupervised collinearity reduction
- `none` — pass-through (default)

### VAE Architecture & Training

**Denoising β-VAE** (`src/train_vae/vae_trainer.py`):
- Trains unsupervised on X_all = concat(X_train, X_val, X_test) — no labels (constitution II exception)
- Input dropout (neural inpainting) → Encoder → reparameterize → Decoder → reconstruct clean X
- KL annealing: `beta_t` ramps from `vae.beta_start=0.0` to `vae.beta_max` over `vae.warmup_epochs` — prevents posterior collapse
- `latent_dim: 8` (fixed; set in params.yaml); `encoder_dims: [256, 128, 64]`
- Logs per-epoch: `vae_elbo`, `vae_reconstruction_loss`, `vae_kl_loss`, `kl_beta`

### CTGAN Augmentation

**CTGANAugmenter** (`src/augment/augmenter.py`):
- Fits TVAE on Fatal-class rows of X_train only
- Generates synthetic Fatal rows until `fatal_fraction >= augment.target_fatal_ratio` (0.05)
- Outputs `X_train_augmented.npy` + `y_train_augmented.npy` as DVC-tracked artifacts
- X_val and X_test are NEVER augmented (constitution III v3.3.0)

### MLP Classifier Architecture

**ShallowMLP** (`src/train_dl/trainer.py`):
- Fixed architecture: `Linear(8, 64) → ReLU → Dropout(dl.dropout_p) → Linear(64, 3)` (constitution IV)
- Operates on Z vectors (8-dim latent space) — same input space as XGBoost
- `CrossEntropyLoss(weight=computed_class_weights)`; runtime-computed weights from y_train distribution
- Early stopping on val loss (`dl.patience`); 10 seeds; best seed by `eout_macro_f1`

### GE Layer Architecture (`great_expectations/gx/utils/`)

The validate stage uses a three-class utility layer with distinct responsibilities. All classes wrap GE v1 (`FileDataContext`) and work exclusively with pandas DataFrames.

```
params.yaml (validation.columns)
        │
        ▼
GEContextBuilder          — INFRASTRUCTURE LAYER (great_expectations/gx/utils/ge_context_builder.py)
  └── build()             — initialises FileDataContext; creates datasource + dataframe asset
                            only; NO suite creation, NO expectation logic
        │
        ▼
GEManager                 — SUITE + PREPARATION LAYER (great_expectations/gx/utils/ge_manager.py)
  ├── build_suite()       — iterates ColumnContract objects; generates ExpectationSuite;
  │                         for range checks uses row_condition per sentinel_value to
  │                         exclude sentinels explicitly (no _RANGE_MOSTLY global tolerance);
  │                         calls context.suites.add_or_update(suite)
  └── select_asset_and_suite() → set_batch_definition() → pre_validate(df)
                            — binds to the built suite; creates BatchDefinition in context;
                              sanity-checks df (non-empty, all expected columns present)
        │
        ▼
GECheckpointRunner        — EXECUTION LAYER (great_expectations/gx/utils/ge_checkpoint_runner.py)
  └── run(df)             — reads the BatchDefinition GEManager registered; builds/updates
                            a Checkpoint with UpdateDataDocsAction +
                            StoreValidationResultAction; calls checkpoint.run(dataframe=df);
                            parses result → returns CheckpointRunResult(success,
                            failed_expectations, data_docs_path)
        │
        ▼
UpdateDataDocsAction      — fires automatically inside the Checkpoint; renders HTML
                            to great_expectations/gx/uncommitted/data_docs/
```

**GEContextBuilder** — infrastructure only. `build()` initialises the GE `FileDataContext`, creates the datasource, and creates the empty dataframe asset. No suite creation, no expectation logic, no `_RANGE_MOSTLY`.

**GEManager** — suite construction + preparation layer. `build_suite(suite_name, asset_name, datasource_name)` generates the `ExpectationSuite` from `ColumnContract` objects:
- `ExpectColumnValuesToNotBeNull` — always generated; uses `contract.mostly` for per-column null tolerance
- `ExpectColumnValuesToBeBetween` — generated when `min`/`max` set; if `contract.sentinel_values` is non-empty, adds a `row_condition` per sentinel to exclude those rows explicitly (no global `mostly` tolerance); if no sentinels, strict (no `mostly`)
- `ExpectColumnValuesToBeInSet` — generated when `allowed_values` set
Then `select_asset_and_suite()` → `set_batch_definition()` → `pre_validate(df)` prepares the context for execution.

**GECheckpointRunner** — execution layer. Constructor + `run(df)` only. Reads the `BatchDefinition` GEManager registered in context; wires a GE v1 `Checkpoint` with `UpdateDataDocsAction` and `StoreValidationResultAction`; calls `checkpoint.run()`; parses `result.run_results` to extract failed expectation names; returns `CheckpointRunResult`. Data Docs are generated automatically by the action.

**DataValidator** (`src/validate/validator.py`) is the pipeline-facing class. Its `validate(df)` method orchestrates all three: `GEContextBuilder.build()` → `GEManager.build_suite()` → `GEManager` preparation → `GECheckpointRunner(...).run(df)` → returns `ValidationResult`.

**Pipeline order**: validate runs on `data/raw/CGR_Crash_Data.csv` (before ingest). If validation passes, `run.py` writes `data/processed/.validation_passed` and exits 0. Ingest is gated on that sentinel.

**Data contract source of truth**: `docs/data_contract.md` (human-readable) and `params.yaml` `validation.columns` (machine-readable). The GE suite is always regenerated from `params.yaml` — the persisted `expectations/` JSON is a DVC-tracked artefact, not edited by hand.

### Katib HPO

- Search space defined in `k8s/katib/vae_experiment.yaml` — β over `[0.5, 1.0, 2.0, 4.0, 8.0]`
- Each trial runs `src/tune/trial.py` as a Kubernetes pod: retrains VAE with candidate β → encode X_train_augmented → train winner classifier → evaluate on Z_val
- Trial prints `val_macro_f1=<float>` to stdout for Katib metrics collector (fitness signal)
- Trial logs full metrics to MLflow `crash-severity-tune` (via PVC-mounted mlruns/)
- `HyperparamTuner.tune()` submits Experiment CRD, polls `status.conditions`, reads `currentOptimalTrial.parameterAssignments`
- Best β written to `params.yaml` under `tune.best_params.beta`; DVC detects change → invalidates train_vae and all downstream stages

### Shared Modules

- `src/config.py` — typed dataclasses: `FeaturesConfig`, `DataConfig`, `ModelConfig`, `DLConfig`, `VAEConfig`, `EncodeConfig`, `AugmentConfig`, `MLflowConfig`, `ABTestConfig`, `ValidationConfig`, `FeatureSelectionConfig`, `TuneConfig`; `load_config()` reads `PARAMS_PATH` env var
- `src/metrics.py` — `make_eval_dataset()`, `per_class_matrix()`, `compute_class_weights()` helpers
- `src/featurize/selector.py` — `FeatureSelector` class; `fit(X, y, feature_names)` + `transform(X) → (X, SelectionResult)`
- `src/train_vae/vae_trainer.py` — `Encoder`, `Decoder`, `DenoisingBetaVAE`, `DVAETrainer`; KL annealing; logs per-epoch ELBO + `kl_beta`
- `src/augment/augmenter.py` — `CTGANAugmenter`; fits TVAE on Fatal rows; returns `AugmentResult`
- `src/encode/encoder.py` — `LatentEncoder`; projects X_train_augmented + X_val/test → Z splits via frozen encoder (μ path, no sampling)
- `great_expectations/gx/utils/ge_context_builder.py` — `GEContextBuilder`; infrastructure only; `build()` creates datasource + dataframe asset; no suite logic
- `great_expectations/gx/utils/ge_manager.py` — `GEManager`; suite construction + preparation; `build_suite()` generates `ExpectationSuite` from `ColumnContract` objects using `row_condition` for sentinel exclusion (no `_RANGE_MOSTLY`); `select_asset_and_suite()` → `set_batch_definition()` → `pre_validate(df)` binds context and sanity-checks DataFrame
- `great_expectations/gx/utils/ge_checkpoint_runner.py` — `GECheckpointRunner`; execution layer; `run(df)` uses the `BatchDefinition` registered by `GEManager`, builds a GE Checkpoint with `UpdateDataDocsAction` + `StoreValidationResultAction`, returns `CheckpointRunResult`
- `src/tune/trial.py` — per-trial CLI entrypoint for Katib; accepts hyperparams as argparse args

### Key Files

- `params.yaml` — all pipeline parameters; `features.*` (incl. `cyclical_columns`), `data.*`, `vae.*` (KL annealing fields), `augment.*`, `dl.*`, `feature_selection.*`, `tune.*`
- `dvc.yaml` — 10-stage pipeline DAG with `deps`/`outs`/`params`
- `UBIQUITOUS_LANGUAGE.md` — canonical domain term glossary (constitution XII)
- `.specify/memory/constitution.md` — v3.3.0, 17 non-negotiable principles
- `great_expectations/gx/` — GE v1 file context root; suite in `expectations/`; HTML report in `uncommitted/data_docs/`
- `great_expectations/gx/utils/` — GE utility layer: `ge_context_builder.py` → `ge_manager.py` → `ge_checkpoint_runner.py`
- `docs/data_contract.md` — human-readable column contract (dtype, ranges, null rates, sentinels)
- `models/registry_receipt.json` — DVC output of register stage
- `k8s/pvc.yaml` — hostPath PVC mounting project root at `/app` for all KFP pods
- `k8s/katib/vae_experiment.yaml` — Katib Experiment CRD for β HPO (search space + trial template)
- `airflow/dags/` — tutorial/learning DAGs only; not part of active pipeline
- `pipelines/kubeflow/pipeline.py` — 10-component KFP pipeline; each component calls `dvc repro <stage>`
- `docker/Dockerfile` — `python:3.12-slim` + uv + deps + `src/` + `dvc.yaml`
- `.gitattributes` — `*.py text eol=lf`

### Specs

- `specs/002-mlops-portfolio/` — active MLOps portfolio spec
  - `spec.md` — feature spec (grill-me reviewed 2026-04-23)
  - `plan.md` — implementation plan (updated 2026-04-29)
  - `tasks.md` — T001–T121; Phases 1–3D complete; Phase 3E needs revision (encode); Phase 3F T101 done, T102–T103 pending; next entry point: T102/T103
  - `contracts/stage-interface.md` — per-stage env var contracts (updated 2026-04-29)
  - `data-model.md` — entities and artifact registry (updated 2026-04-29)

## MLflow Conventions

- `MLFLOW_TRACKING_URI` — set in `params.yaml` under `mlflow.tracking_uri`; every stage calls `mlflow.set_tracking_uri(config.mlflow.tracking_uri)` first
- Experiments: `crash-severity-vae` (β-VAE), `crash-severity-ml` (XGBoost), `crash-severity-dl` (MLP), `crash-severity-tune` (Katib trials)
- Mandatory metrics per run: `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`, `generalisation_gap`
- VAE-specific per-epoch metrics: `vae_elbo`, `vae_reconstruction_loss`, `vae_kl_loss`, `kl_beta`
- `mlflow.sklearn.autolog()` and `mlflow.autolog()` are **disabled** — all metrics logged explicitly
- Champion model alias: `models:/crash-severity@champion`
- Featurize logs: `n_features_raw`, `n_features_selected`, `feature_selection_method`, `samples_per_param_ratio`

## Constitution (v3.4.0)

18 non-negotiable principles at `.specify/memory/constitution.md`. Key ones:

- **I** Feature leakage prevention — no post-crash columns as model inputs
- **II** 3-way split (70/15/15) — train fits weights; val used for HPO fitness + early stopping; test strictly reserved for final A/B evaluation only
- **III** At most three complementary imbalance mechanisms: (1) runtime-computed class weights, (2) CTGAN augmentation on X_train Fatal rows in dedicated `augment` DVC stage, (3) KL annealing to prevent posterior collapse in VAE. X_val/X_test NEVER augmented. Raw SMOTE/ADASYN/interpolation prohibited.
- **IV** Sample complexity gate — `featurize` computes `N_train / n_params` and halts if ratio < 3.0
- **V** MLflow mandatory metrics: `ein_macro_f1`, `eout_macro_f1`, `generalisation_gap`, `eout_fatal_recall`
- **VI** Macro F1 primary metric; 3-class gates: F1 > 0.35, fatal recall > 0.50 (PDO sacrifice accepted — Fatal misclassification cost far exceeds PDO misclassification cost)
- **VII** DVC for all data and model versioning
- **VIII** Great Expectations validation before any training stage
- **XII** `UBIQUITOUS_LANGUAGE.md` maintained; all terms canonical
- **XIII** Grill-me pass before any spec is finalised
- **XIV** Deep module architecture — small interfaces (constructor + 1 public method), large implementation; boundary tests only
- **XV** TDD for all `src/` code — red→green→refactor, vertical slices only
- **XVI** GE is the exclusive data quality assertion layer — no ad-hoc checks in stage code or tests; boundary tests downstream of `validate` use `data/processed/raw.csv`, not `data/raw/`
- **XVIII** Real data fixtures only — all tests MUST use real pipeline artifacts or minimal real-data subsets; purely synthetic/dummy data (e.g., `np.random.randn`) prohibited

Constitution version: **v3.4.0** (last amended 2026-05-02)

## Design Rules (session-established)

- **Design before code**: complete tasks.md before writing any implementation. Code is written only after `/speckit.implement` is invoked.
- **Class pattern**: every stage = one class in `<stage>/<module>.py` (constructor takes typed config objects; one primary public method; all logic private). `run.py` is the only file that reads env vars, opens files, and logs to MLflow.
- **Parameterize everything**: column lists, split ratios, ordinal category orders, feature selection method, regularization strengths — all in `params.yaml`, never hardcoded.
- **Search space separation**: Katib Experiment YAML owns HPO bounds; `params.yaml` owns runtime defaults and stores winning params after tuning.

## Speckit Workflow

Before starting any feature: `/ubiquitous-language` → `/grill-me` → `/speckit.plan` → `/speckit.tasks` → `/speckit.implement`

Slash commands:
- `/speckit.specify` — write feature spec
- `/speckit.plan` — generate implementation plan (runs constitution check)
- `/speckit.tasks` — generate task list
- `/speckit.analyze` — cross-artifact consistency check (read-only)
- `/speckit.implement` — execute tasks
- `/grill-me` — stress-test a spec before planning
- `/ubiquitous-language` — extract/update canonical domain glossary

## Installed Skills

Project-level (`.agents/skills/`):
- `data-scientist` — structured ML workflow: Define → Collect → Engineer → Train → Evaluate → Communicate
- `mlflow` — MLflow experiment tracking patterns
- `mlops-engineer` — MLOps infrastructure and pipeline best practices
- `airflow-dag-patterns` — Airflow DAG design patterns
- `grill-me` — adversarial spec review via structured interview
- `tdd` — red→green→refactor workflow; boundary tests; no internal mocking
- `improve-codebase-architecture` — deep module analysis; RFC GitHub issues
- `ubiquitous-language` — DDD canonical glossary extraction to `UBIQUITOUS_LANGUAGE.md`

Global (`~/.claude/skills/`):
- `find-skills` — search and install skills from skills.sh

## Airflow Tutorial DAGs

`airflow/dags/` contains learning DAGs (do not modify):
- `01_hello_airflow.py` — DAG basics
- `02_crash_ml_pipeline.py` — ML pipeline with notebooks (legacy)
- `03_advanced_concepts.py` — sensors, TaskGroups, XCom, dynamic tasks

Active pipeline DAG: `airflow/dags/crash_ml_pipeline.py` (calls `dvc repro <stage>`)

See `airflow/README.md` and `airflow/CHEATSHEET.md` for reference.
