# Tasks: MLOps Learning Portfolio — Crash Severity Use Case

**Input**: Design documents from `specs/002-mlops-portfolio/`
**Prerequisites**: spec.md ✅ | plan.md ✅ | research.md ✅ | data-model.md ✅ | contracts/stage-interface.md ✅

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story this task belongs to (US1–US5)
- All file paths are relative to repo root

---

## Phase 1: Setup

**Purpose**: Scaffold directories, install dependencies, create shared config.

- [x] T001 Create all required directories: `src/ingest/`, `src/validate/`, `src/featurize/`, `src/train_ml/`, `src/train_dl/`, `src/evaluate/`, `src/register/`, `pipelines/kubeflow/`, `docker/`, `models/`, `docs/`, `data/processed/`, `data/dvc-remote/`
- [x] T002 [P] Add new dependencies to `pyproject.toml`: `dvc>=3.0`, `great-expectations>=1.0`, `kfp>=2.0`, `scipy>=1.11` — then run `uv sync`
- [x] T003 [P] Create `params.yaml` at repo root with all sections: `features` (columns list, numeric_columns, ordinal_columns with explicit category orders for DAYOFWEEK and MONTH, target_column, sentinel_columns), `data` (train_size: 0.70, val_size: 0.15, test_size: 0.15, random_state, sentinel_value: 999), `model`, `dl` (includes `weight_decay: 1e-4` for L2 regularization via Adam), `mlflow`, `great_expectations`, `ab_test`, `feature_selection` (method: none, n_features, threshold), `tune` (katib_namespace, katib_experiment_name, n_trials, parallel_trials — search space bounds live in k8s/katib/*.yaml, not here) — values from research.md Decision 9
- [x] T004 [P] Create `.dvcignore` excluding: `mlruns/`, `**/__pycache__/`, `**/*.pyc`, `.venv/`, `data/dvc-remote/`
- [x] T005 [P] Create `src/__init__.py` and `src/<stage>/__init__.py` for all 8 stage packages: `ingest`, `validate`, `featurize`, `train_ml`, `train_dl`, `evaluate`, `tune`, `register`
- [ ] T060 [P] Create `.gitattributes` at repo root: `*.py text eol=lf` and `*.sh text eol=lf` — enforces LF line endings for all Python and shell files when checked out on Windows, preventing CRLF issues in Linux Docker containers

**Checkpoint**: Directories exist, dependencies installed, params.yaml and .dvcignore in place.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Initialise DVC and GE — everything else depends on these.

⚠️ **CRITICAL**: No user story work can begin until this phase is complete.

- [x] T006 Run `dvc init` at repo root — creates `.dvc/` directory and `config` file
- [x] T007 Configure DVC local remote: `dvc remote add -d local data/dvc-remote` — updates `.dvc/config`
- [x] T008 Track raw dataset: `dvc add data/raw/CGR_Crash_Data.csv` — creates `data/raw/CGR_Crash_Data.csv.dvc` pointer and `data/raw/.gitignore`; commit both pointer files to git
- [x] T009 Delete existing `great_expectations/gx/` directory entirely (no valid context or expectations exist there). Initialise fresh GE v1 context: `python -c "import great_expectations as gx; gx.get_context(mode='file', project_root_dir='great_expectations/gx')"` — creates `great_expectations/gx/great_expectations.yml`
- [x] T010 [P] Create `src/config.py` with typed dataclass accessors for all `params.yaml` sections: `FeaturesConfig(columns, numeric_columns, ordinal_columns: dict[str,list[str]], target_column, sentinel_columns)`, `DataConfig(train_size, val_size, test_size, random_state, sentinel_value)`, `ModelConfig(class_weight_neg, class_weight_pos, n_select, macro_f1_threshold, minority_recall_threshold)`, `DLConfig(epochs, patience, batch_size, lr, hidden_1, hidden_2, dropout, weight_decay)`, `MLflowConfig(tracking_uri, experiment_name_ml, experiment_name_dl, model_name)`, `ABTestConfig(seeds, alpha, tiebreak)`, `ValidationConfig(columns: dict)`, `FeatureSelectionConfig(method, n_features, threshold)`. Top-level `load_config(path=None)` reads `PARAMS_PATH` env var, validates required sections present, returns `ProjectConfig`. All stage scripts import from `src/config.py` — no raw `yaml.safe_load` in stage scripts.
- [ ] T068 [P] Create `src/ingest/ingester.py`: `IngestResult(row_count, output_path)` dataclass; `Ingester(input_path, output_path)` class whose `run() → IngestResult` reads the source CSV and writes it to the output path. Schema-agnostic — no dataset-specific logic. No env var reads or file-path decisions inside the class.
- [ ] T071 [P] Add `feature_selection` section to `params.yaml`: `method` (one of `mutual_info`, `rfe`, `correlation`, `vif`, `none`; default `none`), `n_features` (int; used by supervised methods — number of top features to keep), `threshold` (float; used by unsupervised methods — max allowed pairwise correlation or VIF value before a feature is dropped). Add `FeatureSelectionConfig(method: str, n_features: int, threshold: float)` dataclass to `src/config.py` and include it in `ProjectConfig`.
- [ ] T072 [P] Create `src/featurize/selector.py`: `SelectionResult(selected_cols: list[str], dropped_cols: list[str], scores: dict[str, float] | None)` dataclass; `FeatureSelector(method, n_features, threshold)` class with two public methods — `fit(X: np.ndarray, y: np.ndarray | None, feature_names: list[str]) → FeatureSelector` (fits on train data only; supervised methods require y, unsupervised ignore it) and `transform(X: np.ndarray) → tuple[np.ndarray, SelectionResult]` (applies fitted selection mask). Supported methods: `mutual_info` → `SelectKBest(mutual_info_classif, k=n_features)` fitted on X_train + y_train; `rfe` → `RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=n_features)` fitted on X_train + y_train; `correlation` → iteratively drop the column from each pair with pairwise Pearson correlation > threshold (no y required); `vif` → iteratively drop the column with highest VIF > threshold until all VIFs ≤ threshold (no y required); `none` → pass-through, all columns kept. All fitting on training data only — never sees val or test.
- [ ] T069 [P] Create `src/featurize/featurizer.py`: `FeaturizeResult(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, selector, feature_cols, selected_cols, n_params, samples_per_param_ratio)` dataclass; `Featurizer(feature_cols, numeric_cols, ordinal_cols, target_col, train_size, val_size, test_size, random_state, sentinel_value, sentinel_cols, feature_selector: FeatureSelector | None)` class whose `fit_transform(df) → FeaturizeResult` pipeline is: `_select_and_recode` → `_separate_target` → `_split` (3-way) → `_fit_preprocess` (3-group ColumnTransformer) → optionally `feature_selector.fit(X_train, y_train).transform(all three splits)` → `_sample_complexity`. When `feature_selector` is provided, sample complexity is computed on the post-selection feature count. No file I/O or env var reads inside the class.
- [ ] T070 [P] Create stub class modules for all remaining stages — each defines the constructor signature and typed result dataclass; method body is `raise NotImplementedError`: `src/validate/validator.py` (`DataValidator(ge_context_root, validation_config)` → `validate(df) → ValidationResult`); `src/train_ml/trainer.py` (`MLTrainer(mlflow_config, model_config, seeds)` → `train(X_train, y_train, X_val, y_val) → MLTrainResult`); `src/train_dl/trainer.py` (`DLTrainer(mlflow_config, dl_config, seeds)` → `train(...) → DLTrainResult`); `src/evaluate/evaluator.py` (`ABEvaluator(mlflow_config, ab_test_config)` → `evaluate(X_test, y_test) → EvaluationResult`); `src/tune/tuner.py` (`HyperparamTuner(tune_config, winner)` → `tune(X_train, y_train, X_val, y_val) → TuneResult` — submits Katib Experiment and polls for completion); `src/register/registrar.py` (`ModelRegistrar(mlflow_config)` → `register(winner, run_id) → RegistryReceipt`)
- [ ] T061 [P] Create `src/metrics.py` with `make_eval_dataset(y_true, y_pred, feature_names=None)` that returns an `mlflow.data.from_numpy` evaluation dataset compatible with `mlflow.evaluate()`. No MLflow logging inside this module — pure data preparation.

**Checkpoint**: `dvc status` reports data/raw/CGR_Crash_Data.csv.dvc as tracked; `great_expectations/great_expectations.yml` exists; `.dvc/config` has local remote.

---

## Phase 3: User Story 1 — Reproducible DVC Pipeline (Priority: P1) 🎯 MVP

**Goal**: Define the full 7-stage DVC pipeline; implement ingest and featurize stages; verify end-to-end caching and reproducibility.

**Independent Test**: Run `dvc repro featurize` from a clean state — confirm all 6 arrays and `models/preprocessing_pipeline.joblib` are created; confirm `X_train.shape[1]` matches `n_features_selected` logged in MLflow. Change `feature_selection.method` from `none` to `mutual_info` in `params.yaml` — confirm only featurize re-runs and output array width changes. Run again with no changes — confirm all stages cached.

### Implementation for User Story 1

- [ ] T050 [US1] **RED** — Write `tests/test_ingest.py`: assert that given a valid CSV at `INPUT_PATH`, the stage exits 0 and produces a CSV at `OUTPUT_PATH` with the same row count; assert that given a missing `INPUT_PATH`, the stage exits 1 without writing any output. Run — confirm both tests FAIL (no implementation yet).
- [ ] T011 [US1] **GREEN** — Create `src/ingest/run.py` (thin entry point): load config via `src/config.py`; read `INPUT_PATH` and `OUTPUT_PATH` env vars with defaults from config; `os.makedirs` for output dir; instantiate `Ingester(input_path, output_path)` and call `.run()`; print row count; catch `FileNotFoundError` → exit 1; exit 0 on success
- [ ] T051 [US1] **RED** — Write `tests/test_featurize.py`: assert stage exits 0 and writes all 6 arrays + `preprocessing_pipeline.joblib`; assert `X_train` has no NaN values; assert `y_train` contains only 0 and 1; assert split sizes match `train_size`/`val_size`/`test_size` from params (±1 row); assert the fitted `ColumnTransformer` has three groups named `num`, `cat`, `ord`; assert `ord` group encodes DAYOFWEEK as integer 0 for Monday and 6 for Sunday; when `feature_selection.method` is `mutual_info` with `n_features=10`, assert `X_train.shape[1] == 10` and MLflow run logs both `n_features_raw` and `n_features_selected`; when method is `none`, assert `n_features_raw == n_features_selected`; assert MLflow run has `samples_per_param_ratio` ≥ 3.0 (computed on selected feature count); assert stage exits 1 when `ratio < 3.0`. Run — confirm tests FAIL.
- [ ] T012 [US1] **GREEN** — Create `src/featurize/run.py` (thin entry point): load config; read `INPUT_PATH`, `OUTPUT_DIR`, `PIPELINE_PATH` env vars; read CSV; compute `drop_pct` → exit 1 if > 5%; construct `FeatureSelector(config.feature_selection.method, n_features, threshold)` (or `None` if method is `none`); instantiate `Featurizer(..., feature_selector=selector)`; call `.fit_transform(df)`; save 6 arrays and pipeline+selector to `PIPELINE_PATH`; open MLflow run and log `n_samples`, `n_features_raw`, `n_features_selected`, `method`, `mlp_n_params`, `samples_per_param_ratio`; exit 1 if ratio < 3.0. All encoding, selection, and split logic lives in `Featurizer`/`FeatureSelector` — run.py only handles I/O and MLflow logging
- [ ] T013 [US1] Create `dvc.yaml` at repo root defining all 7 stages with correct `cmd`, `deps`, `outs`, and `params` entries — stage DAG: ingest→validate→featurize→(train_ml, train_dl)→evaluate→register; use stub cmds for validate/train_ml/train_dl/evaluate/register (e.g., `python -m src.validate.run`) so DVC graph is complete from the start
- [ ] T014 [US1] Run `dvc repro ingest featurize` — confirm all numpy arrays and joblib file created; run `dvc status` — confirm all outputs cached
- [ ] T015 [US1] Verify pipeline caching: run `dvc repro` a second time with no changes — confirm output `Skipped. Stage is cached.` for all completed stages
- [ ] T016 [P] [US1] Verify parameter-triggered re-run: change `data.val_size` from 0.15 to 0.20 in `params.yaml`, run `dvc repro featurize` — confirm only featurize re-executes and all 6 output arrays are regenerated; revert params after test
- [ ] T017 [P] [US1] Run `dvc push` to sync cached artifacts to `data/dvc-remote/`; confirm the remote directory contains the cached files

**Checkpoint**: US1 complete — DVC pipeline skeleton defined, ingest+featurize working, caching verified, artifacts pushed to remote.

---

## Phase 3b: Data Contract Definition (prerequisite for Phase 4)

**Purpose**: Define column-level data requirements collaboratively from EDA findings before writing any GE expectations.

- [ ] T058 Create `docs/data_contract.md`: for each column in `config.features.columns` plus CRASHSEVER, document: expected dtype, valid range or allowed value set, acceptable null rate (%), and sentinel values if any. Base ranges on `docs/eda_findings.md`. This is a collaborative document — domain knowledge drives the values, not engineering assumptions.
- [ ] T059 [P] Encode the data contract into `params.yaml` under a `validation` section: one entry per column with keys `dtype`, `min`/`max` or `allowed_values`, `mostly` (null tolerance 0.0–1.0). Update `src/config.py` `ValidationConfig` dataclass to expose these values typed. This is the machine-readable version of `docs/data_contract.md`.

**Checkpoint**: `docs/data_contract.md` exists and covers all feature columns; `params.yaml` has a `validation` section; `src/config.py` exposes typed validation config.

---

## Phase 4: User Story 2 — Automated Data Quality Validation (Priority: P2)

**Goal**: Define the GE expectation suite for the crash dataset; implement the validate stage; verify that bad data halts the pipeline.

**Independent Test**: Run `dvc repro validate` on clean data — confirm `great_expectations/uncommitted/data_docs/index.html` is created. Corrupt the raw CSV (add a row with SPEEDLIMIT=999), re-run — confirm non-zero exit and no downstream stages execute.

### Implementation for User Story 2

- [ ] T052 [US2] **RED** — Write `tests/test_validate.py`: assert that given a clean CSV the stage exits 0 and `great_expectations/gx/uncommitted/data_docs/index.html` is created; assert that given a CSV with a value violating the `params.yaml` validation contract (e.g. SPEEDLIMIT outside allowed range) the stage exits 1 and stdout contains the violated expectation name. Run — confirm tests FAIL.
- [ ] T018 [US2] **GREEN** — Implement `DataValidator.validate(df)` in `src/validate/validator.py` (replacing NotImplementedError): load GE v1 context from `self._ge_context_root`; programmatically build expectation suite from `self._validation_config` — one expectation per column using dtype, min/max or allowed_values, and mostly threshold (do NOT hardcode); run suite; call `context.build_data_docs()`; return `ValidationResult(success, failed_expectations, data_docs_path)`. Then create `src/validate/run.py` (thin entry point): load config; read `INPUT_PATH`; instantiate `DataValidator(ge_context_root, config.validation.columns)`; call `.validate(df)`; print summary; exit 1 if `not result.success`
- [ ] T019 [US2] Run `dvc repro validate` on the real crash CSV — confirm exit 0 and Data Docs HTML generated at `great_expectations/gx/uncommitted/data_docs/index.html`; verify all column expectations are present and match the data contract
- [ ] T020 [US2] Run `dvc repro validate` on the real crash CSV — confirm exit 0 and Data Docs HTML generated at `great_expectations/uncommitted/data_docs/index.html`; open the HTML and verify all 5 expectation categories show results
- [ ] T021 [US2] Test validation failure path: temporarily add a row with SPEEDLIMIT=500 to `data/processed/raw.csv`; run `python -m src.validate.run` directly — confirm exit code 1 and the violated expectation is named in stdout; restore original CSV
- [ ] T022 [P] [US2] Commit `great_expectations/expectations/crash_data_suite.json` and `great_expectations/great_expectations.yml` to git (these are the committed suite definition — `uncommitted/` stays in .gitignore)

**Checkpoint**: US2 complete — GE suite validates 5 categories, validate stage integrated in DVC pipeline, failure path confirmed.

---

## Phase 5: User Story 3 — ML vs DL Statistical A/B Test (Priority: P3)

**Goal**: Train PyCaret ML and PyTorch MLP each across 10 seeds; run Welch's t-test on the macro F1 distributions; register the winner.

**Independent Test**: Run `dvc repro train_ml train_dl evaluate register`. Open MLflow UI — confirm exactly 10 runs per experiment tagged with seeds 0–9. Check `docs/ab_test_comparison.json` — confirm it contains `p_value`, `cohens_d`, `ci_ml`, `ci_dl`, `winner`, `significant`, and `gates_passed`. Run `mlflow.pyfunc.load_model("models:/crash-severity@champion")` — confirm loads in < 30s.

### Implementation for User Story 3 — PyCaret ML Training

- [ ] T053 [US3] **RED** — Write `tests/test_train_ml.py`: assert that after the stage runs with `ab_test.seeds=[0]`, exactly 1 MLflow run exists in `crash-severity-ml` tagged `seed=0` with metrics `eout_macro_f1`, `eout_minority_recall`, `ein_macro_f1`, `generalisation_gap`; assert no autolog runs exist; assert `models/best_ml_model.pkl` exists. Run — confirm tests FAIL.
- [ ] T023 [US3] **GREEN** — Implement `MLTrainer.train(X_train, y_train, X_val, y_val)` in `src/train_ml/trainer.py`: `mlflow.sklearn.autolog(disable=True)`; loop over `self._seeds`; start MLflow run with tags `seed=<seed>`, `model_type=pycaret-ml`; PyCaret `setup(...)` with class weights and `session_id=seed`; `compare_models(n_select=config.model.n_select, sort="F1")`; `tune_model(best)`; call `mlflow.evaluate()` with `make_eval_dataset()`; log `generalisation_gap`; track best run_id; return `MLTrainResult`. Then create `src/train_ml/run.py` (thin entry point): load config + arrays; instantiate `MLTrainer`; call `.train()`; save best model; exit 0
- [ ] T024 [US3] Run `dvc repro train_ml` — confirm exactly 10 MLflow runs appear in `crash-severity-ml` experiment, each with `ein_macro_f1`, `eout_macro_f1`, `generalisation_gap` and tag `seed` visible in MLflow UI

### Implementation for User Story 3 — PyTorch DL Training

- [ ] T054 [US3] **RED** — Write `tests/test_train_dl.py`: assert that after the stage runs with `ab_test.seeds=[0]`, exactly 1 MLflow run exists in `crash-severity-dl` tagged `seed=0` with `eout_macro_f1` and `eout_minority_recall` logged via `mlflow.evaluate()`; assert per-epoch `ein_loss` history has ≥1 entry; assert `models/mlp_model.pth` is a valid torch checkpoint loadable with `torch.load`. Run — confirm tests FAIL.
- [ ] T062 [US3] **RED** — Write `tests/test_pyfunc.py`: instantiate `ShallowMLPWrapper` from `src/train_dl/pyfunc.py` with a saved checkpoint; call `predict(context, pd.DataFrame(X_test[:5]))` — assert output is a numpy array of 0s and 1s with shape `(5,)`. Run — confirm test FAILS.
- [ ] T025 [US3] **GREEN** — Create `src/train_dl/pyfunc.py`: define `ShallowMLPWrapper(mlflow.pyfunc.PythonModel)` with `load_context(self, context)` loading `ShallowMLP` state dict and `input_dim` from `context.artifacts["model_path"]`; `predict(self, context, model_input)` accepting a pandas DataFrame, running forward pass, applying sigmoid + threshold 0.5, returning binary numpy array.
- [ ] T025b [US3] **GREEN** — Implement `DLTrainer.train(X_train, y_train, X_val, y_val)` in `src/train_dl/trainer.py`: define `ShallowMLP(nn.Module)` with `__init__(input_dim, dropout)` constructing Linear(input_dim,128)→BatchNorm1d→ReLU→Dropout(dropout)→Linear(128,64)→BatchNorm1d→ReLU→Dropout(dropout)→Linear(64,1); loop over seeds; `BCEWithLogitsLoss(pos_weight=[2.74])`; `Adam(lr=config.dl.lr, weight_decay=config.dl.weight_decay)` — L2 regularisation via Adam's weight_decay (complements Dropout which prevents co-adaptation); early stopping on val loss; log per-epoch `ein_loss`, `eout_loss`, `gap_f1`; return `DLTrainResult`. Then create `src/train_dl/run.py` (thin entry point): load config + arrays; instantiate `DLTrainer`; call `.train()`; save best checkpoint; exit 0
- [ ] T026 [US3] Implement training loop in `src/train_dl/run.py`: `train_one_epoch(model, optimizer, loader, criterion)` function returning avg loss; `evaluate_model(model, X, y)` returning `(loss, macro_f1)` using threshold 0.5 on sigmoid output; early stopping class tracking best val loss with patience from params
- [ ] T027 [US3] Implement seed loop in `src/train_dl/run.py`: for each seed in `params.ab_test.seeds`: set `torch.manual_seed(seed)`, `numpy.random.seed(seed)`; build TensorDatasets and DataLoaders (train batch=256, test batch=512); instantiate `ShallowMLP(input_dim=X_train.shape[1])`; `BCEWithLogitsLoss(pos_weight=tensor([2.74]))` for training, unweighted for eval; Adam(lr=params.dl.lr); run training loop with early stopping; start MLflow run with tags `seed=<seed>`, `model_type=pytorch-mlp`, `architecture=128-64-1-dropout0.3`; log per-epoch `ein_loss`, `eout_loss`, `gap_f1` with `step=epoch`; log final `eout_macro_f1`; end run; track best seed; after loop, save best-seed `model.state_dict()` + `input_dim` to `MODEL_OUTPUT_PATH` via `torch.save`
- [ ] T028 [US3] Run `dvc repro train_dl` — confirm 10 MLflow runs in `crash-severity-dl` with per-epoch metrics visible as line charts in MLflow UI; verify early stopping fired before epoch 100 for at least one seed

### Implementation for User Story 3 — Statistical A/B Test & Evaluation

- [ ] T055 [US3] **RED** — Write `tests/test_evaluate.py`: assert that given mocked MLflow runs (N=3 seeds each experiment), the stage exits 0 and `docs/ab_test_comparison.json` contains keys `p_value`, `cohens_d`, `ci_ml`, `ci_dl`, `winner`, `significant`, `gates_passed`; assert that when all scores are below 0.55 the stage exits 1. Run — confirm tests FAIL.
- [ ] T029 [US3] **GREEN** — Implement `ABEvaluator.evaluate(X_test, y_test)` in `src/evaluate/evaluator.py`: query MLflow for per-seed `eout_macro_f1` scores from both experiments; Welch's t-test; Cohen's d; 95% CIs; declare winner (higher mean if p < alpha, tiebreak from config); assert constitutional gates (F1 > 0.55, recall > 0.40); return `EvaluationResult`. Then create `src/evaluate/run.py` (thin entry point): load config + arrays; instantiate `ABEvaluator`; call `.evaluate()`; write `REPORT_PATH` and `AB_REPORT_PATH` JSONs; log artifact to MLflow; exit 1 if gates fail
- [ ] T030 [US3] Run `dvc repro evaluate` — confirm `docs/ab_test_comparison.json` contains `p_value`, `cohens_d`, `ci_ml`, `ci_dl`, `winner`, `significant`, `gates_passed: true`; print the winner and margin to stdout

### Implementation for User Story 3 — Model Registration

- [ ] T056 [US3] **RED** — Write `tests/test_register.py`: assert that given a `REPORT_PATH` JSON with `gates_passed=true` and a valid MLflow run, the stage exits 0, `models:/crash-severity@champion` is resolvable, and `models/registry_receipt.json` exists containing `model_name`, `version`, and `alias`; assert that given `gates_passed=false` the stage exits 1 immediately without touching the registry or writing the receipt. Run — confirm tests FAIL.
- [ ] T031 [US3] **GREEN** — Implement `ModelRegistrar.register(winner, run_id)` in `src/register/registrar.py`: call `mlflow.register_model()`; set `@champion` alias; return `RegistryReceipt`. Then create `src/register/run.py` (thin entry point): load config; read `REPORT_PATH` → exit 1 if `gates_passed` is false; read `winner` and `best_run_id`; instantiate `ModelRegistrar(config.mlflow)`; call `.register()`; write `models/registry_receipt.json`; exit 0
- [ ] T032 [US3] Run `dvc repro register` — confirm `models:/crash-severity@champion` exists; verify `mlflow.pyfunc.load_model("models:/crash-severity@champion")` loads without error and `model.predict(X_test[:5])` returns predictions

**Checkpoint**: US3 complete — 10 ML runs + 10 DL runs in MLflow, Welch's t-test completed.

---

## Phase 5b: User Story 3b — Hyperparameter Optimisation with Optuna (Priority: P3b)

**Goal**: Run Optuna Bayesian HPO on the winning model family; log every trial to MLflow; write best params to `params.yaml`; register the tuned champion.

**Independent Test**: Run `dvc repro tune` — confirm N runs appear in `crash-severity-tune` MLflow experiment each tagged with `trial=<n>` and hyperparameter values; confirm `params.yaml` updated under `tune.best_params`; confirm best trial metric ≥ pre-tune winner metric from A/B test.

### Implementation for User Story 3b

- [ ] T002b [P] Add `kubernetes>=28.0` (Python client) to `pyproject.toml` for Katib Experiment submission — then run `uv sync`. No Optuna dependency; HPO is handled entirely by Katib in-cluster.
- [ ] T063 Add `tune` params section to `params.yaml`: `tune.katib_namespace` (default `kubeflow`), `tune.katib_experiment_name` (`crash-severity-tune`), `tune.n_trials` (default 50), `tune.parallel_trials` (default 2), `tune.mlflow_experiment_name` (`crash-severity-tune`). Search space bounds live in Katib Experiment YAML manifests (T073/T074) — not in `params.yaml`. Update `src/config.py` with `TuneConfig(katib_namespace, katib_experiment_name, n_trials, parallel_trials, mlflow_experiment_name)`.
- [ ] T073 [P] Create `k8s/katib/ml_experiment.yaml`: Katib `Experiment` CRD defining the ML HPO search — objective metric `eout_macro_f1` (maximize), algorithm `bayesianoptimization`, `maxTrialCount: 50`, `parallelTrialCount: 2`; parameters covering architecture + regularization: `n_estimators` (int, 50–500), `max_depth` (int, 3–10), `learning_rate` (double, 0.01–0.3), `reg_lambda` (double, 1e-3–10.0), `reg_alpha` (double, 1e-3–1.0); trialTemplate using the pipeline Docker image with command `python -m src.tune.trial --winner=ml --<param>=<value>` for each trial; metrics collector reads `eout_macro_f1=<value>` from trial stdout.
- [ ] T074 [P] Create `k8s/katib/dl_experiment.yaml`: Katib `Experiment` CRD for DL HPO — same objective and algorithm; parameters: `lr` (double, 1e-4–1e-1), `dropout` (double, 0.1–0.5), `weight_decay` (double, 1e-5–1e-2), `hidden_dim` (int, 32–256); trialTemplate command `python -m src.tune.trial --winner=dl --<param>=<value>`.
- [ ] T075 [P] Create `src/tune/trial.py`: CLI entrypoint for a single Katib trial — accepts all hyperparams as `argparse` arguments plus `--winner ml|dl`; loads data arrays from PVC paths; if ML: runs PyCaret setup+compare+tune with sampled params, evaluates with `mlflow.evaluate()`; if DL: instantiates `ShallowMLP` with sampled dims and regularization, trains with early stopping, evaluates; logs all metrics to MLflow under `crash-severity-tune` experiment; prints `eout_macro_f1=<value>` to stdout on the last line (required by Katib stdout metrics collector). No env var config reading — all params come from CLI args.
- [ ] T064 **RED** — Write `tests/test_tune.py`: mock the Kubernetes client; assert that `HyperparamTuner.tune()` submits a Katib `Experiment` with the correct name and namespace; assert it polls until `experiment.status.completionTime` is set; assert it reads `status.currentOptimalTrial.parameterAssignments` and returns them as `TuneResult.best_params`; assert `params.yaml` contains `tune.best_params` after `run.py` completes. Run — confirm tests FAIL.
- [ ] T065 [US3b] **GREEN** — Implement `HyperparamTuner.tune(X_train, y_train, X_val, y_val)` in `src/tune/tuner.py`: load the appropriate Katib Experiment YAML (`ml_experiment.yaml` or `dl_experiment.yaml`) based on `self._winner`; submit via `kubernetes.client.CustomObjectsApi().create_namespaced_custom_object()`; poll `status.conditions` until `Succeeded` or `Failed`; on success read `status.currentOptimalTrial.parameterAssignments` as `best_params`; return `TuneResult(best_params, best_value, n_trials, best_run_id)`. Then create `src/tune/run.py` (thin entry point): load config + arrays; read winner from `REPORT_PATH`; instantiate `HyperparamTuner`; call `.tune()`; write `best_params` to `params.yaml` under `tune.best_params`; exit 0.
- [ ] T066 [US3b] Run `dvc repro tune` — confirm Katib Experiment appears in Katib UI (`http://localhost:8888` → Experiments (AutoML)); confirm all trials appear in MLflow under `crash-severity-tune` tagged with their hyperparams; confirm `params.yaml` updated with `tune.best_params`
- [ ] T067 [US3b] Run `dvc repro register` after tune — confirm register uses tuned model; confirm `models/registry_receipt.json` reflects new version

**Checkpoint**: US3b complete — Katib Experiment submitted and completed, best params written to params.yaml, all trials visible in both Katib UI and MLflow, tuned champion registered.

---

## Phase 6: User Story 4 — Local Workflow Orchestration via Airflow (Priority: P4)

**Goal**: Express the full pipeline as an Airflow TaskFlow DAG with parallel train steps; verify task-level monitoring and failure recovery.

**Independent Test**: Start Airflow standalone, trigger `crash_severity_pipeline` DAG manually, confirm all 7 tasks turn green. Mark the `validate` task failed manually, clear and restart it — confirm `ingest` and `featurize` are NOT re-run.

### Implementation for User Story 4

- [ ] T057 [US4] **RED** — Write `tests/test_airflow_dag.py`: import the DAG, assert it has exactly 8 tasks; assert dependency order `ingest → validate → featurize → train_ml → train_dl → evaluate → tune → register` (sequential); assert `retries=2` in `default_args`; assert each task command calls `dvc repro <stage>`. Run — confirm tests FAIL.
- [ ] T033 [US4] **GREEN** — Create `airflow/dags/crash_ml_pipeline.py` using TaskFlow API: define 8 tasks — each calls `subprocess.run(["dvc", "repro", "<stage>"], check=True, cwd=PROJECT_ROOT)`; set `default_args={"retries": 2, "retry_delay": timedelta(minutes=5)}`; set `schedule=None`, `catchup=False`, `tags=["mlops", "crash-severity"]`; wire sequentially: `ingest >> validate >> featurize >> train_ml >> train_dl >> evaluate >> tune >> register`
- [ ] T034 [US4] Start Airflow: `cd airflow && uv run airflow standalone`; trigger `crash_severity_pipeline` DAG from UI; confirm all 7 tasks complete successfully and logs are accessible per-task
- [ ] T035 [US4] Verify parallel execution: confirm `train_ml` and `train_dl` appear side-by-side in the Airflow Graph view with no dependency edge between them
- [ ] T036 [US4] Test failure recovery: in the Airflow UI, mark the `validate` task as failed on the last run; fix nothing (it should pass on retry); clear the task state; confirm only `validate` re-runs and `ingest`+`featurize` remain in success state

**Checkpoint**: US4 complete — Airflow DAG with parallel train tasks, task logs, retry config, and failure recovery working.

---

## Phase 7: User Story 5 — Container-Native Kubeflow Orchestration (Priority: P5)

**Goal**: Build a Docker image for the pipeline; install Kubeflow Pipelines standalone on Docker Desktop Kubernetes; express the pipeline as a KFP v2 workflow and run it.

**Independent Test**: Submit pipeline from KFP UI — confirm 7 steps appear with correct dependency arrows and train_ml/train_dl show as parallel. Inspect the `validate` pod logs — confirm GE validation output is visible. Check MLflow UI — confirm a new run appears with tag `orchestrator=kubeflow`.

### Implementation for User Story 5

- [ ] T037 [US5] Create `docker/Dockerfile`: `FROM python:3.12-slim`; `WORKDIR /app`; `COPY pyproject.toml uv.lock ./`; `RUN pip install uv && uv sync --frozen`; `COPY src/ ./src/`; `COPY dvc.yaml ./`; `COPY .dvcignore ./`; `COPY params.yaml ./` (local default — overridden by PVC mount at runtime); `ENV PYTHONPATH=/app`. Do NOT copy `mlruns/`, `data/`, `models/`, or `.dvc/cache/` — these come from the PVC mount at runtime.
- [ ] T038 [US5] Build Docker image: `docker build -f docker/Dockerfile -t mlops-portfolio:latest .`; smoke-test each stage: `docker run --rm -v $(pwd)/data:/app/data mlops-portfolio:latest python -m src.ingest.run`; confirm exit 0
- [ ] T039 [US5] Enable Kubernetes in Docker Desktop (Settings → Kubernetes → Enable Kubernetes → Apply & Restart); verify with `kubectl cluster-info`
- [ ] T039b [US5] Create `k8s/pvc.yaml`: a `PersistentVolume` and `PersistentVolumeClaim` using `hostPath` pointing to the project root (`C:\Users\loren\documents\mlops-portfolio`). Both PV and PVC use `storageClassName: manual`, `accessModes: [ReadWriteOnce]`, `capacity: 20Gi`. Apply with `kubectl apply -f k8s/pvc.yaml`. This single PVC provides all KFP pods access to `mlruns/`, `data/`, `models/`, `.dvc/cache/`, and `params.yaml` at the same paths as the host.
- [ ] T040 [US5] Install Kubeflow Pipelines standalone: `kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources/base?ref=2.2.0"`; wait for CRDs; `kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=2.2.0"`; wait for all pods ready: `kubectl -n kubeflow wait --for=condition=Ready pods --all --timeout=300s`
- [ ] T041 [US5] Create `pipelines/kubeflow/pipeline.py`: define 8 `@dsl.component(base_image="mlops-portfolio:latest")` functions — one per stage — each calling `subprocess.run(["dvc", "repro", "<stage>"], check=True, cwd="/app")`; mount the PVC from T039b on each component at `/app`; define `@dsl.pipeline(name="crash-severity-pipeline")` wiring all 8 stages sequentially: `ingest >> validate >> featurize >> train_ml >> train_dl >> evaluate >> tune >> register`; compile to `pipelines/kubeflow/pipeline.yaml`
- [ ] T042 [US5] Compile the KFP pipeline: `python pipelines/kubeflow/pipeline.py` — confirm `pipelines/kubeflow/pipeline.yaml` is created
- [ ] T043 [US5] Port-forward KFP UI: `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80`; open `http://localhost:8888`; upload `pipeline.yaml`; create and start a run; confirm all 7 steps appear with correct dependency graph and parallel train steps
- [ ] T044 [US5] Inspect the `validate` step pod logs in KFP UI — confirm GE validation output is visible; inspect the `train_ml` step — confirm MLflow run appears in UI tagged `orchestrator=kubeflow`

**Checkpoint**: US5 complete — pipeline compiled, deployed, and running on Docker Desktop Kubernetes.

---

## Phase 8: Polish & Validation

**Purpose**: Assert all constitutional gates, commit final state, verify full reproducibility.

- [ ] T045 [P] Assert constitutional gates in isolation: run `python -c "import json; r=json.load(open('docs/evaluation_report.json')); print('F1 gate:', 'PASS' if r['winner_macro_f1']>0.55 else 'FAIL'); print('Recall gate:', 'PASS' if r['winner_minority_recall']>0.40 else 'FAIL')"`
- [ ] T046 [P] Update `CLAUDE.md` architecture section: replace old notebook-based entries with new `src/`, `dvc.yaml`, `params.yaml`, `great_expectations/`, `pipelines/kubeflow/`, `docker/` entries
- [ ] T047 [P] Create `.gitignore` additions: `mlruns/`, `data/processed/`, `models/`, `great_expectations/uncommitted/`, `data/dvc-remote/`, `pipelines/kubeflow/pipeline.yaml`
- [ ] T048 Commit all final tracked files: `dvc.yaml`, `params.yaml`, `src/`, `great_expectations/expectations/`, `great_expectations/great_expectations.yml`, `airflow/dags/crash_ml_pipeline.py`, `pipelines/kubeflow/pipeline.py`, `docker/Dockerfile`; run `dvc push` to sync all artifacts to remote
- [ ] T049 [P] Full reproducibility smoke test: delete `data/processed/` and `models/`; run `dvc pull && dvc repro`; confirm all artifacts are restored and pipeline completes end-to-end

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
  └── Phase 2 (Foundational — DVC init + GE init) — BLOCKS all user stories
        ├── Phase 3 (US1: DVC Pipeline + Ingest + Featurize)   ← start here for MVP
        │     └── Phase 4 (US2: GE Validation)
        │           └── Phase 5 (US3: ML + DL Training + A/B Test + Register)
        │                 ├── Phase 6 (US4: Airflow DAG)
        │                 └── Phase 7 (US5: Kubeflow + Docker)
        └── Phase 8 (Polish) — after all stories complete
```

### Stage Execution Order (within DVC pipeline)

| Order | Stage | Parallel with | Blocks |
|---|---|---|---|
| 1 | ingest | — | validate |
| 2 | validate | — | featurize |
| 3 | featurize | — | train_ml, train_dl |
| 4a | train_ml | train_dl | evaluate |
| 4b | train_dl | train_ml | evaluate |
| 5 | evaluate | — | tune |
| 6 | tune | — | register |
| 7 | register | — | — |

### Parallel Opportunities

- T002, T003, T004, T005 (Phase 1 setup) — all parallel
- T016, T017 (US1 verification tasks) — parallel
- T022, T025 (US2 commit) — parallel
- T023/T024 (train_ml implementation + repro) — sequential within ML track
- T025–T028 (train_dl implementation + repro) — sequential within DL track
- **train_ml track and train_dl track are fully parallel after T014**
- T033–T035 (Airflow verify tasks) — sequential (need DAG running)
- T045, T046, T047 (Polish) — all parallel

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) + Phase 2 (Foundational)
2. Complete Phase 3 (US1: DVC pipeline + ingest + featurize)
3. **STOP and VALIDATE**: `dvc repro featurize` → arrays exist; `dvc status` → all cached
4. This alone demonstrates DVC pipeline versioning and artifact caching

### Incremental Delivery

1. Phase 1 + 2 → environment ready
2. Phase 3 → DVC pipeline skeleton + featurize working (MVP ✅)
3. Phase 4 → data validation gate working
4. Phase 5 → full A/B test + champion model registered
5. Phase 6 → Airflow orchestration
6. Phase 7 → Kubeflow container-native orchestration
7. Phase 8 → final commit and reproducibility verification

### Parallel Team Strategy

After Phase 2 completes:
- Track A: US1 (DVC) → US2 (GE) → US3 (ML/DL/A/B)
- Track B (once Docker image exists in T037–T038): US5 (Kubeflow)
- Track C (once US3 complete): US4 (Airflow)

---

## Notes

- `train_ml` and `train_dl` run **sequentially within their DVC stage** (N seeds in a loop) but the two stages themselves are **parallel in the DVC DAG**
- PyCaret's `compare_models()` is slow (~5–10 min per seed) — expect `train_ml` to take 50–100 min total for N=10 seeds. Consider reducing `params.ab_test.seeds` to `[0,1,2]` during development
- `MLFLOW_TRACKING_URI` inside a Kubeflow pod must point to a volume-mounted path or a networked server — local `mlruns/` is not accessible inside the pod without a PVC mount
- The `register` stage writes only to the MLflow Model Registry (no filesystem output) — DVC does not track the registry; this stage has no `outs` in `dvc.yaml`
- Kubeflow Pipelines standalone install takes ~5 minutes; pods need ~4GB RAM — close other Docker workloads before starting
