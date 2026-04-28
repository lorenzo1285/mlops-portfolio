# Tasks: MLOps Learning Portfolio — VAE-Based Crash Severity Pipeline

**Input**: Design documents from `specs/002-mlops-portfolio/`
**Prerequisites**: spec.md ✅ | plan.md ✅ | research.md ✅ | data-model.md ✅ | contracts/stage-interface.md ✅
**Architecture**: VAE-based, 10-stage pipeline (validate→ingest→featurize→train_vae→encode→train_ml→train_dl→evaluate→tune→register)
**Constitution**: v3.1.0 — TDD for all `src/` (XV); boundary tests only (XIV); GE exclusive QA layer (XVI)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story (US1–US6 mapping to spec.md priorities P1–P6)
- All file paths relative to repo root

---

## Phase 1: Setup

**Purpose**: Scaffold directories, install dependencies, create shared config.

- [x] T001 Create all required directories: `src/ingest/`, `src/validate/`, `src/featurize/`, `src/train_ml/`, `src/train_dl/`, `src/evaluate/`, `src/tune/`, `src/register/`, `pipelines/kubeflow/`, `docker/`, `models/`, `docs/`, `data/processed/`, `data/dvc-remote/`
- [x] T002 [P] Add dependencies to `pyproject.toml`: `dvc>=3.0`, `great-expectations>=1.0`, `kfp>=2.0`, `scipy>=1.11`, `xgboost>=2.0`, `torch>=2.0` — run `uv sync`
- [x] T003 [P] Create initial `params.yaml` with `features.*`, `data.*`, `model.*`, `dl.*`, `mlflow.*`, `great_expectations.*`, `ab_test.*`, `feature_selection.*`, `tune.*` sections
- [x] T004 [P] Create `.dvcignore` excluding `mlruns/`, `**/__pycache__/`, `**/*.pyc`, `.venv/`, `data/dvc-remote/`
- [x] T005 [P] Create `src/__init__.py` and `src/<stage>/__init__.py` for ingest, validate, featurize, train_ml, train_dl, evaluate, tune, register
- [x] T006 [P] Create `.gitattributes`: `*.py text eol=lf`, `*.sh text eol=lf`

**Checkpoint**: Directories exist, dependencies installed, initial params.yaml in place. ✅

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: DVC init, GE init, typed config, stage stubs — everything else depends on these.

⚠️ **CRITICAL**: No user story work can begin until this phase is complete.

- [x] T007 Run `dvc init` at repo root; creates `.dvc/` directory
- [x] T008 Configure DVC local remote: `dvc remote add -d local data/dvc-remote`
- [x] T009 Track raw dataset: `dvc add data/raw/CGR_Crash_Data.csv`; commit pointer file
- [x] T010 Initialise GE v1 context: `python -c "import great_expectations as gx; gx.get_context(mode='file', project_root_dir='great_expectations/gx')"`
- [x] T011 [P] Create `src/config.py` with typed dataclass accessors: `FeaturesConfig`, `DataConfig`, `ModelConfig`, `DLConfig`, `MLflowConfig`, `ABTestConfig`, `ValidationConfig`, `FeatureSelectionConfig`, `TuneConfig`; top-level `load_config(path=None)` reads `PARAMS_PATH` env var
- [x] T012 [P] Create `src/ingest/ingester.py`: `IngestResult` dataclass; `Ingester(input_path, output_path)` class with `run() → IngestResult`
- [x] T013 [P] Add `feature_selection` section to `params.yaml`; add `FeatureSelectionConfig` to `src/config.py`
- [x] T014 [P] Create `src/featurize/selector.py`: `SelectionResult` dataclass; `FeatureSelector(method, n_features, threshold)` with `fit()` and `transform()`
- [x] T015 [P] Create `src/featurize/featurizer.py`: `FeaturizeResult` dataclass; `Featurizer(...)` with `fit_transform(df) → FeaturizeResult`
- [x] T016 [P] Create stub class modules for validate, train_ml, train_dl, evaluate, tune, register (constructor + `NotImplementedError`)
- [x] T017 [P] Create `src/metrics.py` with `make_eval_dataset(y_true, y_pred, feature_names=None)` helper

### Phase 2 Extension: VAE Architecture Scaffolding

- [x] T081 [P] Create `src/train_vae/` and `src/encode/` package directories with `__init__.py`; add `src/train_vae/__init__.py` and `src/encode/__init__.py`
- [x] T082 [P] Extend `params.yaml`: add `vae` section (`encoder_dims: [256,128,64]`, `latent_dim: 32`, `beta: 1.0`, `dropout_p: 0.15`, `epochs: 200`, `patience: 20`, `batch_size: 512`, `lr: 0.001`, `experiment_name: crash-severity-vae`); add `encode` section (`lsa_target_ratio: 0.05`, `min_fatal_samples: 10`); update `model` section (`n_classes: 3`, `macro_f1_threshold: 0.45`, `fatal_recall_threshold: 0.30` — remove `class_weight_neg`/`class_weight_pos`); update `dl` section (`input_dim: 32`, `hidden_dim: 64` — remove `hidden_1`/`hidden_2`); add `experiment_name_vae` + `experiment_name_tune` to `mlflow` section
- [x] T083 [P] Extend `src/config.py`: add `VAEConfig(encoder_dims, latent_dim, beta, dropout_p, epochs, patience, batch_size, lr, experiment_name)` dataclass; add `EncodeConfig(lsa_target_ratio, min_fatal_samples)` dataclass; update `ModelConfig` (`n_classes`, `macro_f1_threshold`, `fatal_recall_threshold` — remove binary weight fields); update `DLConfig` (`input_dim`, `hidden_dim` — remove `hidden_1`/`hidden_2`; remove `weight_decay` from DLConfig; remove EvoTorch fields); add `VAEConfig` and `EncodeConfig` to `ProjectConfig`
- [x] T084 [P] Extend `src/metrics.py`: add `per_class_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) → dict` that returns precision, recall, F1 per class as a JSON-serialisable dict. Add `compute_class_weights(y: np.ndarray, n_classes: int) → np.ndarray` that computes `w_c = N / (n_classes × class_count_c)`
- [x] T085 [P] Create stub class modules for new stages (constructor + `NotImplementedError` body): `src/train_vae/vae_trainer.py` — `VAETrainResult` dataclass; `DVAETrainer(vae_config, mlflow_config)` with `train(X_all: np.ndarray) → VAETrainResult`; `src/encode/encoder.py` — `EncodeResult` dataclass; `LatentEncoder(encoder_path, encode_config, latent_dim)` with `encode(X_train, y_train, X_val, y_val, X_test) → EncodeResult`

**Checkpoint**: New packages exist; params.yaml has vae.* and encode.* sections; config.py exposes `VAEConfig`/`EncodeConfig`; metrics.py has `per_class_matrix` and `compute_class_weights`.

---

## Phase 3A: Ingest + Featurize TDD (US1) 🎯 MVP

**Goal**: Ingest and featurize stages implemented and tested; 3-class target encoding confirmed.

**Independent Test**: `INPUT_PATH=data/raw/CGR_Crash_Data.csv OUTPUT_PATH=data/processed/raw.csv uv run python -m src.ingest.run` → exit 0. Then `INPUT_PATH=data/processed/raw.csv uv run python -m src.featurize.run` → exit 0; `y_train` contains values 0, 1, 2 only.

- [x] T018 [US1] **RED** — Write `tests/test_ingest.py`: assert stage exits 0 and produces output CSV with same row count; assert missing `INPUT_PATH` exits 1. Run — confirm FAIL.
- [x] T019 [US1] **GREEN** — Create `src/ingest/run.py`: load config; read `INPUT_PATH`/`OUTPUT_PATH` env vars; call `Ingester.run()`; exit 0/1.
- [x] T020 [US1] **RED** — Update `tests/test_featurize.py`: use `data/processed/raw.csv` as fixture (constitution XVI — NOT `data/raw/`); assert stage exits 0 and writes 6 arrays + `preprocessing_pipeline.joblib`; assert `X_train` has no NaN; **assert `y_train` contains only values `{0, 1, 2}`** (PDO=0, Injury=1, Fatal=2 — not binary); assert CRASHSEVER `"Fatal"` rows map to label `2`; assert split sizes match params (±1 row); assert `ColumnTransformer` has `num`/`cat`/`ord` groups; assert `samples_per_param_ratio` logged to MLflow; assert stage exits 1 when ratio < 3.0. Run — confirm FAIL.
- [x] T021 [US1] **GREEN** — Update `src/featurize/featurizer.py`: `_separate_target` method maps CRASHSEVER string → int: `"Property Damage Only" → 0`, `"Injury" → 1`, `"Fatal" → 2`; store mapping as class-level constant (not in params.yaml — it is a dataset invariant, not a hyperparameter). Update `src/featurize/run.py` to match: 3-class target; log `n_classes=3` to MLflow.

**Checkpoint**: Both stages exit 0; `y_train/y_val/y_test` contain values 0/1/2; `samples_per_param_ratio` logged. ✅ when T021 green.

---

## Phase 3b: Data Contract Definition

- [x] T027 Create `docs/data_contract.md`: document dtype, valid range/values, null rate, sentinels for all feature columns + CRASHSEVER
- [x] T028 [P] Encode into `params.yaml validation.*`; update `ValidationConfig` in `src/config.py`

**Checkpoint**: Contract defined; `params.yaml` has `validation` section. ✅

---

## Phase 3B: GE Validate Stage (US2)

**Goal**: GE three-class workflow implemented; validate stage exits 0 on clean data and exits 1 + names violated expectation on bad data.

**Independent Test**: `uv run python -m src.validate.run` → exit 0, HTML Data Docs created. Inject SPEEDLIMIT=500 → exit 1, expectation name in stdout.

- [x] T029a [US2] Refactor `GEContextBuilder` to infrastructure only (datasource + empty asset, no suite logic); add `sentinel_values: list[Any] | None` to `ColumnContract`; rewrite `GEManager.build_suite()` to use `row_condition` per sentinel for range checks (no `_RANGE_MOSTLY`); remove `_RANGE_MOSTLY` entirely
- [x] T029 [US2] **RED** — Write `tests/test_validate.py`: use `data/raw/CGR_Crash_Data.csv` as fixture (validate runs on raw data — constitution XVI); assert exit 0 + Data Docs HTML created on clean CSV; assert exit 1 + violated expectation in stdout on corrupt CSV. Run — confirm FAIL.
- [x] T029b [US2] **GREEN** — Create `great_expectations/gx/utils/ge_checkpoint_runner.py`: `CheckpointRunResult` dataclass; `GECheckpointRunner` with `run(df) → CheckpointRunResult` using `UpdateDataDocsAction` + `StoreValidationResultAction`
- [x] T030 [US2] **GREEN** — Implement `DataValidator.validate(df)` in `src/validate/validator.py` (three-class GE workflow: Builder → Manager.build_suite → Manager preparation → CheckpointRunner.run); create `src/validate/run.py`: reads `INPUT_PATH`, calls `DataValidator`, writes sentinel `data/processed/.validation_passed` on exit 0

**Checkpoint**: validate exits 0/1 correctly; Data Docs HTML created; sentinel written on success.

---

## Phase 3C: DVC Pipeline Integration (US1 + US2)

**Goal**: 10-stage `dvc.yaml` wired; caching and parameter-triggered re-runs verified.

**Independent Test**: `dvc repro featurize` → all 6 arrays + joblib written; `dvc status` → all cached. Change a param → only downstream stages re-run.

- [x] T022 [US1+US2] Create `dvc.yaml` at repo root with all 10 stages and correct `cmd`/`deps`/`outs`/`params`:
  - `validate` → `outs: [data/processed/.validation_passed]`; `deps: [data/raw/CGR_Crash_Data.csv]`
  - `ingest` → `deps: [data/processed/.validation_passed]`; `outs: [data/processed/raw.csv]`
  - `featurize` → `deps: [data/processed/raw.csv]`; `outs: [X_train, X_val, X_test, y_train, y_val, y_test, preprocessing_pipeline.joblib]`; `params: [features.*, data.*, feature_selection.*]`
  - `train_vae` → `deps: [X_train, X_val, X_test]`; `params: [vae.*]`; `outs: [models/vae_encoder.pth, models/vae_decoder.pth]`
  - `encode` → `deps: [models/vae_encoder.pth, X_train, y_train, X_val, X_test]`; `params: [encode.*, vae.latent_dim]`; `outs: [Z_train_augmented.npy, Z_val.npy, Z_test.npy, y_train_augmented.npy]`
  - `train_ml` → `deps: [Z_train_augmented.npy, y_train_augmented.npy, Z_val.npy, y_val.npy, Z_test.npy, y_test.npy]`; `params: [model.*, ab_test.*]`; `outs: [models/best_ml_model.pkl]`
  - `train_dl` → same deps as train_ml; `params: [dl.*, ab_test.*]`; `outs: [models/mlp_model.pth]`
  - `evaluate` → `deps: [models/best_ml_model.pkl, models/mlp_model.pth, Z_test.npy, y_test.npy]`; `outs: [docs/evaluation_report.json, docs/ab_test_comparison.json]`
  - `tune` → `deps: [docs/evaluation_report.json]`; `outs: none` (writes to params.yaml)
  - `register` → `deps: [docs/evaluation_report.json]`; `outs: [models/registry_receipt.json]`
  - Use stub cmds for incomplete stages so the full 10-stage DAG is defined from the start
- [x] T023 [US1] Run `dvc repro featurize` — confirm sentinel created, then all numpy arrays and joblib written; `dvc status` → all cached
- [x] T024 [US1] Verify caching: run `dvc repro` again with no changes — all stages report `Skipped. Stage is cached.`
- [x] T025 [P] [US1] Verify param-triggered re-run: change `data.val_size` in `params.yaml` → `dvc repro featurize` re-runs only featurize; revert
- [x] T026 [P] [US1] Run `dvc push` — confirm artifacts synced to `data/dvc-remote/`
- [x] T031 [US2] Run `dvc repro validate` on clean crash CSV — confirm exit 0 and Data Docs HTML at `great_expectations/gx/uncommitted/data_docs/local_site/index.html` (actual GE v1 path)
- [x] T032 [US2] Open Data Docs HTML — verify not-null, range, and value-set expectations all appear with pass/fail counts (54/54 passed, 100% success rate)
- [x] T033 [US2] Test failure path: add row with SPEEDLIMIT=500; `python -m src.validate.run` → confirm exit 1 and expectation name in stdout; restore CSV
- [x] T034 [P] [US2] Commit `great_expectations/gx/expectations/crash_data_suite.json` and `great_expectations.yml` to git

**Checkpoint**: 10-stage DVC DAG defined; validate + ingest + featurize fully wired and cached; failure path halts pipeline; GE suite committed.

---

## Phase 3D: VAE Training Stage (US3) 🔑 New

**Goal**: Denoising β-VAE trains unsupervised on full X; ELBO curve converges and is visible in MLflow `crash-severity-vae`.

**Independent Test**: `dvc repro train_vae` → `models/vae_encoder.pth` + `models/vae_decoder.pth` exist. MLflow UI → `crash-severity-vae` → `vae_elbo` metric decreases over training epochs.

- [ ] T086 [US3] **RED** — Write `tests/test_train_vae.py`: instantiate `DVAETrainer` with minimal `VAEConfig` (small encoder_dims, few epochs, latent_dim=4) and dummy `X_all` (100 × 10 array); call `trainer.train(X_all)` → assert returns `VAETrainResult` with `best_epoch >= 1`; assert `vae_encoder.pth` and `vae_decoder.pth` are written to the configured paths; assert an MLflow run exists in `crash-severity-vae` with `vae_elbo` logged at `step=0` and `step=best_epoch`; assert encoder output shape is `(n_samples, latent_dim)` when called on `X_all`. Run — confirm FAIL.
- [ ] T087 [US3] **GREEN** — Implement `src/train_vae/vae_trainer.py`: define `Encoder(nn.Module)` (Linear → LayerNorm → ReLU stack with configurable dims; final Linear(last_dim, latent_dim) for μ and log_σ²); define `Decoder(nn.Module)` (mirrors encoder dims in reverse; final Linear output matches input_dim); define `DenoisingBetaVAE(nn.Module)` with `forward(x)` applying `F.dropout(x, p=dropout_p, training=True)` then encoder → reparameterize → decoder; `reparameterize(mu, log_var)` returns `mu + eps * std`; ELBO loss: `F.mse_loss(x_hat, x_clean) + beta * kl_loss` (reconstruction target is clean `x`, not corrupted); implement `DVAETrainer.train(X_all: np.ndarray) → VAETrainResult`: build `TensorDataset` from full X (no Y); Adam optimiser; training loop with `nn.Dropout` inpainting active; per-epoch val ELBO on a held-out 10% slice of X_all; early stopping on val ELBO (patience from config); save best encoder/decoder checkpoints; log per-epoch `vae_elbo`, `vae_reconstruction_loss`, `vae_kl_loss` with `step=epoch`; log params; return `VAETrainResult(best_epoch, final_elbo, encoder_path, decoder_path, run_id)`
- [ ] T088 [US3] **GREEN** — Create `src/train_vae/run.py`: load config via `src/config.py`; read `TRAIN_X_PATH`, `VAL_X_PATH`, `TEST_X_PATH`, `ENCODER_OUTPUT_PATH`, `DECODER_OUTPUT_PATH`, `MLFLOW_TRACKING_URI` env vars with defaults from config; concatenate all three X arrays (no Y); set `mlflow.set_tracking_uri`; instantiate `DVAETrainer(config.vae, config.mlflow)`; call `.train(X_all)`; exit 0 on success, exit 1 on error
- [ ] T089 [US3] Run `dvc repro train_vae` — confirm both `.pth` artifacts exist and MLflow `crash-severity-vae` experiment has a completed run
- [ ] T090 [US3] Open MLflow UI → `crash-severity-vae` → select run → Metrics → `vae_elbo` — confirm line chart shows decreasing trend; confirm `vae_reconstruction_loss` and `vae_kl_loss` both logged per epoch

**Checkpoint**: VAE trains on full X (no Y); ELBO curve logged per epoch; encoder + decoder artifacts written; MLflow run complete.

---

## Phase 3E: Encode Stage (US3)

**Goal**: Frozen encoder produces Z_train/val/test; LSA augments Z_train fatal class to ≥5%.

**Independent Test**: `dvc repro encode` → 4 `.npy` files exist. `np.unique(y_train_augmented)` → `[0, 1, 2]`. Fatal class fraction in `y_train_augmented` ≥ 0.05. `Z_val.shape[1] == 32`. No LSA applied to Z_val or Z_test.

- [ ] T091 [US3] **RED** — Write `tests/test_encode.py`: create a tiny `DVAETrainer` and train on dummy X to get a real encoder checkpoint; instantiate `LatentEncoder(encoder_path, encode_config, latent_dim=4)` with synthetic `(X_train, y_train, X_val, y_val, X_test)` where y_train has at least 10 Fatal-class (label=2) samples; call `.encode(...)` → assert `EncodeResult` with `Z_train_augmented.shape[1] == latent_dim`; assert fatal class fraction in `y_train_augmented` ≥ `lsa_target_ratio`; assert `Z_val.shape == (len(X_val), latent_dim)` (not augmented); assert `Z_test.shape == (len(X_test), latent_dim)` (not augmented); assert stage raises `RuntimeError` when fewer than `min_fatal_samples` Fatal rows exist in y_train. Run — confirm FAIL.
- [ ] T092 [US3] **GREEN** — Implement `src/encode/encoder.py`: `LatentEncoder.encode(X_train, y_train, X_val, y_val, X_test) → EncodeResult`: load encoder checkpoint from `encoder_path`; set model to eval mode (`torch.no_grad()`); encode each split → μ vectors as Z (use μ directly, not sampled z, for deterministic encoding at inference time); check `n_fatal = (y_train == 2).sum()` — raise `RuntimeError` if `n_fatal < min_fatal_samples`; apply LSA to Z_train: compute `fatal_mean` and `fatal_std` per dimension from real fatal Z vectors; sample `n_synthetic = max(0, int(len(Z_train) * lsa_target_ratio) - n_fatal)` Gaussian vectors around centroid; stack `Z_train_augmented = np.vstack([Z_train, synthetic_z])`; `y_train_augmented = np.hstack([y_train, np.full(n_synthetic, 2)])`; return `EncodeResult(Z_train_augmented, Z_val, Z_test, y_train_augmented, n_real_fatal, n_synthetic)`
- [ ] T093 [US3] **GREEN** — Create `src/encode/run.py`: load config; read env vars for encoder path, all X/y paths, output dir; instantiate `LatentEncoder`; call `.encode()`; save 4 numpy arrays; exit 0 on success, exit 1 on `RuntimeError` (too few fatal samples)
- [ ] T094 [US3] Run `dvc repro encode` — confirm 4 arrays written; verify `Z_train_augmented.shape[1] == 32` and fatal class fraction ≥ 0.05
- [ ] T095 [P] [US3] Confirm LSA isolation: load `Z_val.npy` and `Z_test.npy` — assert shapes match original split sizes from featurize (no extra rows); assert unique labels in `y_val.npy` and `y_test.npy` are unchanged from `y_val` / `y_test` produced by featurize

**Checkpoint**: US3 complete — ELBO converges, Z vectors produced for all splits, LSA applied to Z_train only, fatal fraction ≥ 5%, Z_val and Z_test untouched.

---

## Phase 4: Multi-Class A/B Test (US4)

**Goal**: XGBoost and MLP each trained N=10 seeds on Z_train_augmented; Welch's t-test produces p-value and declares winner; winner registered as `@champion`.

**Independent Test**: `dvc repro train_ml train_dl evaluate register` → 10 runs per MLflow experiment; `docs/ab_test_comparison.json` has `p_value`, `cohens_d`, `winner`, `gates_passed`; `per_class_matrix.json` artifact visible per run; `mlflow.pyfunc.load_model("models:/crash-severity@champion")` loads in < 30s.

### XGBoost Training (train_ml)

- [ ] T035 [US4] **RED** — Rewrite `tests/test_train_ml.py` for XGBoost on Z vectors: assert that with `ab_test.seeds=[0]`, exactly 1 MLflow run in `crash-severity-ml` tagged `seed=0`, `model_type=xgboost`; assert metrics `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`, `generalisation_gap` all logged; assert `per_class_matrix.json` artifact exists in the run; assert `models/best_ml_model.pkl` exists and is loadable as an `XGBClassifier`; assert `mlflow.sklearn.autolog()` was NOT used (no autolog params present). Run — confirm FAIL.
- [ ] T036 [US4] **GREEN** — Rewrite `src/train_ml/trainer.py`: implement `MLTrainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test) → MLTrainResult`; `mlflow.sklearn.autolog(disable=True)`; loop over seeds; for each seed: `XGBClassifier(objective='multi:softprob', num_class=3, random_state=seed, early_stopping_rounds=10)`; compute `sample_weight = compute_class_weights(y_train, n_classes=3)` from `src/metrics.py`; `clf.fit(Z_train, y_train, sample_weight=..., eval_set=[(Z_val, y_val)])`; start MLflow run with tags; log params + mandatory metrics + `per_class_matrix(y_test, y_pred, ['PDO','Injury','Fatal'])` as JSON artifact; track best seed by `eout_macro_f1`; return `MLTrainResult`
- [ ] T036b [US4] **GREEN** — Rewrite `src/train_ml/run.py`: load config; read `TRAIN_Z_PATH`, `TRAIN_Y_PATH`, `VAL_Z_PATH`, `VAL_Y_PATH`, `TEST_Z_PATH`, `TEST_Y_PATH`, `MODEL_OUTPUT_PATH` env vars; instantiate `MLTrainer(config.mlflow, config.model, config.ab_test.seeds)`; call `.train()`; save best pkl; exit 0/1
- [ ] T037 [US4] Run `dvc repro train_ml` — confirm 10 MLflow runs in `crash-severity-ml`; open MLflow UI, inspect `per_class_matrix.json` artifact for one run; confirm Fatal class metrics present

### MLP Training (train_dl)

- [ ] T038 [US4] **RED** — Rewrite `tests/test_train_dl.py` for MLP on Z vectors (no NAS): assert with `ab_test.seeds=[0]`, 1 MLflow run in `crash-severity-dl` tagged `seed=0`, `model_type=pytorch-mlp`, `architecture=32-64-3-dropout0.3`; assert per-epoch `ein_loss`/`eout_loss`/`gap_f1` logged with `step=epoch`; assert final `eout_macro_f1` and `eout_fatal_recall` logged; assert `models/mlp_model.pth` loadable with `torch.load` containing `state_dict`, `input_dim=32`, `hidden_dim=64`, `n_classes=3`. Run — confirm FAIL.
- [ ] T039 [US4] **RED** — Rewrite `tests/test_pyfunc.py`: save a minimal 3-class `MLP(32→64→3)` checkpoint; instantiate `MLPWrapper` from `src/train_dl/pyfunc.py`; call `predict(context, pd.DataFrame(Z_test[:5]))` → assert output is numpy array of shape `(5,)` with values in `{0, 1, 2}`. Run — confirm FAIL.
- [ ] T040 [US4] **GREEN** — Rewrite `src/train_dl/pyfunc.py`: `MLPWrapper(mlflow.pyfunc.PythonModel)`; `load_context` loads checkpoint, reconstructs `MLP(input_dim=32, hidden_dim=64, n_classes=3)` with saved weights; `predict` applies `softmax` → `argmax` → returns int numpy array
- [ ] T041 [US4] **GREEN** — Rewrite `src/train_dl/trainer.py`: define `MLP(nn.Module)` with `__init__(input_dim=32, hidden_dim=64, n_classes=3, dropout=0.3)`: `Linear(32,64) → ReLU → Dropout(0.3) → Linear(64,3)` (no BatchNorm — Z vectors are already normalised by the VAE); implement `DLTrainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test) → DLTrainResult`; no NAS; loop over seeds; compute `class_weights = compute_class_weights(y_train, 3)` → `CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float))`; Adam(lr from config); early stopping on val loss (patience from config); log per-epoch metrics with `step=epoch`; log final `eout_macro_f1`, `eout_fatal_recall`, `per_class_matrix` artifact; tag `architecture=32-64-3-dropout0.3`; track best seed; return `DLTrainResult`
- [ ] T042 [US4] **GREEN** — Create `src/train_dl/run.py`: load config; read Z path env vars; instantiate `DLTrainer(config.mlflow, config.dl, config.ab_test.seeds)`; call `.train()`; save best checkpoint dict (`state_dict`, `input_dim=32`, `hidden_dim`, `n_classes=3`) to `MODEL_OUTPUT_PATH`; exit 0/1
- [ ] T043 [US4] ~~Superseded by T041 and T042~~
- [ ] T044 [US4] Run `dvc repro train_dl` — confirm 10 runs in `crash-severity-dl`; verify per-epoch `ein_loss`/`eout_loss` visible as line charts in MLflow UI; verify early stopping fired before max epochs for at least one seed

### Evaluate + Register

- [ ] T046 [US4] **RED** — Rewrite `tests/test_evaluate.py`: mock N=3 MLflow runs per experiment with `eout_macro_f1` scores above and below thresholds; assert exit 0 + `docs/ab_test_comparison.json` contains `p_value`, `cohens_d`, `ci_ml`, `ci_dl`, `winner`, `significant`, `gates_passed`; **assert `gates_passed=false` when winner mean F1 ≤ 0.45 or winner mean fatal recall ≤ 0.30**; assert exit 1 when gates fail. Run — confirm FAIL.
- [ ] T047 [US4] **GREEN** — Rewrite `ABEvaluator.evaluate(Z_test, y_test)` in `src/evaluate/evaluator.py`: query MLflow for `eout_macro_f1` from `crash-severity-ml` (tagged `xgboost`) and `crash-severity-dl` (tagged `pytorch-mlp`); Welch's t-test; Cohen's d; 95% CIs; declare winner; assert `mean_macro_f1 > model.macro_f1_threshold (0.45)` AND `mean_fatal_recall > model.fatal_recall_threshold (0.30)`; return `EvaluationResult`. Create `src/evaluate/run.py`: load config + Z_test/y_test; call `.evaluate()`; write JSON reports; exit 1 if gates fail
- [ ] T048 [US4] Run `dvc repro evaluate` — confirm JSON output has all required fields; verify `gates_passed` and winner printed to stdout
- [ ] T049 [US4] **RED** — Write `tests/test_register.py`: assert with `gates_passed=true` → exit 0, `models:/crash-severity@champion` resolvable, `models/registry_receipt.json` exists; assert with `gates_passed=false` → exit 1, no registry mutation. Run — confirm FAIL.
- [ ] T050 [US4] **GREEN** — Implement `ModelRegistrar.register(winner, run_id)` in `src/register/registrar.py`; create `src/register/run.py`
- [ ] T051 [US4] Run `dvc repro register` — confirm `@champion` alias set; `mlflow.pyfunc.load_model("models:/crash-severity@champion")` loads; `model.predict(Z_test[:5])` returns array of shape `(5,)` with values in `{0,1,2}`

**Checkpoint**: US4 complete — 10 ML + 10 DL runs; Welch's t-test with p-value and per-class matrix; winner registered with constitutional gates enforced.

---

## Phase 4b: Katib β-HPO (US5)

**Goal**: Katib searches β over [0.5, 1.0, 2.0, 4.0, 8.0]; each trial retrains VAE + encodes + trains winner; fitness on Z_val (not Z_test); best β written to `params.yaml`.

**Independent Test**: `dvc repro tune` → N runs in `crash-severity-tune` tagged with `beta=<value>`; `params.yaml tune.best_params.beta` set; best trial `val_macro_f1` > pre-tune winner metric.

- [ ] T052 [P] Add `kubernetes>=28.0` to `pyproject.toml`; run `uv sync`
- [ ] T053 [P] Create `k8s/katib/vae_experiment.yaml`: Katib `Experiment` CRD; objective metric `val_macro_f1` (maximize, printed as `val_macro_f1=<value>` to stdout); algorithm `bayesianoptimization`; `maxTrialCount: 5` (one per β value), `parallelTrialCount: 1`; parameter: `beta` with feasibleSpace `{0.5, 1.0, 2.0, 4.0, 8.0}` (list type); trialTemplate command: `python -m src.tune.trial --beta={{.HyperParameters.beta}} --winner={{winner}}` inside `mlops-portfolio:latest` container; metrics collector reads `val_macro_f1=` from stdout
- [ ] T054 ~~SUPERSEDED — `k8s/katib/ml_experiment.yaml` replaced by `vae_experiment.yaml` (β HPO replaces classifier HPO)~~
- [ ] T055 ~~SUPERSEDED — `k8s/katib/dl_experiment.yaml` replaced by `vae_experiment.yaml`~~
- [ ] T056 [US5] **RED + GREEN** — Rewrite `src/tune/trial.py`: accepts `--beta <float>` and `--winner <ml|dl>` as argparse args; loads all X/y splits from PVC paths; instantiates `DVAETrainer` with candidate β; calls `trainer.train(X_all)` to get encoder checkpoint; instantiates `LatentEncoder` with the new checkpoint; calls `encoder.encode(X_train, y_train, X_val, y_val, X_test)` (LSA applied); trains winner classifier on `Z_train_augmented` (1 seed, seed=0 for trials); evaluates on **Z_val** (not Z_test — constitution II); logs full trial to MLflow `crash-severity-tune` tagged `beta=<value>`, `winner=<ml|dl>`, `trial_type=katib`; **prints `val_macro_f1=<float>` on last stdout line** (required by Katib metrics collector — this is the fitness signal); exits 0
- [ ] T057 [US5] **RED** — Write `tests/test_tune.py`: mock Kubernetes client; assert `HyperparamTuner.tune()` submits Experiment with correct name/namespace; assert it reads `currentOptimalTrial.parameterAssignments.beta`; assert `params.yaml` updated with `tune.best_params.beta` after `run.py` completes. Run — confirm FAIL.
- [ ] T058 [US5] **GREEN** — Implement `HyperparamTuner.tune()` in `src/tune/tuner.py`: load `k8s/katib/vae_experiment.yaml`; inject winner into trial template; submit via `kubernetes.client.CustomObjectsApi`; poll `status.conditions` until Succeeded/Failed; read `status.currentOptimalTrial.parameterAssignments` → extract `beta`; return `TuneResult(best_beta, best_val_macro_f1, n_trials)`. Create `src/tune/run.py`: load config; read winner from `REPORT_PATH`; call `HyperparamTuner.tune()`; write `tune.best_params.beta = <value>` to `params.yaml` with `yaml.safe_dump`; exit 0
- [ ] T059 [US5] Run `dvc repro tune` on local Kubernetes — confirm Katib Experiment in Katib UI; confirm 5 MLflow runs in `crash-severity-tune`; confirm `params.yaml` has `tune.best_params.beta` set to a float value
- [ ] T060 [US5] Run `dvc repro register` after tune — DVC detects `params.yaml` change → re-runs train_vae → encode → train_ml/train_dl → evaluate → register with best β; confirm new registry version registered

**Checkpoint**: US5 complete — β searched via Katib; val_macro_f1 as fitness (Z_val only); best β in params.yaml; downstream stages invalidated and re-run.

---

## Phase 5: KFP Orchestration (US6)

**Goal**: Full 10-stage pipeline compiled to `pipeline.yaml`; runs on Docker Desktop Kubernetes via KFP UI.

**Independent Test**: Submit pipeline from KFP UI → all 10 steps appear with correct dependency arrows; `train_vae` and `encode` sequential before `train_ml`/`train_dl`; MLflow runs tagged `orchestrator=kubeflow`.

- [ ] T066 [US6] Create `docker/Dockerfile`: `FROM python:3.12-slim`; install `uv`; copy `pyproject.toml`, `uv.lock`; `uv sync --frozen`; copy `src/`, `dvc.yaml`, `params.yaml`, `great_expectations/gx/`; `ENV PYTHONPATH=/app`. Do NOT copy `mlruns/`, `data/`, `models/` — these come from PVC mount.
- [ ] T067 [US6] Build image: `docker build -f docker/Dockerfile -t mlops-portfolio:latest .`; smoke-test: `docker run --rm -v $(pwd)/data:/app/data mlops-portfolio:latest python -m src.ingest.run` → exit 0
- [ ] T068 [US6] Enable Kubernetes in Docker Desktop; verify with `kubectl cluster-info`
- [ ] T069 [US6] Create `k8s/pvc.yaml`: hostPath PV + PVC mounting project root at `/app` (`storageClassName: manual`, `accessModes: ReadWriteOnce`, `capacity: 20Gi`); apply with `kubectl apply -f k8s/pvc.yaml`
- [ ] T070 [US6] Install KFP standalone: apply cluster-scoped CRDs then platform-agnostic manifests; wait for all pods ready; `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80`
- [ ] T071 [US6] Rewrite `pipelines/kubeflow/pipeline.py` with **10** `@dsl.component(base_image="mlops-portfolio:latest")` functions — one per stage — each calling `subprocess.run(["dvc","repro","<stage>"], check=True, cwd="/app")`; mount PVC from T069 at `/app`; define `@dsl.pipeline(name="crash-severity-pipeline")` wiring all 10 stages: `validate >> ingest >> featurize >> train_vae >> encode`, then `encode >> train_ml` and `encode >> train_dl` (parallel), then `train_ml >> evaluate`, `train_dl >> evaluate`, then `evaluate >> tune >> register`; compile to `pipelines/kubeflow/pipeline.yaml`
- [ ] T072 [US6] Compile pipeline: `uv run python pipelines/kubeflow/pipeline.py` → confirm `pipeline.yaml` created
- [ ] T073 [US6] Upload `pipeline.yaml` to KFP UI; create and start run; confirm all 10 steps appear with correct arrows (train_ml and train_dl shown as parallel after encode)
- [ ] T074 [US6] Inspect `train_vae` pod logs in KFP UI — confirm ELBO per-epoch output visible; inspect `train_ml` pod — confirm MLflow run tagged `orchestrator=kubeflow`

**Checkpoint**: US6 complete — 10-stage KFP pipeline compiled and running on Docker Desktop Kubernetes.

---

## Phase 6: Polish & Validation

**Purpose**: Assert all constitutional gates; update documentation; verify full reproducibility.

- [ ] T075 [P] Assert constitutional gates on final evaluation report: `python -c "import json; r=json.load(open('docs/evaluation_report.json')); print('F1:', 'PASS' if r['winner_macro_f1']>0.45 else 'FAIL'); print('Fatal recall:', 'PASS' if r['winner_fatal_recall']>0.30 else 'FAIL')"`
- [ ] T076 [P] Update `CLAUDE.md`: update architecture table (add `train_vae` + `encode` rows); update pipeline description to 10-stage; update DL section (remove EvoTorch NAS, remove FlexMLP, add MLP on Z description); update featurize section (3-class target encoding)
- [ ] T077 [P] Add to `.gitignore`: `data/processed/Z_*.npy`, `data/processed/y_train_augmented.npy`, `models/vae_*.pth`, `models/registry_receipt.json`
- [ ] T078 Commit all tracked files: `dvc.yaml`, `params.yaml`, `src/`, `great_expectations/gx/expectations/`, `pipelines/kubeflow/pipeline.py`, `docker/Dockerfile`, `k8s/`
- [ ] T079 [P] Full reproducibility smoke test: delete `data/processed/` and `models/`; run `dvc pull && dvc repro`; confirm all 10 stages complete and artifacts restored
- [ ] T080 [P] Remove `apache-airflow` from `pyproject.toml` (if present); run `uv sync`; confirm no import errors

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ✅
  └── Phase 2 (Foundational) ✅ + T081-T085 extensions
        └── Phase 3A (US1: Featurize TDD)           ← MVP start
              ├── Phase 3b (Data Contract) ✅
              │     └── Phase 3B (US2: GE Validate)
              └── Phase 3C (DVC Pipeline Integration)
                    └── Phase 3D (US3: train_vae)    ← NEW
                          └── Phase 3E (US3: encode) ← NEW
                                ├── Phase 4 (US4: train_ml/train_dl/evaluate/register)
                                │     └── Phase 4b (US5: Katib β-HPO)
                                └── Phase 5 (US6: KFP)
  Phase 6 (Polish) — after all stories complete
```

### Stage Execution Order (DVC DAG)

| Order | Stage | Parallel with | Blocks |
|---|---|---|---|
| 1 | validate | — | ingest |
| 2 | ingest | — | featurize |
| 3 | featurize | — | train_vae |
| 4 | train_vae | — | encode |
| 5 | encode | — | train_ml, train_dl |
| 6a | train_ml | train_dl | evaluate |
| 6b | train_dl | train_ml | evaluate |
| 7 | evaluate | — | tune, register |
| 8 | tune | — | (invalidates train_vae → register) |
| 9 | register | — | — |

### Parallel Opportunities

- T081–T085 (Phase 2 extensions) — all parallel
- T023–T026 (DVC verification) — T023/T024 sequential; T025/T026 parallel after T024
- T035, T038, T039 (test writing in Phase 4) — all parallel
- T036b and T042 (run.py files for train_ml and train_dl) — parallel
- T066–T070 (Docker + Kubernetes setup) — T066/T067 sequential; T068–T070 sequential after T067
- T075, T076, T077, T079, T080 (Polish) — all parallel

---

## Implementation Strategy

### MVP First (US1 — DVC Reproducibility)

1. Complete Phase 1 + 2 (already done ✅ + T081–T085)
2. Complete Phase 3A (T020, T021 — 3-class featurize)
3. Complete Phase 3C T022 — wire the 10-stage dvc.yaml
4. Run `dvc repro featurize` — arrays exist, pipeline caches
5. **STOP and VALIDATE**: `dvc status` → all cached; `y_train` values = {0,1,2}

### Incremental Delivery

1. Phase 2 extensions (T081–T085) → VAE scaffolding ready
2. Phase 3A–3C → featurize + DVC pipeline wired
3. Phase 3D–3E → VAE trains; Z vectors produced (US3 checkpoint)
4. Phase 4 → A/B test complete; champion registered (US4 checkpoint)
5. Phase 4b → β tuned via Katib (US5 checkpoint)
6. Phase 5 → KFP orchestration (US6 checkpoint)
7. Phase 6 → constitutional gates asserted; full reproducibility confirmed

### Parallel Team Strategy

After Phase 2 extensions (T081–T085) complete:
- **Track A**: US1 → US2 → US3 (VAE + encode) → US4 (A/B test) → US5 (Katib)
- **Track B** (after Docker image built in T066–T067): US6 (KFP)

---

## Notes

- TDD required for all `src/` code (constitution XV) — RED test MUST fail before GREEN implementation
- `train_ml` and `train_dl` are sequential within their DVC stage (N seeds in a loop) but run in parallel in the DVC DAG
- XGBoost on 32-dim Z vectors trains fast — 10 seeds should complete in < 10 minutes
- Katib trials retrain the full VAE + encode + classifier per trial — each trial may take 5–15 min; plan for 25–75 min total tune stage
- `val_macro_f1` (Katib fitness, Z_val) and `eout_macro_f1` (final test metric, Z_test) are different numbers from different splits — never conflate them
- Z_val and Z_test MUST NOT be augmented under any circumstances (constitution III v3.1.0)
- The `register` stage has no `outs` that DVC tracks (MLflow Model Registry is not a filesystem output) — `registry_receipt.json` is the only DVC-tracked output
- After Katib writes `tune.best_params.beta` to `params.yaml`, DVC detects the change and invalidates `train_vae` and all downstream stages — a full re-run follows automatically
