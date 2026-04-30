# Tasks: MLOps Learning Portfolio — VAE-Based Crash Severity Pipeline

**Input**: Design documents from `specs/002-mlops-portfolio/`
**Prerequisites**: spec.md ✅ | plan.md ✅ | research.md ✅ | data-model.md ✅ | contracts/stage-interface.md ✅
**Architecture**: VAE-based, 10-stage pipeline (validate→ingest→featurize→train_vae→encode→train_ml→train_dl→evaluate→tune→register)
**Constitution**: v3.3.0 — TDD for all `src/` (XV); boundary tests only (XIV); GE exclusive QA layer (XVI); CTGAN augmentation on X_train only (III)

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

- [x] T086 [US3] **RED** — Write `tests/test_train_vae.py`: instantiate `DVAETrainer` with minimal `VAEConfig` (small encoder_dims, few epochs, latent_dim=4) and dummy `X_all` (100 × 10 array); call `trainer.train(X_all)` → assert returns `VAETrainResult` with `best_epoch >= 1`; assert `vae_encoder.pth` and `vae_decoder.pth` are written to the configured paths; assert an MLflow run exists in `crash-severity-vae` with `vae_elbo` logged at `step=0` and `step=best_epoch`; assert encoder output shape is `(n_samples, latent_dim)` when called on `X_all`. Run — confirm FAIL.
- [x] T087 [US3] **GREEN** — Implement `src/train_vae/vae_trainer.py`: define `Encoder(nn.Module)` (Linear → LayerNorm → ReLU stack with configurable dims; final Linear(last_dim, latent_dim) for μ and log_σ²); define `Decoder(nn.Module)` (mirrors encoder dims in reverse; final Linear output matches input_dim); define `DenoisingBetaVAE(nn.Module)` with `forward(x)` applying `F.dropout(x, p=dropout_p, training=True)` then encoder → reparameterize → decoder; `reparameterize(mu, log_var)` returns `mu + eps * std`; ELBO loss: `F.mse_loss(x_hat, x_clean) + beta * kl_loss` (reconstruction target is clean `x`, not corrupted); implement `DVAETrainer.train(X_all: np.ndarray) → VAETrainResult`: build `TensorDataset` from full X (no Y); Adam optimiser; training loop with `nn.Dropout` inpainting active; per-epoch val ELBO on a held-out 10% slice of X_all; early stopping on val ELBO (patience from config); save best encoder/decoder checkpoints; log per-epoch `vae_elbo`, `vae_reconstruction_loss`, `vae_kl_loss` with `step=epoch`; log params; return `VAETrainResult(best_epoch, final_elbo, encoder_path, decoder_path, run_id)`
- [x] T088 [US3] **GREEN** — Create `src/train_vae/run.py`: load config via `src/config.py`; read `TRAIN_X_PATH`, `VAL_X_PATH`, `TEST_X_PATH`, `ENCODER_OUTPUT_PATH`, `DECODER_OUTPUT_PATH`, `MLFLOW_TRACKING_URI` env vars with defaults from config; concatenate all three X arrays (no Y); set `mlflow.set_tracking_uri`; instantiate `DVAETrainer(config.vae, config.mlflow)`; call `.train(X_all)`; exit 0 on success, exit 1 on error
- [x] T089 [US3] Run `dvc repro train_vae` — confirm both `.pth` artifacts exist and MLflow `crash-severity-vae` experiment has a completed run
- [x] T090 [US3] Open MLflow UI → `crash-severity-vae` → select run → Metrics → `vae_elbo` — confirm line chart shows decreasing trend; confirm `vae_reconstruction_loss` and `vae_kl_loss` both logged per epoch

**Checkpoint**: VAE trains on full X (no Y); ELBO curve logged per epoch; encoder + decoder artifacts written; MLflow run complete.

---

## Phase 3E: Encode Stage (US3)

**Goal**: Frozen encoder projects CTGAN-augmented `X_train_augmented` → `Z_train_augmented`; `Z_val` and `Z_test` encoded from original splits unchanged. No LSA — CTGAN handles augmentation upstream.

**Independent Test**: `dvc repro encode` → 3 `.npy` files exist (`Z_train_augmented`, `Z_val`, `Z_test`). `Z_train_augmented.shape[0] == len(X_train_augmented)`. `Z_val.shape[1] == latent_dim`. `Z_test.shape[1] == latent_dim`. No extra rows in Z_val or Z_test.

- [x] T091 [US3] **RED** — Rewrite `tests/test_encode.py`: create a tiny `DVAETrainer` and train on dummy X to get a real encoder checkpoint; instantiate `LatentEncoder(encoder_path, latent_dim=4)` with synthetic `(X_train_augmented, y_train_augmented, X_val, X_test)` — `X_train_augmented` already contains CTGAN-generated Fatal rows; call `.encode(...)` → assert `EncodeResult` with `Z_train_augmented.shape == (len(X_train_augmented), latent_dim)`; assert `Z_val.shape == (len(X_val), latent_dim)`; assert `Z_test.shape == (len(X_test), latent_dim)`; assert no LSA or synthetic row injection occurs inside encoder. Run — confirm FAIL.
- [x] T092 [US3] **GREEN** — Rewrite `src/encode/encoder.py`: `LatentEncoder.encode(X_train_augmented, y_train_augmented, X_val, X_test) → EncodeResult`: load encoder checkpoint from `encoder_path`; set model to eval mode (`torch.no_grad()`); encode each split → μ vectors as Z (use μ directly, not sampled z, for deterministic encoding); no LSA — augmentation was handled by the `augment` stage; return `EncodeResult(Z_train_augmented, Z_val, Z_test)`
- [x] T093 [US3] **GREEN** — Rewrite `src/encode/run.py`: load config; read `X_TRAIN_AUG_PATH`, `Y_TRAIN_AUG_PATH`, `X_VAL_PATH`, `X_TEST_PATH`, `ENCODER_PATH`, `OUTPUT_DIR`, `MLFLOW_TRACKING_URI` env vars; instantiate `LatentEncoder`; call `.encode()`; save 3 numpy arrays + pass-through `y_train_augmented.npy`; exit 0 on success, exit 1 on error
- [x] T094 [US3] Run `dvc repro encode` — confirm `Z_train_augmented.npy`, `Z_val.npy`, `Z_test.npy` written; verify `Z_train_augmented.shape[1] == latent_dim`; verify row count matches `X_train_augmented`. ✅ All checks passed: latent_dim=8, Z_train_augmented rows=54676 match X_train_augmented.
- [x] T095 [P] [US3] Confirm val/test isolation: load `Z_val.npy` and `Z_test.npy` — assert shapes match original split sizes from featurize (no extra rows); assert `Z_val.shape[0] == len(X_val)` and `Z_test.shape[0] == len(X_test)`. ✅ Val/test isolation confirmed: both have 11147 rows, no augmentation leak.

- [x] T100 [US3] Create `notebooks/vae_eda.ipynb` — Phase 3E checkpoint notebook: load `X_train/val/test.npy` + `y_*` + VAE encoder/decoder checkpoints; run mean-path reconstructions (no sampling); plot (a) per-sample MSE histogram, (b) per-feature MSE bar chart, (c) real vs reconstructed KDE overlays for numeric features, (d) KS-test table per feature, (e) latent-space PCA scatter coloured by severity class, (f) reconstruction MSE boxplot per class, (g) posterior variance per latent dim; notebook includes stale checkpoint guard `if enc_ckpt['input_dim'] != X_all.shape[1]`. ✅ All 7 visualizations present.

**Checkpoint**: ✅ **Phase 3E Complete** — US3 complete: ELBO converges, Z vectors produced for all splits from CTGAN-augmented X_train, no LSA in encode stage, Z_val and Z_test untouched; VAE EDA notebook confirms reconstruction fidelity and latent structure.

---

## Phase 3F: Config Foundation (Blocking Prerequisites for Phases 3G–4B)

**Purpose**: params.yaml and config.py updated before any cyclical-encoding, KL-annealing, augment, or train_dl code is written. All downstream phases depend on this.

- [x] T101 Amend `.specify/memory/constitution.md` Principle III: bump version to 3.3.0; add third permitted mechanism — "Generative augmentation (CTGAN/TVAE) is permitted in a dedicated `augment` DVC stage on the training split only; augmented data must be a DVC-tracked artifact; val and test splits must never be augmented"; update Quality Gates table entry to reflect three mechanisms
- [x] T102 [P] Update `params.yaml`: (a) remove HOUR from `features.numeric_columns`; remove MONTH from `features.ordinal_columns`; add `features.cyclical_columns: {HOUR: 24, MONTH: 12}`; (b) update `vae.*`: `latent_dim: 8`, `lr: 0.0005`, replace `beta: 1.0` with `beta_start: 0.0`, `beta_max: 0.5`, `warmup_epochs: 15`; (c) remove `encode.*` section (no LSA parameters needed — encode stage is now a pure projection pass); (d) add `augment` section: `tvae_epochs: 500`, `target_fatal_ratio: 0.05`, `random_state: 42`; (e) update `dl.*` section for shallow MLP: `input_dim: 8` (= latent_dim), `hidden_dim: 64`, `dropout_p: 0.1`, `epochs: 100`, `patience: 10`, `batch_size: 256`, `lr: 0.001`, `experiment_name: crash-severity-dl`
- [x] T103 [P] Update `src/config.py`: (a) update `VAEConfig`: replace single `beta: float` field with `beta_start: float`, `beta_max: float`, `warmup_epochs: int`; (b) update `EncodeConfig`: remove `lsa_target_ratio` and `min_fatal_samples` fields (LSA removed); (c) add `AugmentConfig(tvae_epochs: int, target_fatal_ratio: float, random_state: int)` dataclass; (d) update `DLConfig`: set fields to `input_dim: int`, `hidden_dim: int`, `dropout_p: float`, `epochs: int`, `patience: int`, `batch_size: int`, `lr: float`, `experiment_name: str`; (e) add `AugmentConfig` to `ProjectConfig` and `load_config()`; remove any `CVAEConfig` if present

**Checkpoint**: `params.yaml` has `augment.*`, `features.cyclical_columns`, updated `vae.*` with annealing fields, updated `dl.*` for shallow MLP; `VAEConfig` has annealing fields; `AugmentConfig`/`DLConfig` importable from `src/config.py`; `EncodeConfig` has no LSA fields; constitution III at v3.3.0.

---

## Phase 3G: Featurize — Cyclical Encoding [US3]

**Goal**: MONTH and HOUR replaced by sine/cosine pairs in featurize; `X_train.npy` has `MONTH_sin`, `MONTH_cos`, `HOUR_sin`, `HOUR_cos`; feature count changes by +2 (2 removed, 4 added).

**Independent Test**: After `dvc repro featurize`, load `X_train.npy` and assert 4 cyclical columns present in feature names; values bounded `[-1.0, 1.0]`; no standalone MONTH or HOUR integer column.

- [x] T104 [US3] **RED** — Update `tests/test_featurize.py`: add assertions that the list returned by `preprocessor.get_feature_names_out()` includes `num__MONTH_sin`, `num__MONTH_cos`, `num__HOUR_sin`, `num__HOUR_cos`; assert no feature named `ord__MONTH` or `num__HOUR`; assert all four cyclical values are in `[-1.0, 1.0]`; assert `X_train.shape[1]` == previous column count + 2. Run — confirm FAIL.
- [x] T105 [US3] **GREEN** — Update `src/featurize/featurizer.py`: add `_apply_cyclical(df)` private method that reads `cyclical_columns` dict (`{HOUR: 24, MONTH: 12}`) from config; for each `(col, period)`: adds `{col}_sin = sin(2π × ordinal_integer / period)` and `{col}_cos = cos(2π × ordinal_integer / period)` columns to df; drops original col; update `_select_and_recode` to call `_apply_cyclical` after recoding; add the four sin/cos column names to `numeric_cols` in `_fit_preprocess`; remove HOUR from `numeric_cols` and MONTH from `ord_names` lookups; update `dvc.yaml` featurize `params` to include `features.cyclical_columns`

**Checkpoint**: `dvc repro featurize` exits 0; all X arrays have 4 cyclical columns; MONTH ordinal and HOUR numeric columns absent; downstream train_vae and encode re-run due to changed X shape.

---

## Phase 3H: train_vae — KL Annealing Fix [US3]

**Goal**: `DVAETrainer` applies linear KL warmup (`beta: 0 → beta_max` over `warmup_epochs`) instead of fixed `beta=1.0`; posterior collapse eliminated.

**Independent Test**: After `dvc repro train_vae`, open MLflow `crash-severity-vae` → confirm `kl_beta` logged at step 0 = 0.0 and at step `warmup_epochs` = `beta_max`; `vae_kl_loss` near-zero for first `warmup_epochs` epochs then rises.

- [x] T106 [US3] **RED** — Update `tests/test_train_vae.py`: add assertion that MLflow run logs metric `kl_beta` at `step=0` with value `0.0`; assert `kl_beta` at `step=warmup_epochs` equals `beta_max`; assert encoder output has `std > 0.05` across all `latent_dim` dims on synthetic data (no total collapse). Run — confirm FAIL.
- [x] T107 [US3] **GREEN** — Update `src/train_vae/vae_trainer.py` `DVAETrainer.train()`: each epoch compute `beta_t = min(vae_config.beta_max, vae_config.beta_start + (vae_config.beta_max - vae_config.beta_start) * epoch / max(1, vae_config.warmup_epochs))`; pass `beta_t` into `DenoisingBetaVAE.loss_function(x_hat, x_clean, mu, log_var, beta=beta_t)` instead of stored `self.beta`; remove `self.beta` from `DenoisingBetaVAE.__init__`; log `kl_beta` per epoch with `step=epoch`
- [x] T121 [US3] **VAE Fatal Fix — Weighted Sampler + Augmented Training Data** — Implemented during audit-driven diagnosis (2026-04-29). Updated `src/train_vae/vae_trainer.py` `DVAETrainer.train()` to accept optional `y_all: np.ndarray`; when provided, builds `WeightedRandomSampler` using `compute_class_weights()` so Fatal rows receive proportional gradient share (weight formula: `N / (n_classes × class_count_c)`); updated `src/train_vae/run.py` to load `X_train_augmented.npy` + `y_train_augmented.npy` and pass concatenated `y_all` to trainer; updated `dvc.yaml` train_vae deps to include augmented artifacts (augment is now a prerequisite of train_vae); added MLflow params `weighted_sampler` and `n_fatal_train`; added boundary tests `test_weighted_sampler_logged_when_y_all_provided` and `test_train_without_y_all_still_works` to `tests/test_train_vae.py`. Audit result: KS-aligned dims improved 2→4/8; PDO overlap increase accepted (business decision: Fatal recall > PDO precision). See `docs/vae_fix_plan.md` Fix 1.

**Checkpoint**: KL beta ramps 0 → beta_max over warmup_epochs; `kl_beta` logged per epoch; posterior collapse eliminated; existing T086–T095 tests still green.

---

## Phase 4A: Augment Stage — CTGAN Fatal Class Augmentation [US4]

**Goal**: New `augment` DVC stage generates CTGAN/TVAE synthetic Fatal training samples from `X_train.npy`, outputs `X_train_augmented.npy` and `y_train_augmented.npy` as tracked DVC artifacts; fatal class fraction ≥ `augment.target_fatal_ratio`.

**Independent Test**: `dvc repro augment` → `data/processed/X_train_augmented.npy` and `data/processed/y_train_augmented.npy` exist; `fatal_fraction(y_train_augmented) >= 0.05`; `X_train_augmented.shape[1] == X_train.shape[1]`; non-fatal row count unchanged.

- [x] T108 [P] Create `src/augment/` package: `src/augment/__init__.py`; add `ctgan>=0.10` to `pyproject.toml`; run `uv sync`
- [x] T109 [P] Add `augment` stage to `dvc.yaml` (positioned after `featurize`, parallel with `train_vae`, both feeding into `encode`): `cmd: python -m src.augment.run`; `deps: [src/augment/run.py, src/augment/augmenter.py, data/processed/X_train.npy, data/processed/y_train.npy, src/config.py]`; `params: [augment.*, mlflow.tracking_uri]`; `outs: [data/processed/X_train_augmented.npy, data/processed/y_train_augmented.npy]`; update `encode` stage `deps` to use `data/processed/X_train_augmented.npy` and `data/processed/y_train_augmented.npy` instead of `X_train.npy` and `y_train.npy`
- [x] T110 [US4] **RED** — Write `tests/test_augmenter.py`: create `AugmentConfig(tvae_epochs=2, target_fatal_ratio=0.15, random_state=42)`; instantiate `CTGANAugmenter(config)`; build synthetic `X_train` (60 rows × 8 cols), `y_train` (10 Fatal=2 rows, 50 non-Fatal); call `augmenter.augment(X_train, y_train) → AugmentResult`; assert `result.X_augmented.shape[1] == 8`; assert fatal fraction in `result.y_augmented >= 0.15`; assert non-fatal row count in `result.y_augmented == 50`; assert `result.n_real_fatal == 10`; assert `result.n_synthetic > 0`; assert `RuntimeError` raised when Fatal rows < 10. Run — confirm FAIL.
- [x] T111 [US4] **GREEN** — Implement `src/augment/augmenter.py`: `AugmentResult(X_augmented: np.ndarray, y_augmented: np.ndarray, n_real_fatal: int, n_synthetic: int)` dataclass; `CTGANAugmenter(augment_config: AugmentConfig)` with `augment(X_train: np.ndarray, y_train: np.ndarray) → AugmentResult`: (a) validate `n_fatal = (y_train == 2).sum() >= 10` — raise `RuntimeError` if not; (b) extract `X_fatal = X_train[y_train == 2]` as a `pd.DataFrame`; (c) fit `TVAE(epochs=augment_config.tvae_epochs, cuda=False)` on `X_fatal`; (d) compute `n_synthetic = max(0, ceil((target × len(X_train) - n_fatal) / (1 − target)))`; (e) `synthetic_np = tvae.sample(n_synthetic).to_numpy()`; (f) stack `X_augmented = np.vstack([X_train, synthetic_np])`; `y_augmented = np.hstack([y_train, np.full(n_synthetic, 2)])`; return `AugmentResult`
- [x] T112 [US4] **GREEN** — Create `src/augment/run.py`: load config via `src/config.py`; read `X_TRAIN_PATH`, `Y_TRAIN_PATH`, `X_AUG_OUTPUT`, `Y_AUG_OUTPUT`, `MLFLOW_TRACKING_URI` env vars with config defaults; `mlflow.set_tracking_uri`; instantiate `CTGANAugmenter(config.augment)`; call `.augment(X_train, y_train)`; save `X_train_augmented.npy` and `y_train_augmented.npy`; log `n_real_fatal`, `n_synthetic`, `fatal_fraction_after` to MLflow; exit 0 on success, exit 1 on `RuntimeError`
- [x] T113 [US4] Run `dvc repro augment` — confirm `X_train_augmented.npy` and `y_train_augmented.npy` written; verify `(y_train_augmented == 2).mean() >= augment.target_fatal_ratio`; verify `X_train_augmented.shape[1] == X_train.shape[1]`

**Checkpoint**: augment stage complete; CTGAN Fatal samples generated; fatal fraction at target; both augmented arrays DVC-tracked.

---

## Phase 4B: train_dl Stage — Shallow MLP on Z-space [US4]

**Goal**: Shallow MLP (`Input(latent_dim) → Linear(64) → ReLU → Dropout → Linear(3)`) trained on `Z_train_augmented` with class weights and early stopping; 10-seed `crash-severity-dl` MLflow experiment; `models/mlp_model.pth` artifact written.

**Independent Test**: `dvc repro train_dl` → exactly 10 runs in `crash-severity-dl`; each run has `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`, `generalisation_gap`, `per_class_matrix.json` artifact; `models/mlp_model.pth` exists and loads.

- [x] T114 [P] Verify `src/train_dl/` package exists with `__init__.py`; ensure stub from T016 is in place. ✅ Package exists with `__init__.py` and `trainer.py` stub (NotImplementedError).
- [x] T115 [P] Update `dvc.yaml` `train_dl` stage: `cmd: python -m src.train_dl.run`; `deps: [src/train_dl/run.py, src/train_dl/trainer.py, data/processed/Z_train_augmented.npy, data/processed/y_train_augmented.npy, data/processed/Z_val.npy, data/processed/y_val.npy, data/processed/Z_test.npy, data/processed/y_test.npy, src/config.py]`; `params: [dl.*, model.*, ab_test.*, mlflow.*]`; `outs: [models/mlp_model.pth]`; confirm `evaluate` stage `deps` includes `models/mlp_model.pth`. ✅ All deps, params, and outs present; evaluate stage includes mlp_model.pth.
- [x] T116 [US4] **RED** — Write `tests/test_train_dl.py`: instantiate `DLTrainer` with minimal `DLConfig(input_dim=8, hidden_dim=16, dropout_p=0.1, epochs=3, patience=10, batch_size=32, lr=0.001, experiment_name='test-dl')`; build synthetic `Z_train` (60 × 8), `y_train` (3 classes, at least 5 per class), `Z_val` (20 × 8), `y_val`, `Z_test` (20 × 8), `y_test`; call `trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test) → DLTrainResult`; assert `result.best_epoch >= 1`; assert `mlp_model.pth` written and loadable; assert MLflow run exists in `test-dl` with `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`, `generalisation_gap` logged; assert `per_class_matrix.json` artifact present; assert `mlflow.sklearn.autolog()` was NOT used. ✅ RED confirmed — 5 tests fail with TypeError on model_config parameter.
- [x] T117 [US4] **GREEN** — Implement `src/train_dl/trainer.py`: `DLTrainResult(best_epoch, best_val_loss, model_path, run_id, seed)` dataclass; `DLTrainer(dl_config, mlflow_config, seeds, model_config)` with `train(Z_train, y_train, Z_val, y_val, Z_test, y_test) → DLTrainResult`: `mlflow.autolog(disable=True)`; compute class weights via `compute_class_weights(y_train, n_classes=3)` from `src/metrics.py`; seed loop — for each seed: build `ShallowMLP(input_dim, hidden_dim, n_classes, dropout_p)` = `Linear(input_dim, hidden_dim) → ReLU → Dropout(dropout_p) → Linear(hidden_dim, n_classes)`; `CrossEntropyLoss(weight=class_weights)`; Adam; DataLoader; per-epoch train + val loss; early stopping on val loss (patience); after training evaluate on Z_test: log mandatory metrics `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`, `generalisation_gap`; log `per_class_matrix(y_test, y_pred, ['PDO','Injury','Fatal'])` as JSON artifact; track best seed by `eout_macro_f1`; save checkpoint; return `DLTrainResult`. ✅ GREEN — all 5 tests pass.
- [x] T118 [US4] **GREEN** — Create `src/train_dl/run.py`: load config via `src/config.py`; read `Z_TRAIN_PATH`, `Y_TRAIN_PATH`, `Z_VAL_PATH`, `Y_VAL_PATH`, `Z_TEST_PATH`, `Y_TEST_PATH`, `MODEL_OUTPUT_PATH`, `MLFLOW_TRACKING_URI` env vars with config defaults; `mlflow.set_tracking_uri`; `mlflow.set_experiment(config.dl.experiment_name)`; instantiate `DLTrainer(config.dl, config.mlflow, config.ab_test.seeds, config.model)`; call `.train()`; save best checkpoint to `MODEL_OUTPUT_PATH`; exit 0 on success, exit 1 on error. ✅ Created with proper error handling and logging.
- [x] T119 [US4] Run `dvc repro train_dl` — confirm exactly 10 runs in `crash-severity-dl`; verify `models/mlp_model.pth` exists and is loadable. ✅ 10 runs confirmed (seeds 0-9); best seed 8, epoch 22, val_loss 0.9086; checkpoint loadable with all keys (epoch, model_state_dict, optimizer_state_dict, val_loss).
- [x] T120 [US4] Open MLflow UI → `crash-severity-dl` → select any run → confirm `eout_macro_f1`, `eout_fatal_recall`, `per_class_matrix.json` present; confirm `generalisation_gap` logged; confirm no autolog params. ✅ All mandatory metrics present (eout_macro_f1: 0.3269, eout_fatal_recall: 0.5625, ein_macro_f1: 0.4448, generalisation_gap: 0.1179); per_class_matrix.json artifact confirmed; no autolog params detected (only explicit: seed, lr, batch_size, etc.).

**Checkpoint**: US4 (MLP) complete — 10-seed shallow MLP on Z-space; cross-entropy + class weights; early stopping; mandatory metrics and per-class matrix logged; `models/mlp_model.pth` artifact written.

---

## Phase 4: Multi-Class A/B Test (US4)

**Goal**: XGBoost (tree-based, Z-space) and MLP (neural, Z-space) each trained N=10 seeds; Welch's t-test produces p-value and declares winner; winner registered as `@champion`.

**Independent Test**: `dvc repro train_ml train_dl evaluate register` → 10 runs per MLflow experiment; `docs/ab_test_comparison.json` has `p_value`, `cohens_d`, `winner`, `gates_passed`; `mlflow.pyfunc.load_model("models:/crash-severity@champion")` loads in < 30s.

### XGBoost Training (train_ml)

- [ ] T035 [US4] **RED** — Rewrite `tests/test_train_ml.py` for XGBoost on Z vectors: assert that with `ab_test.seeds=[0]`, exactly 1 MLflow run in `crash-severity-ml` tagged `seed=0`, `model_type=xgboost`; assert metrics `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`, `generalisation_gap` all logged; assert `per_class_matrix.json` artifact exists in the run; assert `models/best_ml_model.pkl` exists and is loadable as an `XGBClassifier`; assert `mlflow.sklearn.autolog()` was NOT used (no autolog params present). Run — confirm FAIL.
- [ ] T036 [US4] **GREEN** — Rewrite `src/train_ml/trainer.py`: implement `MLTrainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test) → MLTrainResult`; `mlflow.sklearn.autolog(disable=True)`; loop over seeds; for each seed: `XGBClassifier(objective='multi:softprob', num_class=3, random_state=seed, early_stopping_rounds=10)`; compute `sample_weight = compute_class_weights(y_train, n_classes=3)` from `src/metrics.py`; `clf.fit(Z_train, y_train, sample_weight=..., eval_set=[(Z_val, y_val)])`; start MLflow run with tags; log params + mandatory metrics + `per_class_matrix(y_test, y_pred, ['PDO','Injury','Fatal'])` as JSON artifact; track best seed by `eout_macro_f1`; return `MLTrainResult`
- [ ] T036b [US4] **GREEN** — Rewrite `src/train_ml/run.py`: load config; read `TRAIN_Z_PATH`, `TRAIN_Y_PATH`, `VAL_Z_PATH`, `VAL_Y_PATH`, `TEST_Z_PATH`, `TEST_Y_PATH`, `MODEL_OUTPUT_PATH` env vars; instantiate `MLTrainer(config.mlflow, config.model, config.ab_test.seeds)`; call `.train()`; save best pkl; exit 0/1
- [ ] T037 [US4] Run `dvc repro train_ml` — confirm 10 MLflow runs in `crash-severity-ml`; open MLflow UI, inspect `per_class_matrix.json` artifact for one run; confirm Fatal class metrics present

### MLP Training (train_dl) — see Phase 4B (T114–T120)

> **T038–T044 are superseded. MLP implementation is in Phase 4B (T114–T120) using Z-space inputs from the encode stage. Do not implement T038–T044.**

- [ ] T038 ~~SUPERSEDED~~
- [ ] T039 ~~SUPERSEDED~~
- [ ] T040 ~~SUPERSEDED~~
- [ ] T041 ~~SUPERSEDED~~
- [ ] T042 ~~SUPERSEDED~~
- [ ] T043 ~~SUPERSEDED~~
- [ ] T044 ~~SUPERSEDED~~

### Evaluate + Register

- [ ] T046 [US4] **RED** — Rewrite `tests/test_evaluate.py`: mock N=3 MLflow runs per experiment with `eout_macro_f1` scores above and below thresholds; assert exit 0 + `docs/ab_test_comparison.json` contains `p_value`, `cohens_d`, `ci_ml`, `ci_dl`, `winner`, `significant`, `gates_passed`; **assert `gates_passed=false` when winner mean F1 ≤ 0.45 or winner mean fatal recall ≤ 0.30**; assert exit 1 when gates fail. Run — confirm FAIL.
- [ ] T047 [US4] **GREEN** — Implement `ABEvaluator.evaluate()` in `src/evaluate/evaluator.py`: query MLflow for `eout_macro_f1` from `crash-severity-ml` (N=10 seeds, XGBoost on Z-space) and `crash-severity-dl` (N=10 seeds, MLP on Z-space); Welch's t-test on the two F1 distributions; Cohen's d; 95% CIs; if p ≥ 0.05 → XGBoost wins (tiebreak); assert winner's `mean_macro_f1 > 0.45` AND `mean_fatal_recall > 0.30`; return `EvaluationResult(winner, p_value, cohens_d, ml_mean_f1, dl_mean_f1, ...)`. Create `src/evaluate/run.py`: load config; instantiate `ABEvaluator(config.mlflow, config.ab_test, config.model)`; call `.evaluate()`; write `docs/evaluation_report.json` and `docs/ab_test_comparison.json`; exit 1 if gates fail
- [ ] T048 [US4] Run `dvc repro evaluate` — confirm JSON output has all required fields; verify `gates_passed` and winner printed to stdout
- [ ] T049 [US4] **RED** — Write `tests/test_register.py`: assert with `gates_passed=true` → exit 0, `models:/crash-severity@champion` resolvable, `models/registry_receipt.json` exists; assert with `gates_passed=false` → exit 1, no registry mutation. Run — confirm FAIL.
- [ ] T050 [US4] **GREEN** — Implement `ModelRegistrar.register(winner, run_id)` in `src/register/registrar.py`; create `src/register/run.py`. **⚠️ Inference path constraint**: the classifier was trained on Z vectors (8-dim latent space), not raw X — the registered `mlflow.pyfunc` artifact must bundle both the VAE encoder checkpoint (`models/vae_encoder.pth`) and the champion classifier so that `model.predict(X_raw)` applies `LatentEncoder → classifier` internally; a bare classifier artifact is not a deployable unit
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
- [ ] T056 [US5] **RED + GREEN** — Rewrite `src/tune/trial.py`: accepts `--beta <float>` and `--winner <ml|dl>` as argparse args; loads all X/y splits from PVC paths; instantiates `DVAETrainer` with candidate β; calls `trainer.train(X_all)` to get encoder checkpoint; loads `X_train_augmented` (pre-generated by augment stage, not re-run per trial); instantiates `LatentEncoder` with the new checkpoint; calls `encoder.encode(X_train_augmented, y_train_augmented, X_val, X_test)` (no LSA — CTGAN augmentation already applied); trains winner classifier on `Z_train_augmented` (1 seed, seed=0 for trials); evaluates on **Z_val** (not Z_test — constitution II); logs full trial to MLflow `crash-severity-tune` tagged `beta=<value>`, `winner=<ml|dl>`, `trial_type=katib`; **prints `val_macro_f1=<float>` on last stdout line** (required by Katib metrics collector — this is the fitness signal); exits 0
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

**Purpose**: Assert all constitutional gates; update documentation; verify full reproducibility; add latent space drift detection.

- [ ] T075 [P] Assert constitutional gates on final evaluation report: `python -c "import json; r=json.load(open('docs/evaluation_report.json')); print('F1:', 'PASS' if r['winner_macro_f1']>0.45 else 'FAIL'); print('Fatal recall:', 'PASS' if r['winner_fatal_recall']>0.30 else 'FAIL')"`
- [ ] T076 [P] Update `CLAUDE.md`: update architecture table (add `train_vae` + `encode` rows); update pipeline description to 10-stage; update DL section (remove EvoTorch NAS, remove FlexMLP, add MLP on Z description); update featurize section (3-class target encoding)
- [ ] T077 [P] Add to `.gitignore`: `data/processed/Z_*.npy`, `data/processed/y_train_augmented.npy`, `models/vae_*.pth`, `models/registry_receipt.json`
- [ ] T078 Commit all tracked files: `dvc.yaml`, `params.yaml`, `src/`, `great_expectations/gx/expectations/`, `pipelines/kubeflow/pipeline.py`, `docker/Dockerfile`, `k8s/`
- [ ] T079 [P] Full reproducibility smoke test: delete `data/processed/` and `models/`; run `dvc pull && dvc repro`; confirm all 10 stages complete and artifacts restored
- [ ] T080 [P] Remove `apache-airflow` from `pyproject.toml` (if present); run `uv sync`; confirm no import errors

### Latent Space Drift Detection (US7)

**Goal**: Detect covariate drift in production by comparing new batch latent representations against the training reference distribution saved at VAE training time.

**Mechanism**: At `train_vae` time, encode X_train with the frozen encoder → save per-dim μ statistics as a reference. At `encode` time, compare the new batch μ vectors against the reference using ELBO and MMD. Results logged to MLflow and written as a JSON report — advisory, does not halt the pipeline.

**Why latent space**: The VAE compresses raw 27-column input into a regularized 32-dim space. Shift in that space is a more sensitive and model-relevant drift signal than raw column statistics.

- [ ] T096 [US7] Extend `train_vae` to save drift reference: after training, encode full X_train with frozen encoder in eval mode → compute per-dim μ_mean and μ_std → save to `models/drift_reference.npz`; add `drift_reference.npz` to `dvc.yaml` train_vae `outs`; add `drift` section to `params.yaml` (`elbo_threshold: 0.5`, `mmd_threshold: 0.1`) and `DriftConfig(elbo_threshold, mmd_threshold)` to `src/config.py`
- [ ] T097 [US7] Create `src/drift/` package and `src/drift/detector.py`: `DriftResult` dataclass (`elbo_score: float`, `mmd_score: float`, `is_drifted: bool`, `n_samples: int`); `DriftDetector(reference_path, encoder_path, vae_config, drift_config)` with single public method `detect(X_new: np.ndarray) -> DriftResult`; load frozen encoder + decoder; compute mean ELBO over X_new batch; compute MMD between new μ vectors and reference μ sample using RBF kernel (`bandwidth=1.0`); set `is_drifted=True` if either metric exceeds its threshold
- [ ] T098 [US7] Extend `src/encode/run.py` to instantiate `DriftDetector` and call `.detect(X_all)` after encoding; log `drift_elbo`, `drift_mmd`, `drift_detected` (0/1) to the active MLflow run; write `docs/drift_report.json` (`elbo_score`, `mmd_score`, `is_drifted`, `n_samples`, `elbo_threshold`, `mmd_threshold`); print `[WARN] DRIFT DETECTED — review drift_report.json` if flagged — advisory only, exit 0 regardless
- [ ] T099 [US7] Run `dvc repro encode` → confirm `docs/drift_report.json` written with all fields; open MLflow → `crash-severity-vae` experiment → confirm `drift_elbo` and `drift_mmd` logged on the encode run; verify `is_drifted=false` on training data (self-reference should not trigger drift)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup) ✅
  └── Phase 2 (Foundational) ✅ + T081-T085 extensions
        └── Phase 3A (US1: Featurize TDD) ✅
              ├── Phase 3b (Data Contract) ✅
              │     └── Phase 3B (US2: GE Validate) ✅
              └── Phase 3C (DVC Pipeline Integration) ✅
                    └── Phase 3F (Config Foundation: T101-T103)  ← BLOCKER
                          ├── Phase 3G (Featurize Cyclical: T104-T105) [US3]
                          │     └── Phase 3H (train_vae KL Annealing: T106-T107) [US3]
                          │           ├── Phase 3E (encode: T091-T095) [US3] ← NEEDS REVISION
                          │           └── Phase 4A (augment: T108-T113) [US4]
                          │                 └── (feeds encode: augment + train_vae → encode)
                          │                       ├── Phase 4 (train_ml: T035-T037) [US4]
                          │                       └── Phase 4B (train_dl MLP: T114-T120) [US4]
                          │                             └── (join) → evaluate → register
                          │                                   └── Phase 4b (US5: Katib β-HPO)
                          └── Phase 5 (US6: KFP)
  Phase 6 (Polish) — after all stories complete
```

### Stage Execution Order (DVC DAG)

| Order | Stage | Parallel with | Blocks |
|---|---|---|---|
| 1 | validate | — | ingest |
| 2 | ingest | — | featurize |
| 3 | featurize | — | train_vae, augment |
| 4a | train_vae | augment | encode |
| 4b | augment | train_vae | encode |
| 5 | encode | — | train_ml, train_dl |
| 6a | train_ml | train_dl | evaluate |
| 6b | train_dl | train_ml | evaluate |
| 7 | evaluate | — | tune, register |
| 8 | tune | — | (invalidates train_vae → register) |
| 9 | register | — | — |

> `train_vae` and `augment` run in parallel (both depend on featurize only). `encode` waits for both. `train_ml` and `train_dl` run in parallel (both on Z-space from encode).

### Parallel Opportunities

- T081–T085 (Phase 2 extensions) — all parallel
- T023–T026 (DVC verification) — T023/T024 sequential; T025/T026 parallel after T024
- T101, T102, T103 (Phase 3F config foundation) — T102 and T103 parallel after T101
- T104, T106 (Phase 3G featurize + Phase 3H VAE test-writing) — parallel after T101–T103
- T108, T109 (augment package + dvc.yaml) — parallel after T101–T103
- T114, T115 (train_dl package + dvc.yaml) — parallel after T101–T103
- T035 (train_ml RED) — parallel with T110, T116 (augment + MLP RED tests) after Phase 3G/3H complete
- T066–T070 (Docker + Kubernetes setup) — T066/T067 sequential; T068–T070 sequential after T067
- T075, T076, T077, T079, T080 (Polish) — all parallel

---

## Implementation Strategy

### MVP First (US1 — DVC Reproducibility) ✅ COMPLETE

Phases 1–3C are done. Current entry point is Phase 3F.

### Incremental Delivery (CTGAN + MLP track)

1. **Phase 3F** (T101–T103) — config foundation; blocks everything downstream
2. **Phase 3G** (T104–T105) — cyclical encoding in featurize; re-runs train_vae + encode
3. **Phase 3H** (T106–T107) — KL annealing in train_vae; fixes posterior collapse
4. **Phase 3E** (T091–T095) — encode stage revision; drops LSA; uses X_train_augmented
5. **Phase 4A** (T108–T113) — augment stage; CTGAN Fatal class augmentation on X_train
6. **Phase 4 XGBoost** (T035–T037) — train_ml on Z-space (parallel with 4B)
7. **Phase 4B** (T114–T120) — train_dl shallow MLP on Z-space
8. **Phase 4 Evaluate** (T046–T051) — Welch's t-test; XGBoost vs MLP; register winner
9. **Phase 4b** (T052–T060) → β tuned via Katib (US5 checkpoint)
10. **Phase 5** → KFP orchestration (US6 checkpoint)
11. **Phase 6** → constitutional gates asserted; full reproducibility confirmed

### Parallel Team Strategy

After Phase 2 extensions (T081–T085) complete:
- **Track A**: US1 → US2 → US3 (VAE + encode) → US4 (A/B test) → US5 (Katib)
- **Track B** (after Docker image built in T066–T067): US6 (KFP)

---

## Notes

- TDD required for all `src/` code (constitution XV) — RED test MUST fail before GREEN implementation
- `train_ml` and `train_dl` are sequential within their DVC stage (N seeds in a loop) but run in parallel in the DVC DAG
- Both XGBoost and MLP train on `Z_train_augmented` (VAE latent features, 8-dim) — same Z-space, different model families; Welch's t-test on eout_macro_f1
- CTGAN augmentation operates on Fatal rows of `X_train.npy` only (training split); `X_val` and `X_test` MUST NOT be augmented (constitution III v3.3.0)
- encode stage receives `X_train_augmented` from the augment stage — no LSA inside encode; augmentation is fully handled upstream
- KL beta schedule: starts at 0.0, ramps linearly to `vae.beta_max` over `vae.warmup_epochs`, then held constant — prevents posterior collapse
- Shallow MLP architecture (constitution IV): `Input(8) → Linear(64) → ReLU → Dropout(0.1) → Linear(3)`; cross-entropy loss with runtime-computed class weights
- Katib trials retrain full VAE + encode + winner classifier per trial — each trial may take 5–15 min; plan for 25–75 min total tune stage
- `val_macro_f1` (Katib fitness, Z_val) and `eout_macro_f1` (final test metric, Z_test) are different numbers — never conflate
- Z_val and Z_test MUST NOT be augmented (constitution III v3.3.0)
- The `register` stage has no `outs` that DVC tracks — `registry_receipt.json` is the only DVC-tracked output
- After Katib writes `tune.best_params.beta` to `params.yaml`, DVC detects the change and invalidates `train_vae` and all downstream stages — a full re-run follows automatically
