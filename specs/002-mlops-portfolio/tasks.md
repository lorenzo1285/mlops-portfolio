# Tasks: MLOps Learning Portfolio — VAE-Based Crash Severity Pipeline

**Constitution**: v3.4.0 | **Architecture**: 10-stage DVC pipeline | **Target**: crash severity 3-class (PDO / Injury / Fatal)

---

## MLOps Lifecycle Map

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   ┌──────────┐      ┌──────────────────┐      ┌────────────────┐   │
│   │          │      │                  │      │                │   │
│   │  DESIGN  │─────▶│ MODEL DEVELOPMENT│─────▶│  OPERATIONS    │   │
│   │          │      │                  │      │                │   │
│   └──────────┘      └──────────────────┘      └────────────────┘   │
│        ▲                    ▲                         │             │
│        │                    └─────────────────────────┘             │
│        │                      Optuna HPO: gates fail →              │
│        │                      re-tune VAE → re-run pipeline         │
│        │                                                             │
│        └─────────────────────────────────────────────────────────── │
│          Drift detected → revisit data contract / feature set       │
└─────────────────────────────────────────────────────────────────────┘
```

**Current position**: O3.5 (Optuna HPO) → if gates fail → M5 (model fixes).  
**Current metrics**: val recall=52.4% ✅, test recall=33.3% ❌ — 4 more correct Fatal predictions needed.

---

## ▶️ WHAT TO DO NEXT — Step-by-Step Execution Order

> Follow this list top to bottom. Each step tells you exactly what to do and where to go next.

### STEP 1 — Optuna HPO scaffold (done ✅)

| Done | Task | Action |
|------|------|--------|
| ✅ | T126a | `uv add optuna` |
| ✅ | T126b | Add `tune.optuna.*` to `params.yaml` |
| ✅ | T126c | Add `OptunaConfig` to `src/config.py` |

### STEP 2 — DVAETrainer pruning hook (TDD)

| Done | Task | Action |
|------|------|--------|
| ✅ | **T127a** | **RED** — write failing tests in `tests/test_train_vae.py` |
| ✅ | **T127b** | **GREEN** — add `optuna_trial=None` param to `DVAETrainer.train()` |

### STEP 3 — OptunaTuner class (TDD)

| Done | Task | Action |
|------|------|--------|
| ✅ | **T128a** | **RED** — write failing tests in `tests/test_optuna_tuner.py` |
| ✅ | **T128b** | **GREEN** — create `src/tune/optuna_tuner.py` |

### STEP 4 — Wire into pipeline + run

| Done | Task | Action |
|------|------|--------|
| ✅ | **T129** | Update `src/tune/run.py` to use `OptunaTuner` |
| ✅ | T130 | Update `dvc.yaml` tune stage params |
| ✅ | **T131** | `uv run dvc repro tune` — smoke test |
| ✅ | **T132** | `uv run dvc repro` full pipeline — **check gates** |

---

### ⚖️ DECISION POINT after T132

```
eout_fatal_recall > 0.50  AND  eout_macro_f1 > 0.35
        YES  →  go to STEP 9 (Register)
        NO   →  go to STEP 5 (Model Fixes)
```

---

### STEP 5 — Fix A: MLP Balanced Focal Loss *(cheapest fix — try first)*

| Done | Task | Action |
|------|------|--------|
| 🔜 | T133a | Add `dl.focal_loss_enabled: false` + `dl.focal_loss_gamma: 2.0` to `params.yaml`; extend `DLConfig` |
| | T133b | **RED** — `BalancedFocalLoss` tests in `tests/test_train_dl.py` |
| | T133c | **GREEN** — add `BalancedFocalLoss` to `src/metrics.py`; wire into `DLTrainer` behind flag |
| | T133d | Set `focal_loss_enabled: true`; `uv run dvc repro train_dl evaluate` → **check gates** |

> Gates pass → STEP 9. Gates fail → STEP 6.

### STEP 6 — Fix B: Cyclical KL Annealing *(VAE cascade)*

| Done | Task | Action |
|------|------|--------|
| ✅ | T135a | Add `vae.cyclical_annealing: false` + `vae.cycle_epochs: 50` to `params.yaml`; extend `VAEConfig` |
| ✅ | T135b | **RED** — cyclical schedule tests in `tests/test_vae_trainer.py` |
| ✅ | T135c | **GREEN** — add cyclical branch to `DVAETrainer.train()` at `vae_trainer.py:276` |
| | T135d | Set `cyclical_annealing: true`; `uv run dvc repro train_vae encode train_ml train_dl evaluate` → **check gates** |

> Gates pass → STEP 9. Gates fail → STEP 7.

### STEP 7 — Fix C: Danger Index Features *(6-stage cascade)*

| Done | Task | Action |
|------|------|--------|
| ✅ | T123a | Leakage audit — confirm safe/unsafe columns; record in `docs/data_contract.md` |
| ✅ | T123b | Config + GE contract (`validation.columns.NUMOFVEHIC`) + DVC wiring; extend `FeaturesConfig` |
| ✅ | T123c | **RED** — 4 tests: leakage guard raises; cols absent when disabled; +2 cols when enabled; forbidden never in output |
| ✅ | T123d | **GREEN** — `FORBIDDEN_COLUMNS` constant + constructor guard + `_compute_danger_index()` + `_select_and_recode` + `run.py` |
| ✅ | T123e | `uv run dvc repro featurize train_vae encode train_ml train_dl evaluate` → **GATES FAILED** (ML macro F1=0.3555 ✅, fatal recall=0.3830 ❌) → Fix D |

> Gates pass → STEP 9. Gates fail → STEP 8.

### STEP 8 — Fix D: XGBoost Focal Loss *(last automated resort)*

| Done | Task | Action |
|------|------|--------|
| ✅ | T125a | Add `model.focal_loss_enabled: false` + `model.focal_loss_gamma: 2.0` to `params.yaml`; extend `ModelConfig` |
| ✅ | T125b | **RED** — custom obj callable tests in `tests/test_train_ml.py` |
| ✅ | T125c | **GREEN** — add `focal_loss_grad_hess()` to `src/metrics.py`; wire into `MLTrainer` |
| ✅ | T125d | `uv run dvc repro evaluate` → **GATES FAILED** (ML macro F1=0.3560 ✅, fatal recall=0.3801 ❌) → Fix E |

> Gates pass → STEP 9. Gates fail → Fix E/F require constitution amendments — stop and discuss.

---

### STEP 9 — Register (once gates pass)

| Done | Task | Action |
|------|------|--------|
| | T051 | `uv run dvc repro register` — `@champion` alias set; receipt written |

---

## DVC Stage Execution Order

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
| 8 | tune | — | writes best params → invalidates train_vae downstream |
| 9 | register | — | — |

---

# 🎨 DESIGN

> Requirements Engineering · ML Use-Case Prioritisation · Data Availability Check

---

## ✅ Phase D1: Requirements + Spec

- [x] T001 Create all required directories
- [x] T002 [P] Add dependencies to `pyproject.toml`
- [x] T003 [P] Create initial `params.yaml`
- [x] T004 [P] Create `.dvcignore`
- [x] T005 [P] Create `src/__init__.py` and stage `__init__.py` files
- [x] T006 [P] Create `.gitattributes`
- [x] T011 [P] Create `src/config.py` typed dataclasses
- [x] T016 [P] Create stub class modules for all stages
- [x] T081 [P] Create `src/train_vae/` and `src/encode/` packages
- [x] T082 [P] Extend `params.yaml` with `vae.*`, `augment.*`, `dl.*`
- [x] T083 [P] Extend `src/config.py` with `VAEConfig`, `AugmentConfig`, `DLConfig`
- [x] T101 Amend constitution III to v3.3.0 — add CTGAN augmentation as third imbalance mechanism

## ✅ Phase D2: Data Contract + Validation (GE)

**Goal**: data quality enforced before any training stage; pipeline halts on contract violation.

- [x] T027 Create `docs/data_contract.md` — dtype, range, nulls, sentinels for all columns
- [x] T028 [P] Encode contract into `params.yaml validation.*`; update `ValidationConfig`
- [x] T029a Refactor `GEContextBuilder` to infrastructure only; rewrite `GEManager.build_suite()` with `row_condition` sentinel exclusion
- [x] T029 **RED** — `tests/test_validate.py`
- [x] T029b **GREEN** — `ge_checkpoint_runner.py`
- [x] T030 **GREEN** — `DataValidator.validate()` + `src/validate/run.py`
- [x] T007 `dvc init`; T008 configure DVC remote; T009 track raw dataset
- [x] T010 Initialise GE v1 context
- [x] T031 `dvc repro validate` — exit 0; Data Docs created
- [x] T032 Inspect Data Docs — all 54 expectations pass
- [x] T033 Failure path — SPEEDLIMIT=500 → exit 1; expectation name in stdout
- [x] T034 [P] Commit GE suite to git

> **Loop back here if**: drift detection (Phase O5) flags a new column distribution shift, or if a new data source is added.

---

# ⚙️ MODEL DEVELOPMENT

> Data Engineering · ML Model Engineering · Model Testing & Validation

---

## ✅ Phase M1: Data Engineering — Featurize

**Goal**: 3-way split; 4-group `ColumnTransformer`; cyclical encoding; sample complexity gate.

- [x] T012 [P] `src/ingest/ingester.py`
- [x] T013 [P] Add `feature_selection` to `params.yaml`
- [x] T014 [P] `src/featurize/selector.py`
- [x] T015 [P] `src/featurize/featurizer.py`
- [x] T017 [P] `src/metrics.py` — `make_eval_dataset`, `per_class_matrix`, `compute_class_weights`
- [x] T084 [P] Extend `src/metrics.py`
- [x] T085 [P] Stub class modules for `train_vae`, `encode`
- [x] T018 **RED** — `tests/test_ingest.py`
- [x] T019 **GREEN** — `src/ingest/run.py`
- [x] T020 **RED** — `tests/test_featurize.py`
- [x] T021 **GREEN** — `src/featurize/featurizer.py` 3-class target encoding
- [x] T102 [P] Update `params.yaml` — cyclical columns, VAE annealing, augment, dl sections
- [x] T103 [P] Update `src/config.py` — annealing fields, `AugmentConfig`
- [x] T104 **RED** — cyclical column assertions in `tests/test_featurize.py`
- [x] T105 **GREEN** — `_apply_cyclical()` in `src/featurize/featurizer.py`
- [x] T022 Create `dvc.yaml` — all 10 stages
- [x] T023 `dvc repro featurize` — all arrays + joblib written
- [x] T024 Verify caching — all stages report cached
- [x] T025 [P] Verify param-triggered re-run
- [x] T026 [P] `dvc push`

## ✅ Phase M2: ML Model Engineering — VAE + Augment + Encode

**Goal**: denoising β-VAE learns latent representation; CTGAN augments Fatal class; frozen encoder projects all splits to Z-space.

- [x] T086 **RED** — `tests/test_train_vae.py`
- [x] T087 **GREEN** — `src/train_vae/vae_trainer.py` — Denoising β-VAE; Encoder/Decoder; reparameterize
- [x] T088 **GREEN** — `src/train_vae/run.py`
- [x] T089 `dvc repro train_vae` — `vae_encoder.pth` + `vae_decoder.pth` written
- [x] T090 MLflow UI — `vae_elbo` decreasing trend confirmed
- [x] T106 **RED** — KL annealing assertions in `tests/test_train_vae.py`
- [x] T107 **GREEN** — linear β warmup in `DVAETrainer.train()`; `kl_beta` logged per epoch
- [x] T121 **GREEN** — `WeightedRandomSampler` in `DVAETrainer` — Fatal rows get proportional gradient share
- [x] T108 [P] Create `src/augment/` package; add `ctgan` dep
- [x] T109 [P] Add `augment` stage to `dvc.yaml`
- [x] T110 **RED** — `tests/test_augmenter.py`
- [x] T111 **GREEN** — `src/augment/augmenter.py` — `CTGANAugmenter` fits TVAE on Fatal rows
- [x] T112 **GREEN** — `src/augment/run.py`
- [x] T113 `dvc repro augment` — `X_train_augmented.npy` + `y_train_augmented.npy`; fatal fraction ≥ 0.05
- [x] T091 **RED** — rewrite `tests/test_encode.py`
- [x] T092 **GREEN** — rewrite `src/encode/encoder.py` — `LatentEncoder` μ-path; deterministic
- [x] T093 **GREEN** — rewrite `src/encode/run.py`
- [x] T094 `dvc repro encode` — Z arrays written; `Z_train_augmented.shape[1] == latent_dim`
- [x] T095 [P] Val/test isolation confirmed — no augmentation leak into Z_val / Z_test
- [x] T100 `notebooks/vae_eda.ipynb` — 7 VAE diagnostic visualisations

## ✅ Phase M3: ML Model Engineering — Classifiers (MLP + XGBoost)

**Goal**: two competing classifiers trained on Z-space; 10 seeds each; mandatory metrics logged; visual diagnostics in MLflow.

- [x] T114 [P] Verify `src/train_dl/` package
- [x] T115 [P] Update `dvc.yaml` `train_dl` stage
- [x] T116 **RED** — `tests/test_train_dl.py`
- [x] T117 **GREEN** — `src/train_dl/trainer.py` — `ShallowMLP` + `DLTrainer`
- [x] T118 **GREEN** — `src/train_dl/run.py`
- [x] T119 `dvc repro train_dl` — 10 runs in `crash-severity-dl`; `models/mlp_model.pth` written
- [x] T120 MLflow UI — mandatory metrics + `per_class_matrix.json` + confusion matrix + ROC confirmed
- [x] T035 **RED** — `tests/test_train_ml.py`
- [x] T036 **GREEN** — `src/train_ml/trainer.py` — `MLTrainer`
- [x] T036b **GREEN** — `src/train_ml/run.py`
- [x] T037 `dvc repro train_ml` — 10 runs in `crash-severity-ml`; `models/best_ml_model.pkl` written

## ✅ Phase M4: Model Testing + Validation — Evaluate

**Goal**: Welch's t-test on macro F1 distributions; constitutional gates enforced; winner declared.

- [x] T046 **RED** — `tests/test_evaluate.py`
- [x] T047 **GREEN** — `ABEvaluator.evaluate()` in `src/evaluate/evaluator.py` + `src/evaluate/run.py`
- [x] T048 `dvc repro evaluate` — gates fail (expected); `evaluation_report.json` written

---

## 🔜 Phase M5: VAE Fatal Recall Fixes

**Goal**: hit `eout_fatal_recall > 0.50`. Triggered by O3.5 Optuna HPO — if gates still fail after Optuna rewrites `params.yaml` and runs full pipeline, apply fixes below in order. Stop as soon as gates pass.

**Rules**:
- Stop as soon as gates pass — do not implement remaining fixes
- Constitution XV: every `src/` change requires RED before GREEN
- Constitution XIV: all flags and thresholds in `params.yaml` — no magic numbers

**DVC cascade cost per fix**:

| Fix | Stages invalidated |
|---|---|
| Fix A — MLP Focal Loss | train_dl → evaluate |
| Fix B — Cyclical KL Annealing | train_vae → encode → train_ml → train_dl → evaluate |
| Fix C — Danger Index Features | featurize → train_vae → encode → train_ml → train_dl → evaluate |
| Fix D — XGBoost Focal Loss | train_ml → evaluate |
| Fix E — Supervised Latent Loss ⛔ | train_vae → encode → train_ml → train_dl → evaluate |
| Fix F — Tomek Links ⛔ contingency | train_ml → evaluate |

---

### ✅ Fix A — MLP Balanced Focal Loss

*No upstream cascade — only `train_dl` reruns.*  
*Formula*: `FL = −α_t · (1 − p_t)^γ · log(p_t)` where `α_t` = class weight, `p_t` = predicted probability for true class.

- [x] T133a [P] Add `dl.focal_loss_enabled: false` and `dl.focal_loss_gamma: 2.0` to `params.yaml`; extend `DLConfig` in `src/config.py`.
- [x] T133b **RED** — `tests/test_train_dl.py`: assert `BalancedFocalLoss(gamma=2.0, weight=w)(logits, targets)` returns scalar; assert focal property (confident correct loss < hard incorrect loss); assert `DLTrainer` uses `BalancedFocalLoss` when enabled, `CrossEntropyLoss` when not.
- [x] T133c **GREEN** — Add `BalancedFocalLoss(nn.Module)` to `src/metrics.py`: `forward` computes softmax → gather `p_t` → apply `−α_t · (1−p_t)^γ · log(p_t)` → mean. Swap into `DLTrainer._train_single_seed()` at `trainer.py:119` behind `focal_loss_enabled` flag.
- [x] T133d Set `dl.focal_loss_enabled: true`; `uv run dvc repro train_dl evaluate`. **Gates pass → T051. Gates fail → Fix B.**

---

### ✅ Fix B — Cyclical KL Annealing

*Triggers VAE retrain cascade.*  
*Schedule*: `β_t = beta_max × min(1, (epoch % cycle_epochs) / warmup_epochs)` — resets β to 0 every `cycle_epochs`.

- [x] T135a [P] Add `vae.cyclical_annealing: false` and `vae.cycle_epochs: 50` to `params.yaml`; extend `VAEConfig`.
- [x] T135b **RED** — `tests/test_vae_trainer.py`: assert cyclical formula gives `β=0.0` at epoch 0, `β=beta_max` at epoch `warmup_epochs`, `β=0.0` at epoch `cycle_epochs`; assert monotonic schedule unchanged when `cyclical_annealing=False`.
- [x] T135c **GREEN** — In `DVAETrainer.train()`, replace `beta_t` expression at `vae_trainer.py:276` with branch: cyclical formula when `cyclical_annealing=True`, existing monotonic formula otherwise.
- [X] T135d Set `vae.cyclical_annealing: true`; `uv run dvc repro train_vae encode train_ml train_dl evaluate`. **Gates pass → T051. Gates fail → Fix C.**

---

### Fix C — Danger Index Feature Engineering

*Full 6-stage cascade: featurize → train_vae → encode → train_ml → train_dl → evaluate.*

- [x] T123a **Leakage Audit** — confirm `NUMOFUNINJ` is post-crash leakage (exclude); confirm `NUMOFVEHIC`, `SPEEDLIMIT`, `DRIVER1AGE` are pre-crash (safe). Record in `docs/data_contract.md`.
- [x] T123b [P] Config + GE contract + DVC wiring:
  - Add `features.danger_index_features: false` and `features.forbidden_columns: [NUMOFKILL, NUMOFINJ, NUMOFUNINJ]` to `params.yaml`
  - Add `NUMOFVEHIC: {dtype: int, min: 1, max: 6, mostly: 1.0}` to `validation.columns` in `params.yaml` (GE source-column contract — Constitution VIII)
  - Extend `FeaturesConfig` with `danger_index_features: bool = False` and `forbidden_columns: list[str]`
  - Add `features.danger_index_features` + `features.forbidden_columns` to featurize `params:` in `dvc.yaml`
- [x] T123c **RED** — `tests/test_featurize.py` — 4 tests using real data from `data/processed/raw.csv`, direct `Featurizer` instantiation:
  - `test_leakage_guard_raises_on_forbidden_in_features` — construct with `feature_cols=['NUMOFUNINJ', ...]` → assert `ValueError`
  - `test_danger_index_columns_absent_when_disabled` — `danger_index_features=False` → `X_train.shape[1]` equals baseline
  - `test_danger_index_columns_present_when_enabled` — `danger_index_features=True` → `X_train.shape[1] == baseline + 2`
  - `test_leakage_columns_never_in_output` — `FeaturizeResult.feature_cols` contains no forbidden columns
- [x] T123d **GREEN** — In `Featurizer`:
  - Add `FORBIDDEN_COLUMNS = frozenset(['NUMOFKILL', 'NUMOFINJ', 'NUMOFUNINJ'])` class constant
  - Add `danger_index_features: bool = False` to `__init__`; constructor guard raises `ValueError` on violation
  - Add `_compute_danger_index(df)`: computes two cols, drops `NUMOFVEHIC`, returns df
    - `solo_highspeed = ((NUMOFVEHIC == 1) & (SPEEDLIMIT >= 45)).astype(int)`
    - `vulnerability_interaction = (((DRIVER1AGE < 25) | (DRIVER1AGE > 70)) & (SPEEDLIMIT > 40)).astype(int)`
  - Modify `_select_and_recode`: when flag set, temporarily retain `NUMOFVEHIC`, call `_compute_danger_index` before split
  - Update `_fit_preprocess`: append derived cols to `num` group when present in `X_train.columns`
  - Update `src/featurize/run.py`: pass `danger_index_features=config.features.danger_index_features` to `Featurizer`
- [x] T123e Set `features.danger_index_features: true`; `uv run dvc repro featurize train_vae encode train_ml train_dl evaluate`. **Result: ML winner, macro F1=0.3555 (PASS), fatal recall=0.3830 (FAIL ≤0.50) → Fix D.**

---

### Fix D — XGBoost Balanced Focal Loss (Last Resort)

*Requires custom gradient/hessian derivation for XGBoost. High regression risk.*

- [x] T125a [P] Add `model.focal_loss_enabled: false` and `model.focal_loss_gamma: 2.0` to `params.yaml`; extend `ModelConfig`.
- [x] T125b **RED** — `tests/test_train_ml.py`: assert `MLTrainer` passes custom `obj` callable when enabled; assert callable returns `(grad, hess)` both shape `(N, K)` for dummy `(N, 3)` probability array.
- [x] T125c **GREEN** — Add `focal_loss_grad_hess(y_true_onehot, y_pred_proba, alpha, gamma) → (grad, hess)` to `src/metrics.py`; wire into `MLTrainer._train_single_seed()` behind flag; switch `eval_metric` to `merror`.
- [x] T125d `uv run dvc repro evaluate`. **Result: ML winner, macro F1=0.3560 (PASS), fatal recall=0.3801 (FAIL ≤0.50) → Fix E.**

---

### Fix E — Supervised Latent Loss ⛔ BLOCKED — Constitution II Amendment Required

*Do NOT write implementation code until T136 amendment is accepted.*

**Conflict**: VAE trains unsupervised on `X_all` (no labels). `L_CE` requires labels. Using val/test labels in VAE training violates constitution II. Proposed scope: `L_CE` on `X_train` rows only.

- [ ] T136 **Amendment** — Draft constitution II scoped exception in `.specify/memory/constitution.md` (v3.5.0): VAE may accept `y_train` for `L_CE` on `X_train` rows only; val/test rows remain unsupervised; add `gamma` to Optuna search space. Update `CLAUDE.md`. **Gate: do not proceed to T137a until accepted.**
- [ ] T137a [P] Add `vae.supervised_latent_loss: false` and `vae.gamma: 0.1` to `params.yaml`; extend `VAEConfig`; add `gamma` to `tune.optuna.search_space`.
- [ ] T137b **RED** — `tests/test_vae_trainer.py`: assert `L_CE` computed only on X_train rows; assert total loss = `L_rec + β·L_KL + γ·L_CE`; assert `y_all=None` still works when disabled.
- [ ] T137c **GREEN** — In `DVAETrainer.train()`: attach `nn.Linear(latent_dim, 3)` classification head; compute `L_CE` on X_train batches only; add `γ·L_CE` to total loss; log `vae_ce_loss` per epoch.
- [ ] T137d `uv run dvc repro train_vae encode train_ml train_dl evaluate`. **Gates pass → T051. Gates fail → Fix F (contingency).**

---

### Fix F — Tomek Link Cleaning ⛔ Contingency — Constitution III Amendment Required

*Only if all Fixes A–E exhausted. Most effective after upstream fixes have improved Z-space quality (active dims ≥ 3/8).*

- [ ] T138 **Amendment** — Draft constitution III amendment in `.specify/memory/constitution.md` (v3.6.0): add boundary-sharpening undersampling in Z-space as fourth permitted mechanism, contingent on ≥ 3/8 active dims in MLflow VAE audit. **Gate: do not proceed to T124a until accepted.**
- [ ] T124a [P] `uv add imbalanced-learn`.
- [ ] T124b **RED** — `tests/test_train_ml.py`: assert `TomekLinks().fit_resample(Z_train, y_train)` called before `clf.fit()`; cleaned arrays passed to classifier; `Z_val`/`Z_test` never resampled.
- [ ] T124c **GREEN** — In `MLTrainer._train_single_seed()`, before `clf.fit()`: `Z_tr, y_tr = TomekLinks().fit_resample(Z_train, y_train)`. Do NOT use `imblearn.Pipeline` — incompatible with XGBoost `eval_set`.
- [ ] T124d `uv run dvc repro evaluate`. **Gates pass → T051. Gates fail → no further automated fixes; escalate to new data acquisition.**

---

# 🚀 OPERATIONS

> ML Model Deployment · CI/CD Pipelines · Monitoring & Triggering

---

## ✅ Phase O1: Register (TDD — deferred run)

- [x] T049 **RED** — `tests/test_register.py`
- [x] T050 **GREEN** — `ModelRegistrar.register()` + `src/register/run.py`

> `dvc repro register` (T051) deferred until gates pass.

## ✅ Phase O2: Docker + Kubernetes

- [x] T066 Create `docker/Dockerfile`
- [x] T067 Build + smoke-test Docker image
- [x] T068 Enable Kubernetes in Docker Desktop
- [x] T069 Create `k8s/pvc.yaml` — hostPath PVC at `/app`
- [x] T070 Install KFP v2.0.5-pns + Katib v0.17.0; verify all pods running

## ✅ Phase O3: Katib HPO (Portfolio Reference)

**Katib remains part of the operational stack** — the CRD, trial script, and `HyperparamTuner` are kept as a portfolio demonstration of Kubernetes-native HPO. Replaced by Optuna as the *active* search engine (Phase O3.5) due to skopt crashes and pod scheduling overhead on a single-machine setup.

**Outcome**: 9/15 trials completed; best `beta_max=0.2`, `latent_dim=32`; fatal recall stuck at 0.25 → triggered Phase M5 fix plan.

- [x] T052 [P] Add `kubernetes>=28.0` to `pyproject.toml`
- [x] T053 [P] Create `k8s/katib/vae_experiment.yaml` — Experiment CRD; skopt algorithm; 15 trials
- [x] T056 **RED + GREEN** — `src/tune/trial.py` — shared trial logic used by both Katib and Optuna
- [x] T057 **RED** — `tests/test_tune.py`
- [x] T058 **GREEN** — `HyperparamTuner.tune()` in `src/tune/tuner.py` + `src/tune/run.py`
- [x] T059 `dvc repro tune` — Katib experiment completed; best params written to `params.yaml`
- [x] T060 `dvc repro` (full) — gates still FAIL; fatal_recall=0.25 → proceed to O3.5 + M5

## 🔜 Phase O3.5: Optuna HPO (Active Search Engine)

**Goal**: replace Katib as the active `tune` stage engine. Expanded 5-param continuous search + MedianPruner on per-epoch ELBO. `HyperparamTuner` and `k8s/katib/` kept — do not delete.

**Why Optuna alongside Katib**:
- Katib: Kubernetes-native, distributed-ready, portfolio demonstration ✅ done
- Optuna: local rapid iteration, continuous search space, pruning, no pod scheduling overhead

**Search space** (bounds in `params.yaml` under `tune.optuna.search_space`):

| Param | Sampler | Range |
|---|---|---|
| `beta_max` | log-uniform | [0.01, 1.0] |
| `latent_dim` | categorical | {8, 16, 32, 64} |
| `warmup_epochs` | int | [5, 30] |
| `lr` | log-uniform | [1e-4, 1e-3] |
| `dropout_p` | uniform | [0.05, 0.30] |

**Fitness**: `val_fitness = val_macro_f1 × (1.0 if val_fatal_recall ≥ 0.50 else 0.5)`  
**dl.input_dim sync**: when Optuna writes `vae.latent_dim`, it must also write `dl.input_dim = latent_dim`.  
**Data paths**: `_objective` loads featurized X arrays — not Z vectors — because each trial retrains VAE from scratch.

### O3.5a — Config Scaffold

- [x] T126a [P] `uv add optuna`; run `uv sync`.
- [x] T126b [P] Add `tune.optuna.*` section to `params.yaml`:
  ```yaml
  optuna:
    n_trials: 30
    study_name: vae-optuna-hpo
    direction: maximize
    pruner:
      n_startup_trials: 5
      n_warmup_steps: 15
    search_space:
      beta_max_low: 0.01
      beta_max_high: 1.0
      latent_dim_choices: [8, 16, 32, 64]
      warmup_epochs_low: 5
      warmup_epochs_high: 30
      lr_low: 0.0001
      lr_high: 0.001
      dropout_p_low: 0.05
      dropout_p_high: 0.30
  ```
- [x] T126c [P] Add `OptunaConfig` dataclass to `src/config.py`; add `optuna: OptunaConfig` field to `TuneConfig`.

### O3.5b — DVAETrainer Pruning Hook (TDD)

- [x] T127a **RED** — `tests/test_vae_trainer.py`: assert `DVAETrainer.train()` accepts optional `optuna_trial=None`; when `optuna_trial.should_prune()` returns `True` after epoch 1, assert `train()` raises `optuna.TrialPruned`; assert `optuna_trial.report(elbo, epoch)` called; assert `optuna_trial=None` completes normally.
- [x] T127b **GREEN** — In `DVAETrainer.train()`, after each epoch ELBO: if `optuna_trial is not None`, call `optuna_trial.report(elbo, epoch)`; if `optuna_trial.should_prune()`: raise `optuna.TrialPruned()`. Default `optuna_trial=None` — existing callers unaffected.

### O3.5c — OptunaTuner Class (TDD)

- [x] T128a **RED** — `tests/test_optuna_tuner.py`: patch `DVAETrainer`, `LatentEncoder`, `MLTrainer` to return fixture metrics; assert `OptunaTuner(...).tune()` returns `TuneResult` with all 5 param keys; assert `latent_dim` ∈ `{8, 16, 32, 64}`; assert `n_trials` matches config; assert `TrialPruned` inside `_objective` is caught as pruned (not crashed).
- [x] T128b **GREEN** — Create `src/tune/optuna_tuner.py`:
  - `__init__`: stores config; loads featurized X arrays (`X_train_augmented`, `X_val`, `X_test`, `y_train_aug`) at construction.
  - `_objective(trial)`: suggest 5 params → clone `VAEConfig` → `DVAETrainer(..., optuna_trial=trial).train()` → `LatentEncoder.encode()` → `MLTrainer(seeds=[0]).train()` → compute `val_fitness` → log to MLflow nested run tagged `trial_type=optuna` → return `val_fitness`.
  - `tune()`: `optuna.create_study(TPESampler, MedianPruner)` → `study.optimize` → return `TuneResult`.

### O3.5d — Wire into DVC + Run

- [ ] T129 [P] Update `src/tune/run.py`: import `OptunaTuner` alongside `HyperparamTuner` (keep both); use `OptunaTuner`; write all 5 best params + `dl.input_dim` to `params.yaml`.
- [ ] T130 [P] Update `dvc.yaml` tune stage `params:` list — add `tune.optuna.*` keys.
- [ ] T131 `uv run dvc repro tune` — study runs `n_trials`; MLflow shows runs tagged `trial_type=optuna`; `params.yaml` updated.
- [ ] T132 `uv run dvc repro` (full downstream) — inspect `docs/evaluation_report.json`. **Gates pass → T051. Gates fail → M5 fixes.**

---

### Gate: Register

- [ ] T051 `uv run dvc repro register` — prerequisite: gates passed from O3.5 or any M5 fix; `@champion` alias set; `mlflow.pyfunc.load_model("models:/crash-severity@champion")` loads; `model.predict(Z_test[:5])` returns shape `(5,)` with values in `{0, 1, 2}`.

---

## 🔜 Phase O4: CI/CD — KFP 10-Stage Pipeline

**Goal**: full pipeline compiled to `pipeline.yaml`; runs end-to-end on Docker Desktop Kubernetes.

- [ ] T071 Rewrite `pipelines/kubeflow/pipeline.py` with 10 `@dsl.component` functions — one per stage — each calling `subprocess.run(["dvc", "repro", "<stage>"], cwd="/app")`; mount PVC at `/app`; compile to `pipeline.yaml`.
- [ ] T072 `uv run python pipelines/kubeflow/pipeline.py` → `pipeline.yaml` created.
- [ ] T073 Upload to KFP UI; start run; confirm all 10 steps with correct dependency arrows.
- [ ] T074 Inspect pod logs — `train_vae` ELBO visible; `train_ml` run tagged `orchestrator=kubeflow`.

## 🔜 Phase O5: Monitoring — Latent Space Drift Detection

**Goal**: production drift detected via MMD on Z vectors. Advisory signal — does not halt pipeline.

- [ ] T096 Extend `train_vae` to save drift reference: `models/drift_reference.npz`; add to `dvc.yaml`; add `DriftConfig` to `src/config.py`.
- [ ] T097 Create `src/drift/detector.py` — `DriftDetector.detect(X_new) → DriftResult`; MMD with RBF kernel.
- [ ] T098 Extend `src/encode/run.py`: call `DriftDetector`; log `drift_elbo`, `drift_mmd`, `drift_detected`; write `docs/drift_report.json`; exit 0 always.
- [ ] T099 `uv run dvc repro encode` → `drift_report.json` written; `is_drifted=false` on training data.

> **Loop back here if**: `is_drifted=true` on production data → revisit data contract (D2) and featurize (M1).

## 🔜 Phase O6: Polish + Final Validation

- [ ] T075 [P] Assert constitutional gates on `docs/evaluation_report.json`.
- [ ] T076 [P] Update `CLAUDE.md` — architecture table, pipeline description.
- [ ] T077 [P] Update `.gitignore`.
- [ ] T078 Commit all tracked files.
- [ ] T079 [P] Full reproducibility smoke test: delete `data/processed/` + `models/`; `dvc pull && dvc repro` → all 10 stages complete.
- [ ] T080 [P] Remove `apache-airflow` from `pyproject.toml` if present; `uv sync`.

---

## Iteration Summary

| Trigger | Loop back to | Mechanism |
|---|---|---|
| evaluate gates FAIL | O3.5 + M5 | Optuna writes best params → `dvc repro` re-runs VAE → encode → classify → evaluate; if still failing, M5 fixes applied in order |
| Drift detected (O5) | Design (D2) | Review data contract; update GE suite; re-featurize |
| New data source | Design (D1 + D2) | Update spec; extend data contract; re-run full pipeline |
