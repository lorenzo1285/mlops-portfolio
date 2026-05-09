# Tasks: GMM Classifier — Third Parallel Training Branch

**Input**: Design documents from `specs/003-gmm-classifier/`  
**Branch**: `003-gmm-classifier`  
**Constitution**: v3.4.0 — TDD vertical slices; real data fixtures; ASCII terminal output; deep module architecture

## Format: `[ID] [P?] [Story?] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[US#]**: User story this task belongs to
- Constitution XV: each test task MUST produce a failing test BEFORE its paired implementation task runs

---

## Phase 1: Setup (Infrastructure)

**Purpose**: Create the package skeleton and wire the DVC DAG so the pipeline recognises the new stage.

- [x] T001 Create `src/train_gmm/__init__.py` (empty, establishes package)
- [x] T002 Add `train_gmm` stage to `dvc.yaml` (cmd, deps: Z_train_augmented, Z_val, Z_test, y_*, src/train_gmm/*, src/config.py; params: gmm.*, model.n_classes, model.macro_f1_threshold, model.fatal_recall_threshold, mlflow.experiment_name_gmm, mlflow.tracking_uri, ab_test.seeds; outs: models/best_gmm_model.pkl) and update `evaluate` stage deps to add `models/best_gmm_model.pkl` and param `mlflow.experiment_name_gmm`

---

## Phase 2: Foundational (Config + Pre-existing Bug Fix — Blocking Prerequisites)

**Purpose**: All config changes and the DLTrainer constitution violation fix that every story depends on. MUST complete before any story phase begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T003 Add `GMMConfig` dataclass to `src/config.py` (fields: `n_components: dict[int, int]` — per-class Gaussian count keyed by class label 0/1/2, `covariance_type: str`, `reg_covar: float`, `max_iter: int`, `n_init: int` — EM random restarts per class to eliminate degenerate solutions when n_components > 1, `fatal_prior_boost: float`, `experiment_name: str`) and add `gmm: GMMConfig` to `ProjectConfig`; extend `load_config()` to parse `raw["gmm"]`, converting YAML integer keys to Python `int` if needed
- [x] T004 Add `gmm:` section to `params.yaml` with values: `n_components: {0: 1, 1: 1, 2: 2}`, `covariance_type: "full"`, `reg_covar: 1.0e-6`, `max_iter: 100`, `n_init: 5`, `fatal_prior_boost: 1.0`, `experiment_name: "crash-severity-gmm"`
- [x] T005 Add `experiment_name_gmm: str` field to `MLflowConfig` dataclass in `src/config.py` and add `experiment_name_gmm: "crash-severity-gmm"` under `mlflow:` in `params.yaml`
- [x] T006 Change `ABTestConfig.tiebreak` from `str` to `list[str]` in `src/config.py`; update `params.yaml` from `tiebreak: "ml"` to `tiebreak: ["ml", "dl", "gmm"]`
- [x] T007 **[BUG FIX]** Fix `DLTrainer.train()` seed-selection in `src/train_dl/trainer.py`: line 93–94 currently selects best seed by `eout_macro_f1` (test set — constitution II violation); add `eval_macro_f1` computation from Z_val inside `_train_single_seed()`, log it to MLflow, and change the best-seed comparator in `train()` to use `eval_macro_f1` (val), matching the MLTrainer pattern. `eout_macro_f1` (test) must still be logged and returned for the ABEvaluator to read.

**Checkpoint**: `uv run python -c "from src.config import load_config; c = load_config(); print(c.gmm, c.mlflow.experiment_name_gmm, c.ab_test.tiebreak)"` prints without error. ✅

---

## Phase 3: User Story 1 — Train GMM on Latent Space (Priority: P1) MVP

**Goal**: `dvc repro train_gmm` produces `models/best_gmm_model.pkl` and logs per-seed metrics to `crash-severity-gmm` MLflow experiment.

**Independent Test**: `uv run dvc repro train_gmm` completes; `models/best_gmm_model.pkl` exists; MLflow experiment `crash-severity-gmm` has ≥1 finished run with all mandatory metrics.

### Slice A — GMMClassifier wrapper

- [x] T008 [US1] **[TDD-RED]** Write failing boundary test for `GMMClassifier` in `tests/test_train_gmm.py`: given real `Z_train_augmented.npy`, `y_train_augmented.npy`, `Z_val.npy`, `y_val.npy` fixtures — construct a `GMMClassifier`, call `predict(Z_val)`, assert output shape equals `(len(Z_val),)` and all values in `{0, 1, 2}`; assert `fatal_prior_boost > 1.0` increases fraction of Fatal (class 2) predictions vs boost=1.0. Test MUST fail (class does not exist yet). ✅
- [x] T009 [US1] **[TDD-GREEN]** Implement `GMMClassifier` in `src/train_gmm/trainer.py`: constructor accepts `gmms: dict[int, GaussianMixture]` (one per class, each fitted with per-class n_components), `log_priors: np.ndarray`, `fatal_prior_boost: float = 1.0`; `predict(Z)` computes `score_c = gmm_c.score_samples(Z) + log_prior_c` for each class (`fatal_prior_boost` multiplies the Fatal *prior in linear space*: `log(boost × P(2)) = log(boost) + log(P(2))` — do NOT multiply the log-prior directly, that inverts the direction for negative log values; `score_samples()` returns log-likelihood already summed over mixture components), returns `argmax` over classes as integer array. Expose `predict()` interface compatible with `self._classifier.predict(Z)` in `CrashSeverityPyfunc`. ASCII-only logging. ✅

### Slice B — GMMTrainer class

- [x] T010 [US1] **[TDD-RED]** Write failing boundary test for `GMMTrainer.train()` in `tests/test_train_gmm.py`: given real Z/y artifacts from `data/processed/`, call `trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)` — assert returns `GMMTrainResult` with non-empty `run_id`, `model_path` pointing to an existing `.pkl` file, `eout_macro_f1 > 0.0` (test-set F1), `eout_fatal_recall >= 0.0`. Test MUST fail (class does not exist yet). ✅
- [x] T011 [US1] **[TDD-GREEN]** Implement `GMMTrainer` in `src/train_gmm/trainer.py`: constructor accepts `gmm_config: GMMConfig`, `model_config: ModelConfig`, `mlflow_config: MLflowConfig`, `ab_test_config: ABTestConfig`; `train(Z_train, y_train, Z_val, y_val, Z_test, y_test) → GMMTrainResult` loops over `ab_test_config.seeds`; for each class fits `GaussianMixture(n_components=gmm_config.n_components[class_label], covariance_type=..., reg_covar=..., max_iter=..., n_init=gmm_config.n_init, random_state=seed)` on that class's Z_train rows; builds `GMMClassifier` with `fatal_prior_boost`; evaluates on Z_val (→ `eval_macro_f1`, `eval_fatal_recall` — seed selection only, never exposed to evaluator) and Z_test (→ `eout_macro_f1`, `eout_fatal_recall` — logged to MLflow and read by ABEvaluator); logs `ein_macro_f1`, `eval_macro_f1`, `eval_fatal_recall`, `eout_macro_f1`, `eout_fatal_recall`, `generalisation_gap` (`= ein_macro_f1 − eout_macro_f1`, positive = overfitting — matches MLTrainer and DLTrainer convention) to MLflow run `gmm_seed_{seed}` in experiment `experiment_name_gmm`; selects best seed by val `eval_macro_f1` (constitution II); saves best `GMMClassifier` to `models/best_gmm_model.pkl` via pickle. ASCII-only print/log. ✅

### Slice C — Entry point and DVC wiring

- [x] T012 [P] [US1] Implement `src/train_gmm/run.py` (thin entry point: load config via `load_config()`, load Z/y arrays from DVC artifact paths, instantiate `GMMTrainer`, call `train()`, log result to stdout in ASCII; exit 0 on success, non-zero on failure)
- [x] T013 [P] [US1] Add `train_gmm` `@dsl.component` to `pipelines/kubeflow/pipeline.py` following the same pattern as `train_ml_op` and `train_dl_op` (calls `dvc repro train_gmm`; depends on `encode_op`)

**Checkpoint**: `uv run dvc repro train_gmm` succeeds; `models/best_gmm_model.pkl` exists and is non-empty; `uv run python -m pytest tests/test_train_gmm.py -v` is all GREEN.

---

## Phase 4: User Story 2 — 3-Way A/B/C Evaluation (Priority: P2)

**Goal**: `dvc repro evaluate` produces `docs/evaluation_report.json` with `winner` from `{ml, dl, gmm}`, 3-way pairwise p-values, and `gates_passed` enforced on the winner.

**Independent Test**: `uv run dvc repro evaluate` completes; `docs/evaluation_report.json` contains `gmm_mean_f1`, `p_value_ml_gmm`, `p_value_dl_gmm`, `winner`, `gates_passed`.

### Slice A — EvaluationResult extension

- [x] T014 [US2] **[TDD-RED]** Write failing boundary test for extended `ABEvaluator.evaluate()` in `tests/test_evaluator.py`: mock MLflow `search_runs` to return synthetic per-seed F1/recall for all three experiments; assert `EvaluationResult` has fields `gmm_mean_f1`, `p_value_ml_gmm`, `p_value_dl_gmm`, `cohens_d_ml_gmm`, `winner` ∈ `{"ml","dl","gmm"}`. Test MUST fail (fields do not exist yet).
- [x] T015 [US2] **[TDD-GREEN]** Extend `EvaluationResult` dataclass in `src/evaluate/evaluator.py`: add `gmm_mean_f1: float`, `gmm_ci_low: float`, `gmm_ci_high: float`, `gmm_mean_fatal_recall: float`; rename `p_value → p_value_ml_dl`, `cohens_d → cohens_d_ml_dl`; add `p_value_ml_gmm: float`, `p_value_dl_gmm: float`, `cohens_d_ml_gmm: float`, `cohens_d_dl_gmm: float`.

### Slice B — ABEvaluator 3-way logic

- [ ] T016 [US2] **[TDD-GREEN]** Extend `ABEvaluator.evaluate()` in `src/evaluate/evaluator.py`: query GMM experiment metrics via `_get_metrics(mlflow_config.experiment_name_gmm)`; run three pairwise Welch's t-tests with Bonferroni-corrected alpha (α/3 ≈ 0.017) — ml↔dl, ml↔gmm, dl↔gmm; apply winner selection algorithm: (1) build candidate set = all classifiers that are significantly better than at least one other (p < α/3); (2) if candidates is non-empty, winner = candidate with highest mean F1; (3) if candidates is empty (all pairwise p ≥ α/3), winner = first entry in `ab_test_config.tiebreak` list; check constitutional gates (macro F1 > threshold AND fatal recall > threshold) on winner; return extended `EvaluationResult`. ASCII-only logging.
- [ ] T017 [US2] Update `src/evaluate/run.py` to pass `config.mlflow.experiment_name_gmm` to `ABEvaluator` constructor and verify `models/best_gmm_model.pkl` exists as an operational precondition before calling `evaluate()`

**Checkpoint**: `uv run dvc repro evaluate` completes after all three training stages; `docs/evaluation_report.json` contains all three classifiers' metrics; `uv run python -m pytest tests/test_evaluator.py -v` is all GREEN.

---

## Phase 5: User Story 3 — Champion Registration with GMM Support (Priority: P3)

**Goal**: `dvc repro register` succeeds when `winner="gmm"` — bundles GMM with VAE encoder as `crash-severity@champion`.

**Independent Test**: Set `winner="gmm"` and `gates_passed=true` in `evaluation_report.json`, run `dvc repro register`; `models/registry_receipt.json` records `winner: gmm`; loaded model predicts without error.

### Slice A — CrashSeverityPyfunc extension

- [ ] T018 [US3] **[TDD-RED]** Write failing boundary test for `CrashSeverityPyfunc` with `winner="gmm"` in `tests/test_register.py`: construct pyfunc with `winner="gmm"` metadata, mock context pointing to real `vae_encoder.pth` and `best_gmm_model.pkl`, call `predict()` with real Z rows, assert output shape and all values in `{0,1,2}`. Test MUST fail (gmm branch does not exist yet).
- [ ] T019 [US3] **[TDD-GREEN]** Extend `CrashSeverityPyfunc.load_context()` in `src/register/registrar.py`: add `elif winner == "gmm"` branch that pickle-loads `context.artifacts["classifier"]` — identical to the `winner == "ml"` branch since `GMMClassifier` exposes `.predict()`. No changes needed to `predict()` since both ml and gmm call `self._classifier.predict(Z)`.

### Slice B — ModelRegistrar and run.py

- [ ] T020 [US3] Extend `ModelRegistrar.register()` in `src/register/registrar.py`: add `elif winner == "gmm"` branch that resolves `self._mlflow_config.experiment_name_gmm` when looking up the champion run via `mlflow.get_experiment_by_name()`
- [ ] T021 [US3] Update `src/register/run.py` to detect `winner` from `evaluation_report.json` and pass the correct `classifier_path`: `"ml"` → `models/best_ml_model.pkl`, `"dl"` → `models/mlp_model.pth`, `"gmm"` → `models/best_gmm_model.pkl`

**Checkpoint**: `uv run dvc repro train_ml train_dl train_gmm evaluate register` completes; `models/registry_receipt.json` records correct winner; `uv run python -m pytest tests/test_register.py -v` is all GREEN.

---

## Phase 6: Tune Stage — Winner Dispatch + fatal_prior_boost HPO

**Goal**: `dvc repro tune` trains the correct classifier per Optuna trial (ml/dl/gmm) and includes `fatal_prior_boost` in the search space when winner=gmm. Fixes the hardcoded recall penalty threshold.

**Independent Test**: Run `dvc repro tune` after a gmm-wins evaluation; Optuna trial MLflow logs contain `fatal_prior_boost` param; fitness is computed from GMM val F1, not XGBoost.

- [ ] T022 Add `fatal_prior_boost_low: float = 1.0` and `fatal_prior_boost_high: float = 5.0` to `OptunaSearchSpace` dataclass in `src/config.py`; add corresponding keys under `tune.optuna.search_space` in `params.yaml`
- [ ] T023 Extend `OptunaTuner.__init__()` in `src/tune/optuna_tuner.py` to accept `gmm_config: GMMConfig | None`; read `winner` from `docs/evaluation_report.json` (already a stage dep); store as `self._winner`
- [ ] T024 Extend `OptunaTuner._objective()` in `src/tune/optuna_tuner.py`: branch on `self._winner` and suggest only the relevant imbalance param (`fatal_threshold` for ml, `focal_loss_gamma` for dl, `fatal_prior_boost` for gmm); instantiate and train the corresponding trainer (`MLTrainer`, `DLTrainer`, or `GMMTrainer`) with a single seed; replace hardcoded `0.35` recall penalty threshold with `self._model_config.fatal_recall_threshold`
- [ ] T025 Update `src/tune/run.py` to load `config.gmm` and pass it to `OptunaTuner`

**Checkpoint**: Optuna trial logs show `fatal_prior_boost` when winner=gmm, `fatal_threshold` when winner=ml, `focal_loss_gamma` when winner=dl.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T026 [P] Update `UBIQUITOUS_LANGUAGE.md` with canonical terms: `GMMClassifier` (the pickled per-class wrapper, not sklearn's `GaussianMixture` directly), `fatal_prior_boost` (scalar log-prior multiplier for Fatal class at GMM prediction time; values > 1.0 increase Fatal recall), `3-way A/B/C test` (pairwise Welch's t-tests with Bonferroni correction across ml/dl/gmm)
- [ ] T027 [P] Update `CLAUDE.md` pipeline description: add `train_gmm` to the parallel block alongside `train_ml ‖ train_dl`; add `train_gmm` row to the stage table (`trainer.py → GMMTrainer`); update evaluate row to note 3-way Bonferroni-corrected comparison

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **BLOCKS all story phases**
- **Phase 3 (US1)**: Depends on Phase 2
- **Phase 4 (US2)**: Depends on Phase 2 + Phase 3 (evaluate queries GMM MLflow experiment)
- **Phase 5 (US3)**: Depends on Phase 2 + Phase 4 (register reads evaluation_report winner)
- **Phase 6 (Tune)**: Depends on Phase 3 (needs GMMTrainer to dispatch to)
- **Phase 7 (Polish)**: Depends on all prior phases complete

### User Story Dependencies

- **US1 (P1)**: Starts after Phase 2 — no dependency on US2 or US3
- **US2 (P2)**: Depends on US1 (evaluate queries GMM MLflow runs)
- **US3 (P3)**: Depends on US2 (register reads winner from evaluation_report)

### Within Each User Story (TDD vertical slices)

```
For each slice:
  1. Write failing test (RED) — MUST fail before proceeding
  2. Write minimal implementation (GREEN)
  3. Refactor if needed
  4. Move to next slice
```

### Parallel Opportunities

- **T012 and T013** (US1 Slice C): run.py and KFP component — different files
- **T026 and T027** (Polish): UBIQUITOUS_LANGUAGE.md and CLAUDE.md — different files

---

## Parallel Example: User Story 1 (Slice C)

After T011 (GMMTrainer) is GREEN:

```bash
# These two tasks can run simultaneously:
Task T012: Implement src/train_gmm/run.py
Task T013: Add train_gmm @dsl.component to pipelines/kubeflow/pipeline.py
```

---

## Implementation Strategy

### MVP (User Story 1 only)

1. Phase 1: Setup (T001–T002)
2. Phase 2: Foundational config + DL bug fix (T003–T007)
3. Phase 3: GMM training (T008–T013)
4. **STOP and validate**: `uv run dvc repro train_gmm` succeeds

### Incremental Delivery

1. Setup + Foundational → config ready, DL bug fixed
2. US1 (T008–T013) → GMM trains, artifacts exist, tests GREEN
3. US2 (T014–T017) → 3-way evaluation works, tests GREEN
4. US3 (T018–T021) → registration handles all three winners, tests GREEN
5. Tune (T022–T025) → winner dispatch + fatal_prior_boost HPO
6. Polish (T026–T027) → glossary and docs updated

---

## Notes

- **All `src/` code**: TDD vertical slices only — write one failing test, then implementation, then refactor before next slice
- **Fixtures**: tests MUST load real `Z_train_augmented.npy`, `Z_val.npy`, `Z_test.npy`, `y_*` from `data/processed/` — no `np.random.randn()` substitutes (constitution XVIII)
- **ASCII only**: no emoji or Unicode in any `print()` or logging call in `src/train_gmm/` (constitution XVII)
- **No ad-hoc quality assertions**: stage code checks only operational preconditions (file exists, non-empty array) — never column-level data quality (constitution XVI)
- **fatal_prior_boost**: default `1.0` = no boost; tune upward (e.g., 2.0–5.0) to increase Fatal recall at the cost of PDO precision — same trade-off as `fatal_threshold` and `focal_loss_gamma`
- **eout vs eval**: `eout_macro_f1` = test-set F1 (logged to MLflow, read by ABEvaluator); `eval_macro_f1` = val-set F1 (seed selection only); both logged per run in all three trainers
- **Same VAE**: `train_gmm` reads Z artifacts from encode — no changes to `train_vae` or `encode`
