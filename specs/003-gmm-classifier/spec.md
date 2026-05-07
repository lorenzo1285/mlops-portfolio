# Feature Specification: GMM Classifier — Third Parallel Training Branch

**Feature Branch**: `003-gmm-classifier`  
**Created**: 2026-05-06  
**Status**: Draft  
**Input**: User description: "Add GMM (Gaussian Mixture Model) as a third parallel classifier branch alongside train_ml and train_dl. The GMM trains on Z_train_augmented (VAE latent space, 8-dim), uses multi-seed training, logs mandatory MLflow metrics, and competes in a 3-way A/B/C evaluation (extended Welch's/ANOVA) against XGBoost and MLP. The winner is promoted to the MLflow registry as crash-severity@champion. Pipeline: encode → train_gmm (parallel with train_ml/train_dl) → evaluate (3-way) → register."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Train GMM on Latent Space (Priority: P1)

An ML engineer runs the pipeline end-to-end and a GMM classifier trains on the VAE-encoded latent vectors alongside XGBoost and MLP, without touching the existing train_ml or train_dl stages.

**Why this priority**: The GMM training stage is the foundational deliverable. Without it, no downstream 3-way comparison is possible.

**Independent Test**: Run `dvc repro train_gmm` after `encode` completes; confirm `models/best_gmm_model.pkl` is produced and an MLflow run appears in the `crash-severity-gmm` experiment with all mandatory metrics logged.

**Acceptance Scenarios**:

1. **Given** Z_train_augmented, Z_val, Z_test, y_train_augmented, y_val, y_test are available from the encode stage, **When** `dvc repro train_gmm` is executed, **Then** `models/best_gmm_model.pkl` is written and tracked by DVC.
2. **Given** the training run completes, **When** the MLflow experiment `crash-severity-gmm` is queried, **Then** each seed run contains `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`, and `generalisation_gap`.
3. **Given** 10 seeds are configured, **When** training finishes, **Then** the best-seed model is selected by validation-set `eout_macro_f1` and persisted.

---

### User Story 2 — 3-Way A/B/C Evaluation (Priority: P2)

An ML engineer runs the evaluate stage and receives a statistically grounded winner decision among all three classifiers (XGBoost, MLP, GMM), with constitutional gates enforced on the winner.

**Why this priority**: Extending evaluation to three-way comparison is the core value of this feature — it determines whether GMM can displace the existing champion.

**Independent Test**: Run `dvc repro evaluate` with all three model artifacts present; confirm `docs/evaluation_report.json` names a winner from {ml, dl, gmm} and records gate outcomes.

**Acceptance Scenarios**:

1. **Given** all three trained models and their per-seed MLflow metrics are available, **When** `dvc repro evaluate` runs, **Then** `docs/evaluation_report.json` is produced with `winner` set to one of `"ml"`, `"dl"`, or `"gmm"`.
2. **Given** the winning model's mean `eout_macro_f1` is below `model.macro_f1_threshold` or mean `eout_fatal_recall` is below `model.fatal_recall_threshold`, **When** evaluate completes, **Then** `gates_passed` is `false` and the register stage refuses to proceed.
3. **Given** two or more classifiers are statistically indistinguishable (p ≥ alpha), **When** evaluate completes, **Then** the tiebreak rule in `params.yaml` determines the winner deterministically.

---

### User Story 3 — Champion Registration with GMM Support (Priority: P3)

An ML engineer runs the register stage and the winning classifier — which may now be a GMM — is bundled with the VAE encoder into the `crash-severity@champion` MLflow model.

**Why this priority**: Registration closes the loop; without it the pipeline produces an artifact but no deployable model.

**Independent Test**: Set winner to `"gmm"` in evaluation report, run `dvc repro register`; confirm `models/registry_receipt.json` records `model_type: gmm` and the MLflow registry entry loads and predicts correctly.

**Acceptance Scenarios**:

1. **Given** evaluate has set winner to `"gmm"` with `gates_passed: true`, **When** `dvc repro register` runs, **Then** `models/registry_receipt.json` is produced and the MLflow model alias `crash-severity@champion` is updated.
2. **Given** the registered GMM champion model is loaded, **When** a raw feature row is passed through the prediction interface, **Then** the model returns one of {Fatal, Injury, PDO} without error.
3. **Given** evaluate has `gates_passed: false`, **When** `dvc repro register` runs, **Then** registration is refused and the stage exits non-zero.

---

### Edge Cases

- What happens when GMM fails to converge for a given seed? That seed is skipped and training continues with remaining seeds; if all seeds fail, the stage exits with a descriptive error.
- What happens when all three classifiers are statistically indistinguishable? The tiebreak order defined in `params.yaml` (e.g., `["ml", "dl", "gmm"]`) resolves the winner deterministically.
- What happens when GMM wins but its `fatal_recall` is below the constitutional threshold? The gate fails; registration is refused regardless of F1.
- What happens when GMM produces degenerate posteriors (all probability mass on one class)? The run is logged with its actual metrics and naturally loses the competition on macro F1.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline MUST include a `train_gmm` DVC stage that runs in parallel with `train_ml` and `train_dl`, depending only on `encode` outputs.
- **FR-002**: `GMMTrainer` MUST train one GMM per seed (10 seeds by default) on Z_train_augmented, selecting the best seed by validation-set `eout_macro_f1`.
- **FR-003**: `GMMTrainer` MUST log `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`, and `generalisation_gap` to MLflow for every seed run under the experiment `crash-severity-gmm`.
- **FR-004**: The system MUST persist the best-seed GMM model as a DVC-tracked artifact at `models/best_gmm_model.pkl`.
- **FR-005**: The evaluate stage MUST extend its comparison from 2-way (XGBoost vs MLP) to 3-way (XGBoost vs MLP vs GMM) using pairwise Welch's t-tests on per-seed `eout_macro_f1` distributions.
- **FR-006**: The evaluate stage MUST enforce constitutional gates (macro F1 threshold, fatal recall threshold) on the 3-way winner, identical to the gates applied in the current 2-way evaluation.
- **FR-007**: `params.yaml` MUST include a `tiebreak` list that covers three-way ties and resolves winner selection deterministically (e.g., `["ml", "dl", "gmm"]`).
- **FR-008**: The register stage MUST support `winner="gmm"` — bundling the GMM model with the VAE encoder in the `CrashSeverityPyfunc` wrapper and registering it as `crash-severity@champion`.
- **FR-009**: GMM hyperparameters (`n_components` as a per-class dict, `covariance_type`, `reg_covar`, `max_iter`, `n_init`) MUST be configurable via `params.yaml` and read through a typed config dataclass — no hardcoded values. Default `n_components`: `{0: 1, 1: 1, 2: 2}` (Fatal gets two Gaussians to handle potential bimodal distribution in Z-space).
- **FR-010**: All existing `train_ml` and `train_dl` stage code MUST remain unmodified.
- **FR-011**: The GMM MUST apply a configurable `fatal_prior_boost` scalar (default `1.0`) that upweights the Fatal-class log-prior at prediction time, providing the same Fatal recall sensitivity as `fatal_threshold` in XGBoost and focal loss in MLP. Value MUST be configurable in `params.yaml` under `gmm.fatal_prior_boost`.

### Key Entities

- **GMMTrainer**: trains one GMM per class on Z_train_augmented rows belonging to that class; predicts by maximum posterior log-likelihood; selects best seed by validation macro F1; saves best model artifact.
- **GMMConfig**: typed dataclass in `src/config.py`; fields: `n_components`, `covariance_type`, `reg_covar`, `max_iter`, MLflow experiment name.
- **GMMTrainResult**: structured result object returned by `GMMTrainer.train()` — mirrors `MLTrainResult` and `DLTrainResult` with `run_id`, `model_path`, `best_seed`, `eout_macro_f1`, `eout_fatal_recall`.
- **MultiEvaluator** (extended ABEvaluator): queries per-seed metrics from all three MLflow experiments; runs pairwise Welch's t-tests; returns winner, gate status, and 3-way comparison report.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The `train_gmm` stage completes in wall-clock time within the same order of magnitude as `train_ml` for 10 seeds on the same machine.
- **SC-002**: All three classifiers' per-seed `eout_macro_f1` distributions are captured in `docs/evaluation_report.json` with mean, std, and p-values for each pairwise comparison.
- **SC-003**: The winner selected by evaluate satisfies both `mean_eout_macro_f1 > macro_f1_threshold` AND `mean_eout_fatal_recall > fatal_recall_threshold` before registration proceeds; any violation blocks registration.
- **SC-004**: The registered model (regardless of whether winner is ml/dl/gmm) accepts raw feature input and returns a 3-class prediction without code changes to the inference path.
- **SC-005**: The end-to-end pipeline (`dvc repro`) completes without error when GMM is the winning classifier.

## Assumptions

- GMM operates in discriminative mode: one GMM fitted per class on Z_train_augmented rows belonging to that class; prediction is argmax of per-class log-likelihood (posterior).
- Best-seed selection uses validation-set `eout_macro_f1` (consistent with the constitutional requirement that the test set is strictly reserved for final evaluation).
- Probability calibration is not required for the A/B/C comparison because the evaluator compares macro F1 distributions — all classifiers are evaluated on the same metric.
- Fatal-class boosting is implemented via `gmm.fatal_prior_boost` — a scalar multiplier on the Fatal-class log-prior at prediction time (analogous to `fatal_threshold` in XGBoost and `focal_loss_gamma` in MLP). Default `1.0` (no boost); values > 1.0 increase Fatal recall at the cost of PDO precision.
- `covariance_type` defaults to `"full"` (each class gets its own full covariance matrix), appropriate for an 8-dim latent space with moderate sample counts.
- `n_components=1` per class is the default (each class modelled as a single Gaussian); this is configurable for future multi-modal class distributions.
- The evaluate stage tiebreak for 3-way ties follows the order defined in `params.yaml` — first listed classifier wins.
- `X_val` and `X_test` are never augmented (constitution III); only Z_train_augmented is used to fit the GMM.
- The `train_ml` and `train_dl` experiments remain unchanged; the evaluate stage queries all three experiments independently.
