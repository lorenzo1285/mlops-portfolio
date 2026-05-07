# Implementation Plan: GMM Classifier — Third Parallel Training Branch

**Branch**: `003-gmm-classifier` | **Date**: 2026-05-06 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `specs/003-gmm-classifier/spec.md`

## Summary

Add a Gaussian Mixture Model (GMM) as a third parallel classifier alongside XGBoost (`train_ml`) and ShallowMLP (`train_dl`). The GMM trains on Z_train_augmented (8-dim VAE latent space), uses per-class Gaussian fitting with multi-seed evaluation, logs all mandatory MLflow metrics, and competes in a 3-way statistical comparison in the evaluate stage. The evaluate stage is extended from a 2-way Welch's t-test to pairwise Welch's t-tests with Bonferroni correction across all three classifiers. The register stage is extended to dispatch on `winner="gmm"`. All existing `train_ml` and `train_dl` code remains unchanged.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: scikit-learn `GaussianMixture`, MLflow, DVC, numpy, scipy (already present)  
**Storage**: DVC-tracked `.npy` artifacts (Z splits) + `models/best_gmm_model.pkl` output  
**Testing**: `uv run python -m pytest` — real pipeline artifacts as fixtures (Z_train_augmented, Z_val, Z_test, y_* arrays)  
**Target Platform**: Windows (local dev) + Docker Desktop Kubernetes (KFP pods)  
**Project Type**: DVC pipeline stage + KFP component  
**Performance Goals**: Train 10 GMM seeds in <60s wall-clock (GMM convergence is fast on 8-dim Z-space)  
**Constraints**: ASCII-only terminal output (cp1252); no ad-hoc data quality assertions; test fixtures from real pipeline artifacts only  
**Scale/Scope**: 74,309 rows × 8 latent dims; 3 classes; 10 seeds; single machine + single K8s pod

## Constitution Check

*GATE: Verified before Phase 0 research.*

| Principle | Gate | Status | Notes |
|-----------|------|--------|-------|
| I — Feature Leakage | GMM inputs are Z_train_augmented (from pre-crash features only) | PASS | No post-crash columns reach GMM |
| II — No-Contamination | GMM fits on Z_train only; val/test used for evaluation only | PASS | Constitution II exception (VAE unsupervised) is upstream |
| III — Imbalance Management | Per-class GMM fitting is structurally imbalance-aware; CTGAN augmentation already applied to Z_train_augmented | PASS | No additional SMOTE or oversampling; class weights not applicable to GMM (no sample_weight param) |
| IV — Shallow Architecture | Not applicable to GMM (generative, not an MLP) | PASS | Constitution IV scoped to discriminative MLP classifiers |
| V — MLflow Tracking | FR-003 mandates eout_macro_f1, eout_fatal_recall, ein_macro_f1, generalisation_gap per seed | PASS | experiment_name_gmm added to MLflowConfig |
| VI — Macro F1 Primary | Winner selected by mean eout_macro_f1; gates enforced on winner | PASS | FR-005, FR-006 |
| VII — DVC Versioning | models/best_gmm_model.pkl is DVC-tracked output of train_gmm stage | PASS | Added to dvc.yaml outs |
| VIII — GE Validation | Validate runs on raw data before pipeline; GMM stage is downstream | PASS | No change needed |
| IX — KFP Pipeline | train_gmm must be added as new @dsl.component to pipelines/kubeflow/pipeline.py | PASS | Task required |
| X — Container-Native | train_gmm stage follows same pattern as train_ml (env vars, DVC, Docker) | PASS | |
| XI — Script-First | Business logic in src/train_gmm/trainer.py; entry in src/train_gmm/run.py | PASS | |
| XII — Ubiquitous Language | "GMMClassifier", "MultiEvaluator", "3-way A/B/C test" are new terms | NOTE | UBIQUITOUS_LANGUAGE.md must be updated before tasks complete |
| XIII — Grill-Me | Spec written, validated (0 NEEDS CLARIFICATION markers, all checklist items pass) | PASS | |
| XIV — Deep Module | GMMTrainer: constructor + train(); ABEvaluator extended (same interface depth) | PASS | |
| XV — TDD | All new src/ code written test-first; red→green→refactor | PASS | |
| XVI — GE Exclusive | No ad-hoc data quality assertions in train_gmm or extended evaluate code | PASS | |
| XVII — ASCII Terminal | All print/log in src/ use ASCII only | PASS | |
| XVIII — Real Data Fixtures | Tests use Z_train_augmented.npy, Z_val.npy, Z_test.npy, y_* from real pipeline artifacts | PASS | |

**Constitution Check result: PASS** — no gate violations. One NOTE (XII) to be resolved during tasks.

## Project Structure

### Documentation (this feature)

```text
specs/003-gmm-classifier/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── stage-interface.md   # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── train_gmm/           # NEW
│   ├── __init__.py
│   ├── trainer.py       # GMMTrainer class (constructor + train())
│   └── run.py           # thin entry point: config, I/O, MLflow logging
├── evaluate/
│   └── evaluator.py     # MODIFIED: ABEvaluator → 3-way pairwise Welch's t-tests
├── register/
│   └── registrar.py     # MODIFIED: CrashSeverityPyfunc load_context/predict for gmm
├── config.py            # MODIFIED: add GMMConfig; MLflowConfig += experiment_name_gmm;
│                        #           ABTestConfig.tiebreak: str → list[str]
└── metrics.py           # UNCHANGED

tests/
├── test_train_gmm.py    # NEW: GMMTrainer boundary tests (real Z artifacts)
├── test_evaluator.py    # MODIFIED: extend to 3-way comparison scenarios
└── test_register.py     # MODIFIED: extend for winner="gmm" path

pipelines/kubeflow/
└── pipeline.py          # MODIFIED: add train_gmm @dsl.component

params.yaml              # MODIFIED: add gmm.* section; mlflow.experiment_name_gmm;
                         #           ab_test.tiebreak → list

dvc.yaml                 # MODIFIED: add train_gmm stage; update evaluate deps
```

**Structure Decision**: Single-project layout (Option 1). New `src/train_gmm/` directory mirrors `src/train_ml/` and `src/train_dl/`. Surgical modifications to evaluate, register, config, and pipeline files only — no new top-level directories.
