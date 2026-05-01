# Tasks: MLOps Learning Portfolio — VAE-Based Crash Severity Pipeline

**Constitution**: v3.3.0 | **Architecture**: 10-stage DVC pipeline | **Target**: crash severity 3-class (PDO / Injury / Fatal)

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
│        │                      Katib HPO: gates fail →               │
│        │                      re-tune VAE → re-run pipeline         │
│        │                                                             │
│        └─────────────────────────────────────────────────────────── │
│          Drift detected → revisit data contract / feature set       │
└─────────────────────────────────────────────────────────────────────┘
```

**Current position**: Phase M4 complete — entering Operations (Phase O1 TDD, then O2 infrastructure, then O3 Katib).
**Next loop trigger**: evaluate gates FAIL (expected) → Katib searches `beta_max` + `latent_dim` → pipeline re-runs with best params.

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

> **Loop back here if**: drift detection (Phase 7) flags a new column distribution shift that breaks the contract, or if a new data source is added.

---

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

**Goal**: two competing classifiers trained on Z-space; 10 seeds each; mandatory metrics logged; visual diagnostics (confusion matrix + ROC) in MLflow.

- [x] T114 [P] Verify `src/train_dl/` package
- [x] T115 [P] Update `dvc.yaml` `train_dl` stage
- [x] T116 **RED** — `tests/test_train_dl.py`
- [x] T117 **GREEN** — `src/train_dl/trainer.py` — `ShallowMLP` + `DLTrainer`; `mlp_seed_{seed}` run names
- [x] T118 **GREEN** — `src/train_dl/run.py`
- [x] T119 `dvc repro train_dl` — 10 runs in `crash-severity-dl`; `models/mlp_model.pth` written
- [x] T120 MLflow UI — mandatory metrics + `per_class_matrix.json` + confusion matrix + ROC confirmed
- [x] T035 **RED** — `tests/test_train_ml.py` — XGBoost on Z vectors
- [x] T036 **GREEN** — `src/train_ml/trainer.py` — `MLTrainer`; `xgb_seed_{seed}` run names; plots via `src/plots.py`
- [x] T036b **GREEN** — `src/train_ml/run.py`
- [x] T037 `dvc repro train_ml` — 10 runs in `crash-severity-ml`; `models/best_ml_model.pkl` written

**Current results (2026-04-30) — LOOP BACK TRIGGERED**:

| Model | macro F1 | Fatal recall | F1 gate (>0.35) | Recall gate (>0.50) |
|---|---|---|---|---|
| XGBoost | 0.3636 | 0.2500 | ✅ PASS | ❌ FAIL |
| MLP | 0.3251 | 0.5500 | ❌ FAIL | ✅ PASS |

Neither model passes both gates. Root cause: `latent_dim=8` + `beta_max=0.5` creates a Z-space bottleneck (XGBoost generalisation gap = 0.278). **Katib HPO (Phase O3) will search `beta_max` + `latent_dim` to fix this.** The pipeline loops back from Operations → Model Development automatically via `params.yaml` update + `dvc repro`.

## ✅ Phase M4: Model Testing + Validation — Evaluate

**Goal**: Welch's t-test on macro F1 distributions; constitutional gates enforced; winner declared. Gates will FAIL with current params — this is expected and correct; it triggers the Katib loop.

- [x] T046 **RED** — Write `tests/test_evaluate.py`: mock MLflow runs (N=3 seeds each experiment); assert `evaluation_report.json` contains `winner`, `p_value`, `cohens_d`, `ml_mean_f1`, `dl_mean_f1`, `gates_passed`; assert `gates_passed=false` when winner mean F1 ≤ `macro_f1_threshold` or winner mean fatal recall ≤ `fatal_recall_threshold`; assert exit 1 when gates fail. Run — confirm FAIL.
- [x] T047 **GREEN** — Implement `ABEvaluator.evaluate()` in `src/evaluate/evaluator.py`: query MLflow for `eout_macro_f1` and `eout_fatal_recall` from both experiments (N=10 seeds); Welch's t-test; Cohen's d; 95% CIs; tiebreak to `ml` if p ≥ alpha; assert constitutional gates; return `EvaluationResult`. Create `src/evaluate/run.py`: write `docs/evaluation_report.json` + `docs/ab_test_comparison.json`; exit 1 if gates fail.
- [x] T048 `dvc repro evaluate` — gates fail (expected); `evaluation_report.json` written; diagnosis confirmed.

> **Loop back trigger**: gates fail here → Katib (Phase O3) searches `beta_max` + `latent_dim` → writes winners to `params.yaml` → `dvc repro` re-runs `train_vae → encode → train_ml → train_dl → evaluate` → gates should PASS after Katib.

---

---

# 🚀 OPERATIONS

> ML Model Deployment · CI/CD Pipelines · Monitoring & Triggering

---

## ✅ Phase O1: ML Model Deployment — Register (TDD only)

**Goal**: register stage implemented and tested; `dvc repro register` deferred until gates pass (post-Katib T060).

- [x] T049 **RED** — Write `tests/test_register.py`: assert with `gates_passed=true` → exit 0, `@champion` alias set, `registry_receipt.json` written; assert with `gates_passed=false` → exit 1, no registry mutation. Run — confirm FAIL.
- [x] T050 **GREEN** — Implement `ModelRegistrar.register()` in `src/register/registrar.py`; create `src/register/run.py`. **Inference path**: registered `mlflow.pyfunc` bundles `vae_encoder.pth` + champion classifier — `model.predict(X_raw)` runs `LatentEncoder → classifier` internally.

> T051 (`dvc repro register`) deferred to Phase O3 — requires `gates_passed=true` which only holds after the Katib loop rewrites `params.yaml` and `dvc repro` re-runs the pipeline.

## 🔜 Phase O2: CI/CD — Docker + Kubernetes Setup

**Goal**: container image built; local Kubernetes cluster running; shared PVC mounted. Required before Katib trial pods can execute.

- [x] T066 Create `docker/Dockerfile`: `FROM python:3.12-slim`; install `uv`; `uv sync --frozen`; copy `src/`, `dvc.yaml`, `params.yaml`, `great_expectations/gx/`; `ENV PYTHONPATH=/app`. Do NOT copy `mlruns/`, `data/`, `models/` — these come from PVC mount.
- [x] T067 Build + smoke-test: `docker build -f docker/Dockerfile -t mlops-portfolio:latest .`; `docker run --rm -v $(pwd)/data:/app/data mlops-portfolio:latest python -m src.ingest.run` → exit 0.
- [x] T068 Enable Kubernetes in Docker Desktop; verify with `kubectl cluster-info`.
- [x] T069 Create `k8s/pvc.yaml`: hostPath PV + PVC mounting project root at `/app`; apply with `kubectl apply -f k8s/pvc.yaml`.
- [x] T070 Install KFP standalone + Katib operator; wait for pods ready; `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80`. **Note**: KFP v2.0.5-pns installed; ml-pipeline API + Argo + Minio + MySQL all running; Katib v0.17.0 fully installed (all 4 pods running); port-forward verified with Katib UI on 8080; ml-pipeline-ui web frontend has ImagePullBackOff (deprecated GCR tags) but KFP API fully functional for programmatic submission.

## 🔜 Phase O3: CI/CD — Katib HPO (VAE Representation Fix)

**Goal**: Katib searches `beta_max` + `latent_dim` jointly; fitness signal penalises failed recall gate; best params written to `params.yaml`; DVC automatically re-runs full pipeline with winner values. This is the **automated loop back** from Operations → Model Development.

**Search space**:
- `beta_max`: `[0.05, 0.1, 0.2, 0.5, 1.0]` — lower KL penalty; encoder retains discriminative structure
- `latent_dim`: `[8, 16, 32]` — wider bottleneck preserves more feature interactions

**Fitness**: `val_fitness = val_macro_f1 * (1.0 if val_fatal_recall >= 0.50 else 0.5)` — evaluated on Z_val (constitution II).

- [x] T052 [P] Add `kubernetes>=28.0` to `pyproject.toml`; run `uv sync`.
- [x] T053 [P] Create `k8s/katib/vae_experiment.yaml`: Katib `Experiment` CRD; objective `val_fitness` (maximize); algorithm `bayesianoptimization`; `maxTrialCount: 15`, `parallelTrialCount: 1`; parameters: `beta_max` list `[0.05, 0.1, 0.2, 0.5, 1.0]` and `latent_dim` list `[8, 16, 32]`; trialTemplate: `python -m src.tune.trial --beta_max={{.HyperParameters.beta_max}} --latent_dim={{.HyperParameters.latent_dim}} --winner={{winner}}`; metrics collector reads `val_fitness=` from stdout.
- [x] T056 **RED + GREEN** — Create `src/tune/trial.py`: accepts `--beta_max`, `--latent_dim`, `--winner`; loads X/y splits from PVC; trains VAE with candidate params; encodes all splits; trains winner classifier (seed=0); evaluates on Z_val; computes `val_fitness`; logs to MLflow `crash-severity-tune` tagged `beta_max`, `latent_dim`, `winner`, `trial_type=katib`; prints `val_fitness=<float>` on last stdout line; exits 0.
- [x] T057 **RED** — Write `tests/test_tune.py`: mock Kubernetes client; assert `HyperparamTuner.tune()` submits Experiment; reads `currentOptimalTrial.parameterAssignments` for `beta_max` and `latent_dim`; assert `params.yaml` updated with both values. Run — confirm FAIL.
- [x] T058 **GREEN** — Implement `HyperparamTuner.tune()` in `src/tune/tuner.py`: load CRD yaml; inject winner; submit via `CustomObjectsApi`; poll until Succeeded/Failed; extract `beta_max` + `latent_dim`; return `TuneResult`. Create `src/tune/run.py`: write `tune.best_params.beta_max` and `tune.best_params.latent_dim` to `params.yaml`; exit 0.
- [ ] T059 `dvc repro tune` — Katib Experiment in Katib UI; 15 MLflow runs in `crash-severity-tune`; `params.yaml` has both best params set.
- [ ] T060 `dvc repro` (full) — DVC detects `params.yaml` change → re-runs `train_vae → encode → train_ml → train_dl → evaluate`; constitution VI gates expected to **PASS**.
- [ ] T051 `dvc repro register` — gates now pass; `@champion` alias set; `mlflow.pyfunc.load_model("models:/crash-severity@champion")` loads; `model.predict(Z_test[:5])` returns shape `(5,)` with values in `{0, 1, 2}`.

## 🔜 Phase O4: CI/CD — KFP 10-Stage Pipeline

**Goal**: full pipeline compiled to `pipeline.yaml`; runs end-to-end on Docker Desktop Kubernetes via KFP UI.

- [ ] T071 Rewrite `pipelines/kubeflow/pipeline.py` with 10 `@dsl.component` functions — one per stage — each calling `subprocess.run(["dvc", "repro", "<stage>"], cwd="/app")`; mount PVC at `/app`; wire: `validate >> ingest >> featurize`; then `featurize >> train_vae` and `featurize >> augment` (parallel); then `train_vae + augment >> encode`; then `encode >> train_ml` and `encode >> train_dl` (parallel); then `train_ml + train_dl >> evaluate >> tune >> register`; compile to `pipeline.yaml`.
- [ ] T072 `uv run python pipelines/kubeflow/pipeline.py` → `pipeline.yaml` created.
- [ ] T073 Upload to KFP UI; start run; confirm all 10 steps with correct dependency arrows.
- [ ] T074 Inspect pod logs — `train_vae` ELBO visible; `train_ml` run tagged `orchestrator=kubeflow`.

## 🔜 Phase O5: Monitoring — Latent Space Drift Detection

**Goal**: production drift detected by comparing new-batch Z vectors against training reference distribution. Advisory signal — does not halt pipeline. Triggers loop back to Design if persistent.

- [ ] T096 Extend `train_vae` to save drift reference: encode full `X_train` after training → compute per-dim μ_mean + μ_std → save `models/drift_reference.npz`; add to `dvc.yaml` outs; add `drift.*` to `params.yaml` and `DriftConfig` to `src/config.py`.
- [ ] T097 Create `src/drift/detector.py`: `DriftResult(elbo_score, mmd_score, is_drifted, n_samples)`; `DriftDetector.detect(X_new) → DriftResult`; MMD with RBF kernel (`bandwidth=1.0`).
- [ ] T098 Extend `src/encode/run.py`: call `DriftDetector.detect(X_all)`; log `drift_elbo`, `drift_mmd`, `drift_detected` to MLflow; write `docs/drift_report.json`; print `[WARN] DRIFT DETECTED` if flagged; exit 0 always.
- [ ] T099 `dvc repro encode` → `drift_report.json` written; `is_drifted=false` on training data (self-reference).

> **Loop back here if**: `is_drifted=true` on new production data → revisit data contract (Phase D2) and featurize (Phase M1).

## 🔜 Phase O6: Polish + Final Validation

**Goal**: all constitutional gates verified end-to-end; full reproducibility confirmed; documentation updated.

- [ ] T075 [P] Assert constitutional gates on `docs/evaluation_report.json`: macro F1 > 0.35, fatal recall > 0.50.
- [ ] T076 [P] Update `CLAUDE.md`: architecture table, pipeline description, DL/MLP section, featurize section.
- [ ] T077 [P] Update `.gitignore`: `Z_*.npy`, `y_train_augmented.npy`, `vae_*.pth`, `registry_receipt.json`.
- [ ] T078 Commit all tracked files: `dvc.yaml`, `params.yaml`, `src/`, GE suite, `pipeline.py`, `Dockerfile`, `k8s/`.
- [ ] T079 [P] Full reproducibility smoke test: delete `data/processed/` + `models/`; `dvc pull && dvc repro` → all 10 stages complete.
- [ ] T080 [P] Remove `apache-airflow` from `pyproject.toml` if present; `uv sync`.

---

## Iteration Summary

| Trigger | Loop back to | Mechanism |
|---|---|---|
| evaluate gates FAIL | Model Development (M2) | Katib writes `beta_max` + `latent_dim` to `params.yaml` → `dvc repro` re-runs VAE → encode → classify → evaluate |
| Drift detected (O5) | Design (D2) | Review data contract; update GE suite; re-featurize |
| New data source | Design (D1 + D2) | Update spec; extend data contract; re-run full pipeline |
| Fatal recall < 0.50 post-Katib | Model Development (M3) | Consider threshold calibration in evaluate; widen MLP hidden layer |
