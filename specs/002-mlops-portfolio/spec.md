# Feature Specification: MLOps Learning Portfolio — Crash Severity Use Case

**Feature Branch**: `002-mlops-portfolio`
**Created**: 2026-04-22
**Status**: Active
**Last amended**: 2026-04-26 — full rewrite: VAE-based architecture; grill-me session locked all decisions

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Reproducible Data & Model Pipeline (Priority: P1)

As a learner building a portfolio, I can version the dataset and all model artifacts, define
the full ML pipeline as a series of named stages, and reproduce any past experiment
exactly — including the data snapshot, preprocessing parameters, trained VAE, and
classifier models — by checking out any prior commit and running a single command.

**Why this priority**: Reproducibility is the foundation of every other MLOps concern.
Without it, no other tool in the stack can be demonstrated reliably.

**Independent Test**: From a clean clone of the repository, run the pipeline end-to-end
and confirm all produced artifacts match a previously committed run at the same git commit
and data version.

**Acceptance Scenarios**:

1. **Given** a clean repository checkout with the raw CSV tracked in artifact storage,
   **When** the learner runs the pipeline,
   **Then** all ten stages complete in order and all output artifacts are written to
   their configured paths.

2. **Given** the pipeline has run successfully once,
   **When** no input data or parameters have changed,
   **Then** re-running the pipeline skips all stages (cached) and reports no work done.

3. **Given** a feature column is added to the raw dataset,
   **When** the pipeline is re-run,
   **Then** only stages downstream of the changed input are re-executed; upstream cached
   stages are reused.

4. **Given** any past git commit with a pinned data version,
   **When** the learner checks out that commit and pulls the tracked data files,
   **Then** the pipeline produces the same artifacts as the original run.

---

### User Story 2 — Automated Data Quality Validation (Priority: P2)

As a learner, I can define a formal data contract for the crash dataset and have it
enforced automatically every time the pipeline runs. If the data fails validation, the
pipeline halts before any processing occurs and produces a human-readable report
identifying every violated expectation.

**Why this priority**: Data quality gates are the most impactful defensive layer in
production ML. This story demonstrates Great Expectations and blocks all downstream work.

**Independent Test**: Introduce a deliberately corrupted dataset version (e.g., add
out-of-range speed limits, inject unexpected null columns) and confirm the validation
stage halts the pipeline and names every violated expectation in its output report.

**Acceptance Scenarios**:

1. **Given** a clean dataset that meets all defined expectations,
   **When** the validate stage runs,
   **Then** validation passes, an HTML data quality report is generated, and the next
   stage is unblocked.

2. **Given** a dataset with null rates exceeding the defined threshold in any column,
   **When** the validate stage runs,
   **Then** the pipeline halts, the failed expectation is named in the report, and no
   downstream stage executes.

3. **Given** a dataset with an unexpected categorical value in a constrained column,
   **When** the validate stage runs,
   **Then** the pipeline halts with the specific column name and unexpected value
   identified in the report.

4. **Given** a successful validation run,
   **When** the learner opens the generated quality report,
   **Then** all expectation results are visible in a browser-readable HTML file.

---

### User Story 3 — VAE Representation Learning (Priority: P3)

As a learner, I can train a Denoising β-VAE purely on the crash feature distributions
(no labels) to learn a compressed, disentangled latent representation of every crash.
The VAE acts as a physics engine — it reconstructs clean crash records from corrupted
inputs and organises the latent space so that similar crashes cluster together.
ELBO convergence is tracked in MLflow per epoch, and the fatal class is augmented
in latent space before any supervised training begins.

**Why this priority**: The VAE is the architectural foundation — both classifiers operate
on its output. Without a well-trained encoder, the downstream A/B test is meaningless.
This story also demonstrates the novel portfolio technique: Latent-Space Augmentation
(LSA) as a principled alternative to raw-feature SMOTE for extreme class imbalance.

**Independent Test**: Run `dvc repro encode`. Confirm `train_vae` ELBO curves in MLflow
show convergence (validation ELBO stops improving). Confirm `encode` outputs `Z_train_augmented`,
`Z_val`, `Z_test` numpy arrays. Confirm `Z_train_augmented` has the fatal class at ≥5% of
total rows. Confirm `Z_val` and `Z_test` are NOT augmented.

**Acceptance Scenarios**:

1. **Given** the featurize stage has produced preprocessed feature arrays,
   **When** the `train_vae` stage runs,
   **Then** the DVAE trains on the full feature set (`X_train`, `X_val`, `X_test`
   concatenated — no target labels), logging `vae_elbo`, `vae_reconstruction_loss`,
   and `vae_kl_loss` per epoch to MLflow, and saves the trained encoder and decoder weights.

2. **Given** the VAE encoder is trained,
   **When** the `encode` stage runs,
   **Then** the frozen encoder produces `Z_train`, `Z_val`, `Z_test` latent vectors
   (each row is a 32-dimensional representation of one crash), and LSA augments
   `Z_train` so that fatal crashes reach the configured target ratio without touching
   `Z_val` or `Z_test`.

3. **Given** the learner opens the MLflow `crash-severity-vae` experiment,
   **When** they inspect the single VAE training run,
   **Then** they can see the ELBO curve converging over epochs as a line chart,
   confirming the VAE learned the data distribution rather than memorising it.

4. **Given** the ELBO does not converge within the configured epoch limit,
   **When** early stopping fires,
   **Then** the best checkpoint (lowest validation ELBO) is saved and the stage exits 0,
   logging the epoch at which best ELBO was achieved.

---

### User Story 4 — Multi-Class Severity A/B Test and Model Registration (Priority: P4)

As a learner, I can train both an XGBoost classifier and a PyTorch MLP classifier
on the same latent Z vectors across N random seeds, track every run in MLflow, and
run a rigorous statistical A/B test (Welch's t-test on macro F1 distributions) to
determine which classifier better separates the three severity classes: Property Damage
Only, Injury, and Fatal. The winner is registered in the MLflow Model Registry.

The multi-class target demonstrates the real cost structure of crash prediction: missing a
fatal crash is the most expensive error. The per-class precision/recall/F1 matrix makes
this explicit. A model that ignores the fatal class fails the constitutional gate regardless
of its overall F1 score.

**Why this priority**: The classification A/B test is the portfolio's headline result.
It shows that model selection is a statistical decision, not a single-run observation,
and that the choice between tree-based and neural classifiers is non-trivial when both
operate on the same disentangled latent space.

**Independent Test**: Run `train_ml` and `train_dl` with N=10 seeds each; confirm 10
MLflow runs appear per experiment; run `evaluate`; confirm the output report contains
a p-value, Cohen's d, 95% CIs, a declared winner, the per-class P/R/F1 matrix, and
PASS/FAIL for the constitutional performance gates.

**Acceptance Scenarios**:

1. **Given** the `encode` stage has produced `Z_train_augmented`, `Z_val`, `Z_test`,
   **When** the `train_ml` stage completes with N seeds,
   **Then** exactly N runs appear in experiment `crash-severity-ml`, each tagged with
   its seed value, with `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`, and
   `generalisation_gap` logged, plus the full per-class P/R/F1 matrix as an MLflow
   artifact, and the best-seed model artifact saved.

2. **Given** the `encode` stage has produced `Z_train_augmented`, `Z_val`, `Z_test`,
   **When** the `train_dl` stage completes with N seeds,
   **Then** exactly N runs appear in experiment `crash-severity-dl`, each tagged with
   seed value, with per-epoch `ein_loss`/`eout_loss`/`gap_f1` and final `eout_macro_f1`
   and `eout_fatal_recall` logged, plus the per-class P/R/F1 matrix as artifact.

3. **Given** both training stages have produced N runs each,
   **When** the `evaluate` stage runs,
   **Then** a Welch's t-test is performed on the two distributions of `eout_macro_f1`,
   and the report contains: mean ± std for both models, p-value, Cohen's d effect size,
   95% confidence intervals, declared winner, per-class P/R/F1 comparison, and
   constitutional gate PASS/FAIL.

4. **Given** p ≥ 0.05 (no significant difference),
   **When** the evaluate stage completes,
   **Then** the simpler model (XGBoost) is selected as the default winner, the result
   is marked "no significant difference," and the pipeline continues to register.

5. **Given** p < 0.05 and the winner's macro F1 > 0.45 AND fatal recall > 0.30,
   **Then** the winner is declared and the `register` stage is unblocked.

6. **Given** the winner is registered as `models:/crash-severity@champion`,
   **When** the learner loads it via `mlflow.pyfunc.load_model(...)`,
   **Then** the model loads and produces 3-class predictions in under 30 seconds.

---

### User Story 5 — Hyperparameter Optimisation with Katib (Priority: P5)

As a learner, after the A/B test has declared a winning classifier, I can run systematic
Bayesian hyperparameter search using Katib to find the optimal β for the β-VAE. Each
trial retrains the VAE with a candidate β, re-encodes the dataset, trains the winner
classifier, and logs the result as a separate MLflow run. The best β is written back to
`params.yaml` and the final model is retrained before registration.

**Why this priority**: β controls the disentanglement/reconstruction trade-off in the
VAE — the most architecturally significant hyperparameter. Demonstrating that it can be
searched systematically via Katib while the full ELBO/F1 history is tracked in MLflow
is a key portfolio differentiator.

**Independent Test**: Run `dvc repro tune`; confirm N trial runs appear in
`crash-severity-tune` each tagged with trial number and β value; confirm `params.yaml`
updated under `tune.best_params`; confirm best trial metric exceeds pre-tune winner metric.

**Acceptance Scenarios**:

1. **Given** the evaluate stage has declared a winner,
   **When** the tune stage runs,
   **Then** a Katib Experiment CRD is submitted to Kubernetes with β search space
   `[0.5, 1.0, 2.0, 4.0, 8.0]` and N trials run as pods using Bayesian optimisation.

2. **Given** a tune trial completes,
   **When** the learner opens the MLflow UI,
   **Then** each trial appears as a separate run in `crash-severity-tune` tagged with
   `trial=<n>`, `beta=<value>`, `winner=<ml|dl>`, and `eout_macro_f1` logged.

3. **Given** all N trials complete,
   **When** Katib selects the best trial,
   **Then** the best β is written to `params.yaml` under `tune.best_params.beta` and
   DVC detects the param change and invalidates downstream stages.

---

### User Story 6 — Container-Native Orchestration via Kubeflow (Priority: P6)

As a learner, I can deploy the full 10-stage pipeline to Kubernetes via Kubeflow
Pipelines, where each stage runs as an isolated container pod. I can submit a pipeline
run from the Kubeflow UI, monitor pod-level execution, and inspect per-stage logs —
all on a local Docker Desktop Kubernetes cluster.

**Why this priority**: Kubeflow represents the production-grade Kubernetes-native approach.
This story demonstrates that the same pipeline logic runs portably in containers —
the key MLOps portfolio differentiator.

**Independent Test**: Submit the pipeline from the Kubeflow UI; confirm all 10 stages
appear as sequential pods in Kubernetes; inspect the `train_vae` pod logs to confirm
ELBO logging is visible; verify the trained encoder artifact is written to the
configured output path.

**Acceptance Scenarios**:

1. **Given** Kubeflow Pipelines is running on Docker Desktop Kubernetes,
   **When** the learner uploads and runs the compiled pipeline,
   **Then** all 10 stages appear as sequential steps in the Kubeflow UI with correct
   dependency arrows.

2. **Given** the validate stage fails due to a data quality issue,
   **When** the learner views the Kubeflow UI,
   **Then** the validate step is marked failed and downstream steps are not scheduled.

3. **Given** a successfully completed Kubeflow pipeline run,
   **When** the learner checks the experiment tracking UI,
   **Then** the training runs logged from inside the Kubeflow pods appear alongside
   runs from local DVC executions under the same experiment names.

---

### Edge Cases

- What happens when the VAE training diverges (ELBO increases instead of decreasing)?
  Early stopping should save the last best checkpoint and exit 0 — not 1. A diverging
  ELBO is a hyperparameter issue, not a data issue; the pipeline should continue with
  the best available encoder.
- What happens when there are fewer than 10 fatal crashes in the train split after LSA?
  The `encode` stage should halt with exit 1 and a clear message — LSA requires at least
  10 real fatal samples to fit a meaningful Gaussian in Z-space.
- What happens when the Kubernetes cluster is not running when a Kubeflow pipeline is
  submitted? The client should raise a connection error before any pods are scheduled.
- What happens when a KFP pipeline run and a local `dvc repro` are triggered
  simultaneously? Concurrent pipeline execution is not supported — only one run
  should be active at a time.
- What happens when `tune.best_params.beta` is written to `params.yaml` but the VAE
  retraining produces worse ELBO than the original? The tune stage should log both the
  original and retrained ELBO, and the `register` stage gates on the final classifier
  quality, not VAE quality directly.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline MUST be fully reproducible from any prior git commit using
  **DVC** — `dvc repro` from a clean checkout with the pinned data version MUST
  reproduce all artifacts without any manual steps.

- **FR-002**: All ten pipeline stages MUST be defined as named stages in **`dvc.yaml`**
  with explicit `cmd`, `deps`, `outs`, and `params` entries:
  `validate → ingest → featurize → train_vae → encode → train_ml → train_dl → evaluate → tune → register`.

- **FR-003**: The validate stage MUST enforce a **Great Expectations** expectation suite
  against the raw dataset and MUST halt all downstream stages if any expectation fails.

- **FR-004**: The `train_vae` stage MUST train a **Denoising β-VAE** on the full feature
  matrix (`X_train`, `X_val`, `X_test` concatenated — no target labels), applying
  `nn.Dropout(p=0.15)` to inputs for neural inpainting. The VAE MUST log `vae_elbo`,
  `vae_reconstruction_loss`, and `vae_kl_loss` per epoch to MLflow experiment
  `crash-severity-vae`. The encoder and decoder weights MUST be saved as DVC-tracked
  artifacts. `latent_dim = 32` (fixed). Encoder architecture configurable via
  `params.yaml` as `vae.encoder_dims` (default `[256, 128, 64]`).

- **FR-005**: The `encode` stage MUST use the frozen VAE encoder to produce latent vectors
  `Z_train`, `Z_val`, `Z_test` from the corresponding preprocessed feature arrays.
  LSA MUST augment `Z_train` by sampling Gaussian noise around fatal-class centroids
  in Z-space until fatal crashes reach `encode.lsa_target_ratio` (default 0.05) of
  total training rows. `Z_val` and `Z_test` MUST NOT be augmented.

- **FR-006**: The `train_ml` stage MUST train an **XGBoost** multi-class classifier on
  `Z_train_augmented` N times (one run per seed in `ab_test.seeds`), logging each as
  a separate MLflow run in `crash-severity-ml` tagged `seed=<value>`, `model_type=xgboost`.
  Each run MUST log `eout_macro_f1`, `eout_fatal_recall`, `ein_macro_f1`,
  `generalisation_gap`, and the full per-class P/R/F1 matrix as an MLflow artifact.
  `mlflow.sklearn.autolog()` MUST be disabled. Best-seed model saved to
  `models/best_ml_model.pkl`.

- **FR-007**: The `train_dl` stage MUST train a **PyTorch MLP** classifier on
  `Z_train_augmented` N times (one run per seed in `ab_test.seeds`), logging each as
  a separate MLflow run in `crash-severity-dl` tagged `seed=<value>`, `model_type=mlp`.
  Each run MUST log per-epoch `ein_loss`/`eout_loss`/`gap_f1`, final `eout_macro_f1`,
  `eout_fatal_recall`, and the per-class P/R/F1 matrix as an MLflow artifact.
  MLP architecture: `Linear(32, 64) → ReLU → Dropout → Linear(64, 3)`, configurable
  in `params.yaml`. Best-seed checkpoint saved to `models/mlp_model.pth`.

- **FR-008**: The `evaluate` stage MUST collect `eout_macro_f1` scores from all N MLflow
  runs in each experiment and perform a **Welch's t-test**. The report MUST include:
  mean ± std for both models, p-value, Cohen's d, 95% CIs, declared winner,
  per-class P/R/F1 comparison table, and constitutional gate PASS/FAIL.
  If p ≥ 0.05, XGBoost is the default winner. All metrics MUST be logged as an MLflow
  run artifact.

- **FR-009**: The constitutional performance gates are: macro F1 > 0.45 AND fatal recall
  > 0.30 on `Z_test`. If either gate fails, the pipeline MUST halt at `evaluate` and
  `register` MUST NOT execute.

- **FR-010**: The `tune` stage MUST submit a **Katib** Experiment CRD to Kubernetes to
  search β over `[0.5, 1.0, 2.0, 4.0, 8.0]` using Bayesian optimisation. Each trial
  retrains the VAE with the candidate β, re-encodes the dataset, trains the winner
  classifier, and logs a separate MLflow run in `crash-severity-tune` tagged
  `trial=<n>`, `beta=<value>`. Best β MUST be written to `params.yaml` under
  `tune.best_params.beta`.

- **FR-011**: Each pipeline stage MUST accept input and output paths via environment
  variables or CLI arguments (not hardcoded), so the same `src/` business logic runs
  identically under `dvc repro` and KFP components.

- **FR-012**: Raw data and all trained model artifacts (VAE encoder/decoder, XGBoost pkl,
  MLP pth) MUST be tracked with **DVC**, with `.dvc` pointer files committed to git.

- **FR-013**: Each **Great Expectations** validation run MUST generate a Data Docs HTML
  report saved as a pipeline artifact, viewable in a browser without additional tooling.

- **FR-014**: Any individual pipeline stage MUST be runnable as a standalone Docker
  container given its declared inputs are available, independent of the orchestrator.

- **FR-015**: The full pipeline MUST be expressible as a **Kubeflow Pipelines (KFP)**
  workflow where each of the 10 stages is a containerised KFP component running as an
  isolated pod on Docker Desktop Kubernetes.

### Key Entities

- **Pipeline Stage**: A named, self-contained unit of work with declared inputs,
  outputs, and an executable command. Runs identically under local DVC and
  Kubeflow execution.

- **VAE Model**: The trained Denoising β-VAE. Consists of encoder weights and decoder
  weights saved as separate artifacts. The encoder alone is used by the `encode` stage;
  both are needed for Katib trials that retrain from scratch.

- **Latent Vector (Z)**: A 32-dimensional compressed representation of one crash record,
  produced by passing preprocessed features through the frozen VAE encoder. The input
  to both classifiers.

- **LSA Augmentation**: Synthetic fatal-class Z vectors generated by adding Gaussian
  noise around real fatal centroids in Z-space. Applied to `Z_train` only.

- **Expectation Suite**: A versioned, committed set of data quality rules applied to
  the raw dataset before any processing.

- **ML Experiment Run**: A single XGBoost training run on `Z_train_augmented`.
  Logged in `crash-severity-ml`. Tagged with seed, model type, β value used.

- **DL Experiment Run**: A single MLP training run on `Z_train_augmented`.
  Logged in `crash-severity-dl`. Per-epoch losses tracked.

- **A/B Test Result**: The output of the `evaluate` stage — Welch's t-test result,
  per-class metrics, declared winner, gate PASS/FAIL. Stored as
  `docs/ab_test_comparison.json`.

- **Registered Model**: The champion model in the MLflow Model Registry.
  Alias: `@champion`. Loadable via `mlflow.pyfunc.load_model("models:/crash-severity@champion")`.

- **KFP Pipeline**: Ten containerised components with sequential dependency structure.
  Each calls `dvc repro <stage>` with the project root mounted via a shared hostPath PVC.

- **Data Contract**: Column-level specification in `docs/data_contract.md` and
  `params.yaml validation.columns`. Enforced exclusively by the GE expectation suite.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `dvc repro` from a clean checkout runs all 10 stages end-to-end with no
  manual steps and no files outside DVC/git tracking.

- **SC-002**: The Great Expectations validate stage automatically catches at least five
  distinct issue categories (schema, null rates, numeric ranges, categorical values,
  row count) and surfaces them in the generated Data Docs HTML report.

- **SC-003**: The VAE training run in MLflow shows a converging ELBO curve (monotonically
  decreasing over at least 80% of training epochs) visible as a line chart in the UI.

- **SC-004**: Both MLflow experiments (`crash-severity-ml` and `crash-severity-dl`) are
  queryable and comparable in the MLflow UI; the A/B test result is visible as a logged
  artifact without accessing `mlruns/` directly.

- **SC-005**: The champion model's per-class P/R/F1 matrix shows fatal recall > 0.30
  and macro F1 > 0.45, confirmed in the evaluation report before registration.

- **SC-006**: The identical `src/` stage logic runs successfully under `dvc repro`
  and the KFP pipeline without duplicating any business logic.

- **SC-007**: Any pipeline stage runs as a Docker container via `docker run` with only
  environment variables as configuration — no host-installed Python packages required.

- **SC-008**: A model promoted to the MLflow Model Registry is loadable via
  `mlflow.pyfunc.load_model("models:/crash-severity@champion")` and produces
  3-class predictions in under 30 seconds from a fresh Python session.

- **SC-009**: The Kubeflow Pipelines UI shows step-level status and logs for every
  pipeline run; all 10 stages appear with correct dependency arrows.

- **SC-010**: Any past pipeline run is fully reproducible by checking out the
  corresponding git commit and running `dvc pull && dvc repro` — confirmed by matching
  the produced model artifact checksums against the original run's logged artifacts.

---

## Assumptions

- Docker Desktop is installed with Kubernetes enabled (single-node local cluster).
- Kubeflow Pipelines standalone is deployed to Docker Desktop Kubernetes.
- Artifact remote storage is a local directory for simplicity; cloud remote is out of scope.
- The crash dataset (`data/raw/CGR_Crash_Data.csv`, 74,309 rows) is the sole input.
- CRASHSEVER encodes three severity levels natively: `Property Damage Only`, `Injury`,
  `Fatal` — no additional columns needed for the multi-class target.
- Fatal crash rate in the dataset is < 5% — LSA is activated when this condition holds.
- The experiment tracking server runs locally; no hosted tracking server is required.
- All pipeline stages share a single base Docker image to reduce build complexity.
- Windows 11 with Docker Desktop is the development environment.
- All stage scripts use `src/config.py` typed dataclass accessors to read `params.yaml`.
- All stages call `mlflow.set_tracking_uri(config.mlflow.tracking_uri)` before any MLflow operation.
- GE is the exclusive data quality assertion layer (Principle XVI): no ad-hoc quality
  assertions in stage code or test code.
- Kubeflow Pipelines components mount the project root via a single hostPath PVC at `/app`.
- Concurrent pipeline execution is not supported. Only one pipeline run should be active at a time.
- The VAE is trained unsupervised (no Y labels) on the full feature matrix — this is
  intentional and constitutionally permitted (Principle II amended). The supervised 3-way
  split governs only the classifiers and all evaluation.
- Windows 11 Python scripts use LF line endings enforced via `.gitattributes`.
