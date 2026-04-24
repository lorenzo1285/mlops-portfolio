# Feature Specification: MLOps Learning Portfolio — Crash Severity Use Case

**Feature Branch**: `002-mlops-portfolio`
**Created**: 2026-04-22
**Status**: Draft
**Input**: User description: "MLOps learning portfolio built on the crash severity use case, covering DVC, Great Expectations, MLflow, Apache Airflow, and Kubeflow Pipelines. Eight pipeline stages: ingest → validate → featurize → train_ml → train_dl → evaluate → tune → register. ML model via PyCaret, DL model via PyTorch MLP. Evaluate runs A/B test and selects the winner. Tune runs Optuna hyperparameter optimisation on the winning model family, with every trial tracked in MLflow. All stages script-based, containerised, orchestrated by both Airflow and Kubeflow."
**Last amended**: 2026-04-23 — added tune stage (Optuna HPO), sequential execution, grill-me decisions applied

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Reproducible Data & Model Pipeline (Priority: P1)

As a learner building a portfolio, I can version the dataset and model artifacts, define
the full ML pipeline as a series of named stages, and reproduce any past experiment
exactly — including the data snapshot, preprocessing parameters, and trained model — by
checking out any prior commit and running a single command.

**Why this priority**: Reproducibility is the foundation of every other MLOps concern.
Without it, no other tool in the stack can be demonstrated reliably. This story produces
the versioned pipeline definition and the tracked dataset and model artifacts.

**Independent Test**: From a clean clone of the repository, run the pipeline end-to-end
and confirm the produced model artifact matches a previously committed run at the same
git commit and data version.

**Acceptance Scenarios**:

1. **Given** a clean repository checkout with the raw CSV tracked in artifact storage,
   **When** the learner runs the pipeline,
   **Then** all six stages complete in order and all output artifacts are written to
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
   **Then** the pipeline produces the same model artifact as the original run.

---

### User Story 2 — Automated Data Quality Validation (Priority: P2)

As a learner, I can define a formal data contract for the crash dataset (expected schema,
null rate limits, value ranges, allowed categorical values) and have that contract
enforced automatically every time the pipeline runs. If the data fails validation, the
pipeline halts before any training occurs and produces a human-readable report
identifying every violated expectation.

**Why this priority**: Data quality gates are the most impactful defensive layer in
production ML. This story demonstrates Great Expectations and establishes the validate
stage that blocks all downstream work.

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
   **Then** all expectation results (pass/fail count, column-level detail) are visible
   in a browser-readable HTML file without installing additional tools.

---

### User Story 3 — ML vs DL Statistical A/B Test and Model Registry (Priority: P3)

As a learner, I can train both a PyCaret ML model and a PyTorch MLP model repeatedly
across N different random seeds, track every run in MLflow, and then run a rigorous
statistical A/B test (Welch's t-test on the distribution of test-set macro F1 scores)
to determine which model family is significantly better. The winner is registered in the
MLflow Model Registry. This demonstrates that model selection is a statistical decision,
not a single-run observation.

**Why this priority**: A single-run comparison is unreliable — it reflects one random
initialisation, not the model family's true performance distribution. A proper A/B test
with effect size and confidence intervals is what separates a portfolio from a tutorial.

**Independent Test**: Run `train_ml` and `train_dl` with N=10 seeds each; confirm 10
MLflow runs appear per experiment; run `evaluate`; confirm the output report contains
a p-value, Cohen's d, 95% CIs for both models, a declared winner, and a PASS/FAIL
for the constitutional performance gates.

**Acceptance Scenarios**:

1. **Given** the `train_ml` stage completes with N seeds,
   **When** the learner opens the MLflow UI,
   **Then** exactly N runs appear in experiment `crash-severity-ml`, each tagged with
   its seed value, with `ein_macro_f1`, `eout_macro_f1`, and `generalisation_gap`
   logged — and the best-seed model artifact saved.

2. **Given** the `train_dl` stage completes with N seeds,
   **When** the learner opens the MLflow UI,
   **Then** exactly N runs appear in experiment `crash-severity-dl`, each tagged with
   its seed value, with per-epoch `ein_loss`/`eout_loss` and final `eout_macro_f1`
   logged.

3. **Given** both training stages have produced N runs each,
   **When** the `evaluate` stage runs,
   **Then** a Welch's t-test is performed on the two distributions of `eout_macro_f1`
   scores, and the report contains: mean ± std for both models, p-value, Cohen's d
   effect size, 95% confidence intervals, and a declared winner.

4. **Given** the statistical test result,
   **When** p < 0.05 and the winner's mean macro F1 > 0.55 and minority recall > 0.40,
   **Then** the winner is declared and the `register` stage is unblocked.

5. **Given** p ≥ 0.05 (no significant difference),
   **When** the evaluate stage completes,
   **Then** the simpler model (ML/PyCaret) is selected as the default winner, the
   result is marked "no significant difference" in the report, and the pipeline
   continues to register without failing.

6. **Given** the winner is registered as `models:/crash-severity@champion`,
   **When** the learner runs `mlflow.pyfunc.load_model("models:/crash-severity@champion")`,
   **Then** the best-seed artifact loads and produces predictions in under 30 seconds.

---

### User Story 3b — Hyperparameter Optimisation with Optuna (Priority: P3b)

As a learner, after the A/B test has declared a winning model family, I can run a
systematic Bayesian hyperparameter search using Optuna on that winner. Every trial is
logged as a separate MLflow run so I can compare the full search history in the UI.
The best hyperparameters are written back to `params.yaml` and DVC-tracked, and the
final model is retrained with optimal parameters before registration.

**Why this priority**: A single training run with default hyperparameters is not a
production-quality model. Optuna + MLflow demonstrates that model selection and model
optimisation are distinct, trackable steps — a key portfolio differentiator.

**Independent Test**: Run `dvc repro tune` after evaluate completes; confirm N trial
runs appear in the `crash-severity-tune` MLflow experiment, each tagged with trial
number and hyperparameter values; confirm `params.yaml` is updated with best params
under `tune.best_params`; confirm the best trial's metric exceeds the pre-tune winner
metric from the A/B test.

**Acceptance Scenarios**:

1. **Given** the evaluate stage has declared a winner (ML or DL),
   **When** the tune stage runs,
   **Then** Optuna runs N trials using Bayesian optimisation (TPE sampler) on the
   winning model family's hyperparameter search space.

2. **Given** a tune trial completes,
   **When** the learner opens the MLflow UI,
   **Then** each trial appears as a separate run in `crash-severity-tune` tagged with
   `trial=<n>` and all hyperparameter values logged as params.

3. **Given** all N trials complete,
   **When** Optuna selects the best trial,
   **Then** the best hyperparameters are written to `params.yaml` under `tune.best_params`
   and DVC detects the param change and invalidates downstream stages.

4. **Given** best params written to `params.yaml`,
   **When** `dvc repro register` is run,
   **Then** the register stage uses the tuned model for registration as champion.

---

### User Story 4 — Local Workflow Orchestration via Airflow (Priority: P4)

As a learner, I can schedule and run the full six-stage ML pipeline as an Airflow DAG,
monitor each task's status and logs in the Airflow UI, configure retry behaviour, and
restart individual failed stages without restarting the whole pipeline.

**Why this priority**: Airflow is the standard local/on-premise orchestration choice.
This story demonstrates DAG design, task dependencies, and operational monitoring as
portfolio evidence.

**Independent Test**: Trigger the DAG manually, artificially fail one task, inspect
its logs in the UI, fix the issue, and restart only the failed task — confirming
upstream completed tasks are not re-run.

**Acceptance Scenarios**:

1. **Given** the Airflow instance is running,
   **When** the learner triggers the ML pipeline DAG,
   **Then** all six tasks execute in dependency order and each reports success in the UI.

2. **Given** the validate task fails due to a data quality issue,
   **When** the learner views the Airflow UI,
   **Then** the failed task is highlighted, its logs show the specific validation error,
   and downstream tasks remain pending without executing.

3. **Given** a failed task has been fixed,
   **When** the learner clears and restarts only that task,
   **Then** the DAG resumes from the failed task and completes without re-running
   previously successful tasks.

4. **Given** the pipeline is scheduled,
   **When** the scheduled time arrives,
   **Then** the DAG triggers automatically and all six stages complete without manual
   intervention.

---

### User Story 5 — Container-Native Orchestration via Kubeflow (Priority: P5)

As a learner, I can deploy the same six-stage ML pipeline to Kubernetes via Kubeflow
Pipelines, where each stage runs as an isolated container pod. I can submit a pipeline
run from the Kubeflow UI, monitor pod-level execution, and inspect per-stage logs —
all on a local Docker Desktop Kubernetes cluster.

**Why this priority**: Kubeflow represents the production-grade Kubernetes-native
approach. This story demonstrates that the same pipeline logic runs portably across
orchestrators — the key MLOps portfolio differentiator.

**Independent Test**: Submit the pipeline from the Kubeflow UI, confirm all six stages
appear as separate pods in Kubernetes, inspect the validate pod logs to confirm data
validation ran inside the container, and verify the trained model artifact is written
to the configured output path.

**Acceptance Scenarios**:

1. **Given** Kubeflow Pipelines is running on Docker Desktop Kubernetes,
   **When** the learner uploads and runs the compiled pipeline,
   **Then** all six stages appear as sequential steps in the Kubeflow UI with correct
   dependency arrows.

2. **Given** the pipeline is running,
   **When** the learner inspects a stage in the Kubeflow UI,
   **Then** the pod logs for that stage are visible and show the same output as the
   equivalent Airflow task log for the same data.

3. **Given** the validate stage fails due to a data quality issue,
   **When** the learner views the Kubeflow UI,
   **Then** the validate step is marked failed and downstream steps are not scheduled.

4. **Given** a successfully completed Kubeflow pipeline run,
   **When** the learner checks the experiment tracking UI,
   **Then** the training run logged from inside the Kubeflow container pod appears
   alongside runs from Airflow and local executions under the same experiment name.

---

### Edge Cases

- What happens when artifact storage is unavailable and the learner runs the pipeline?
  The ingest stage should fail with a clear connectivity error before any processing
  begins — no partial outputs should be written.
- What happens when the Kubernetes cluster is not running when a Kubeflow pipeline is
  submitted? The client should raise a connection error before any pods are scheduled.
- What happens when an experiment name is changed between training runs? Prior runs
  remain under the old experiment name; new runs appear under the new name — no data
  loss occurs.
- What happens when the same pipeline is triggered simultaneously by Airflow and a
  manual local run? Concurrent pipeline execution from multiple orchestrators is not
  supported — only one pipeline run should be active at a time. DVC's internal
  run-cache locking provides best-effort protection but is not a substitute for
  operational discipline.
- What happens when a Docker image for a pipeline stage is not available on the local
  registry when Kubeflow tries to schedule it? The pod enters an ImagePullBackOff state
  and the Kubeflow UI shows the error before any stage logic executes.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline MUST be fully reproducible from any prior git commit using
  **DVC** — `dvc repro` from a clean checkout with the pinned data version MUST
  reproduce the same model artifact without any manual steps.
- **FR-002**: All seven pipeline stages (ingest, validate, featurize, train_ml,
  train_dl, evaluate, register) MUST be defined as named stages in **`dvc.yaml`**
  with explicit `deps`, `outs`, and `cmd` fields so DVC can track the full dependency
  graph.
- **FR-003**: The validate stage MUST enforce a **Great Expectations** expectation suite
  against the ingested dataset and MUST halt all downstream DVC/Airflow/Kubeflow stages
  if any expectation fails.
- **FR-004a**: The `train_ml` stage MUST train a **PyCaret** model N times (one run per
  seed in `ab_test.seeds`), logging each as a separate MLflow run in `crash-severity-ml`
  tagged with `seed=<value>`. Each run MUST log `eout_macro_f1`, `eout_minority_recall`,
  `ein_macro_f1`, and `generalisation_gap` via `mlflow.evaluate()`. `mlflow.sklearn.autolog()`
  MUST be disabled. The best-seed model artifact MUST be saved to `models/best_ml_model.pkl`.
- **FR-004b**: The `train_dl` stage MUST train a **PyTorch** shallow MLP N times (one
  run per seed), logging each run in `crash-severity-dl` with per-epoch
  `ein_loss`/`eout_loss`/`gap_f1`. Final `eout_macro_f1` and `eout_minority_recall`
  MUST be logged via `mlflow.evaluate()` using a `mlflow.pyfunc` wrapper defined in
  `src/train_dl/pyfunc.py`. The best-seed checkpoint MUST be saved to `models/mlp_model.pth`.
- **FR-004c**: The `train_ml` and `train_dl` stages MUST be independent DVC stages with
  no dependency between them — both depend only on `featurize` outputs. They are executed
  sequentially (not in parallel) to avoid resource contention on a single-machine
  environment. Independence is demonstrated via the DAG structure, not simultaneous execution.
- **FR-005**: The `evaluate` stage MUST collect the `eout_macro_f1` scores from all N
  MLflow runs in each experiment and perform a **Welch's t-test** on the two
  distributions. The report MUST include: mean ± std for both models, p-value, Cohen's
  d effect size, 95% confidence intervals, declared winner, and constitutional gate
  PASS/FAIL. All metrics MUST be logged as an MLflow run artifact.
- **FR-005b**: If p ≥ 0.05 (no statistically significant difference), the simpler model
  (PyCaret ML) MUST be selected as the default winner. The result MUST be labelled
  "no significant difference" — the pipeline MUST NOT fail in this case.
- **FR-005c**: Both models MUST be evaluated on the identical `X_test`/`y_test` split
  from `featurize` — no re-splitting between A/B candidates (prevents leakage into
  the comparison). `X_test` MUST NOT be used during training, HPO, or NAS — only
  during the final `evaluate` stage A/B test and constitutional gate assertions.
- **FR-006**: The full pipeline MUST be expressible as an **Apache Airflow** DAG using
  the TaskFlow API, with task-level retry configuration, manual trigger support, and
  per-task log access via the Airflow UI.
- **FR-007**: The full pipeline MUST be expressible as a **Kubeflow Pipelines (KFP)**
  workflow where each stage is a containerised KFP component running as an isolated pod
  on Docker Desktop Kubernetes.
- **FR-008**: Each pipeline stage MUST accept input and output paths via environment
  variables or CLI arguments (not hardcoded), so the same `src/` business logic runs
  identically under `dvc repro`, Airflow TaskFlow, and KFP components.
- **FR-009**: Raw data and trained model artifacts MUST be tracked with **DVC**, with
  `.dvc` pointer files committed to git so any artifact version is retrievable via
  `dvc pull` at the corresponding git commit.
- **FR-010**: Each **Great Expectations** validation run MUST generate a Data Docs HTML
  report saved as a pipeline artifact, viewable in a browser without additional tooling.
- **FR-011**: Any individual pipeline stage MUST be runnable in isolation as a standalone
  Docker container (`docker run`) given its declared inputs are available, independent
  of the orchestrator.
- **FR-012**: A `tune` stage MUST run **Optuna** hyperparameter optimisation on the
  winning model family declared by `evaluate`. The search MUST use the TPE sampler
  (Bayesian optimisation). Each trial MUST be logged as a separate MLflow run in
  `crash-severity-tune` tagged with `trial=<n>`, `model_type=<winner>`, and all
  hyperparameter values as MLflow params. The best trial's hyperparameters MUST be
  written to `params.yaml` under `tune.best_params` so DVC detects the change and
  invalidates downstream stages.
- **FR-012b**: The `tune` stage search space MUST be defined in `params.yaml` under
  `tune.ml_search_space` (for PyCaret) and `tune.dl_search_space` (for PyTorch), so
  the search space is versioned alongside the code and modifiable without script changes.
- **FR-012c**: For PyTorch models, Optuna MUST use the `PyTorchLightningPruningCallback`
  (or equivalent) to prune unpromising trials early based on validation loss, reducing
  total search time.

### Key Entities

- **Pipeline Stage**: A named, self-contained unit of work with declared inputs,
  outputs, and an executable command. Runs identically under local, Airflow, and
  Kubeflow execution.
- **Expectation Suite**: A versioned, committed set of data quality rules applied to
  the ingested dataset. Produces a validation result and a quality report.
- **Experiment Run**: A single training execution with logged parameters, metrics,
  and model artifact. Belongs to a named experiment.
- **Registered Model**: A promoted model artifact identified by name and version,
  independent of the experiment run that produced it.
- **DAG**: The Airflow representation of the pipeline — seven tasks with explicit
  dependency edges, retry configuration, and a schedule. Each task calls `dvc repro <stage>`.
- **KFP Pipeline**: The Kubeflow Pipelines representation — seven containerised
  components with the same logical dependency structure as the DAG. Each component
  calls `dvc repro <stage>` with the project root mounted via a shared hostPath PVC.
- **Data Contract**: The column-level specification of what the raw dataset is allowed
  to contain — valid ranges, allowed values, acceptable null rates. Defined in
  `docs/data_contract.md` and encoded in `params.yaml` for programmatic GE enforcement.
- **Registry Receipt**: The `models/registry_receipt.json` file written by the register
  stage. Contains the registered model name, version, and alias. Serves as the DVC
  output for the register stage.
- **Optuna Trial**: A single hyperparameter configuration evaluated during the tune
  stage. Each trial trains the winning model with a specific set of hyperparameters
  and logs results as one MLflow run in `crash-severity-tune`.
- **Search Space**: The set of hyperparameter ranges Optuna explores for the winning
  model family. Defined in `params.yaml` under `tune.ml_search_space` or
  `tune.dl_search_space`. Versioned by DVC.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: `dvc repro` from a clean checkout runs the full pipeline end-to-end —
  ingest through register, including both `train_ml` and `train_dl` sequentially — with
  no manual steps and no files outside DVC/git tracking.
- **SC-002**: The Great Expectations validate stage automatically catches at least five
  distinct issue categories (schema, null rates, numeric ranges, categorical values, row
  count) and surfaces them in the generated Data Docs HTML report.
- **SC-003**: Both MLflow experiments (`crash-severity-ml` and `crash-severity-dl`) are
  queryable and comparable in the MLflow UI; the A/B test result (winner, metric delta)
  is visible as a logged artifact without accessing `mlruns/` directly.
- **SC-004**: The identical `src/` stage logic runs successfully under `dvc repro`,
  the Airflow DAG, and the KFP pipeline without duplicating any business logic.
- **SC-005**: Any pipeline stage runs as a Docker container via `docker run` with only
  environment variables as configuration — no host-installed Python packages required.
- **SC-006**: A model promoted to the MLflow Model Registry is loadable via
  `mlflow.pyfunc.load_model("models:/crash-severity@champion")` and produces predictions
  in under 30 seconds from a fresh Python session.
- **SC-007**: Both the Airflow UI and the Kubeflow Pipelines UI show task/step-level
  status and logs for every pipeline run; `train_ml` and `train_dl` appear as sequential
  tasks/steps with independent logs, demonstrating their independence in the DAG structure.
- **SC-008**: Any past pipeline run is fully reproducible by checking out the
  corresponding git commit and running `dvc pull && dvc repro` — confirmed by matching
  the produced model artifact checksum against the original run's logged artifact.

---

## Assumptions

- Docker Desktop is installed with Kubernetes enabled (single-node local cluster).
- Kubeflow Pipelines standalone is deployed to Docker Desktop Kubernetes — not the full
  Kubeflow platform. This is lighter weight and appropriate for a learning environment.
- Artifact remote storage is a local directory for simplicity; cloud remote storage is
  out of scope for this portfolio version.
- The crash dataset (`data/CGR_Crash_Data.csv`) is the sole input dataset; no external
  data sources are integrated.
- The experiment tracking server runs locally; no hosted tracking server is required.
- Airflow runs in standalone mode (single process) — not a distributed executor setup.
- The six pipeline stages are fixed for this portfolio. Additional stages are out of
  scope.
- Model serving (real-time inference endpoint) is out of scope; the register stage
  writes to the model registry only — no serving layer is deployed.
- All pipeline stages share a single base Docker image to reduce build complexity.
- Windows 11 with Docker Desktop is the development environment.
- The ML model and evaluation approach from the prior spec (binary classification,
  macro F1 primary metric, class weights for imbalance) are carried forward as
  defaults and are not re-specified here.
- Feature selection (which columns enter the model) is defined in `params.yaml` under
  `features.columns` and driven by EDA findings — not hardcoded in stage scripts.
- The data contract (valid ranges, allowed values, null thresholds per column) is
  defined collaboratively in `docs/data_contract.md` and encoded in `params.yaml`
  for programmatic Great Expectations enforcement.
- All stage scripts use `src/config.py` typed dataclass accessors to read `params.yaml`
  — direct `yaml.safe_load` dict access in stage scripts is prohibited.
- All stages call `mlflow.set_tracking_uri(config.mlflow.tracking_uri)` using the
  absolute path from `params.yaml` before any MLflow operation.
- Kubeflow Pipelines components mount the entire project root via a single hostPath
  PVC at `/app`, sharing `mlruns/`, `data/`, `models/`, `.dvc/cache/`, and `params.yaml`
  with the host and across all pods.
- Concurrent pipeline execution from multiple orchestrators is not supported.
  Only one pipeline run should be active at a time.
- Windows 11 Python scripts use LF line endings enforced via `.gitattributes`
  to ensure correct behaviour inside Linux-based Docker containers.
