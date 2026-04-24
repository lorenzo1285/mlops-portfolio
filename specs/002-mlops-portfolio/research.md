# Research: MLOps Learning Portfolio

**Branch**: `002-mlops-portfolio` | **Date**: 2026-04-22

---

## Decision 1: DVC Version and Pipeline API

**Decision**: DVC 3.x with `dvc.yaml` pipeline format.

**Rationale**: DVC 3 stabilised the `dvc.yaml` declarative pipeline format with
explicit `cmd`, `deps`, `outs`, and `params` fields. Stage caching is automatic —
a stage is skipped if its deps and params are unchanged. `dvc repro` walks the DAG
and re-runs only affected stages. This is the current industry standard.

**Alternatives considered**:
- DVC 2.x: Still works but missing some 3.x UX improvements. Not chosen — no reason
  to pin to an older minor.
- Metaflow: Different paradigm (Python decorators, not YAML); harder to pair with
  Kubeflow. Not chosen.

**Key DVC concepts used**:
- `dvc init` — initialises `.dvc/` tracking directory
- `dvc remote add` — registers local dir as remote storage
- `dvc add data/raw/CGR_Crash_Data.csv` — creates `data/raw/CGR_Crash_Data.csv.dvc`
  pointer file tracked by git
- `dvc repro` — executes changed stages in dependency order
- `dvc push` / `dvc pull` — sync artifacts to/from remote

---

## Decision 2: Great Expectations Version (v1.x vs 0.x)

**Decision**: Great Expectations v1.x (current stable release).

**Rationale**: GE 1.0 introduced a completely new API (`gx.get_context()`,
`context.suites`, `ValidationDefinition`). The 0.x API (`DataContext`,
`get_expectation_suite`) is deprecated. Since this is a new project with no GE
history, use v1 from the start to avoid a future migration.

**Key v1 API used**:
```python
import great_expectations as gx
context = gx.get_context()                          # loads great_expectations.yml
suite = context.suites.add(ExpectationSuite(...))   # create/load suite
data_source = context.data_sources.add_pandas(...)
batch = data_source.get_asset(...).add_batch_definition_whole_dataframe(...)
validation_def = context.validation_definitions.add(
    ValidationDefinition(name="...", data=batch, suite=suite)
)
results = validation_def.run()
context.build_data_docs()                           # generates HTML Data Docs
```

**Alternatives considered**:
- GE 0.18.x (last 0.x): Familiar API but deprecated. Not chosen.
- Pandera: Simpler, no Data Docs HTML. Not chosen — Data Docs are a deliverable (FR-010).
- TDDA: Academic tool, not portfolio-relevant. Not chosen.

---

## Decision 3: MLflow Autolog Strategy

**Decision**: `mlflow.sklearn.autolog()` for PyCaret/sklearn models; manual epoch
logging for any future PyTorch work. MLflow 3.x tracking server on local filesystem.

**Rationale**: `mlflow.sklearn.autolog()` captures parameters, metrics, and the model
artifact without per-metric logging calls. PyCaret is sklearn-compatible, so autolog
works out of the box once enabled before `setup()`. The MLflow Model Registry is
embedded in the local tracking store — no separate server needed.

**Model registry URI pattern**: `models:/crash-severity/<version>`

**Key MLflow 3.x changes from 2.x**:
- `mlflow.MlflowClient()` is the unified client (no separate `MlflowClient` import)
- Model registry stages (`Staging`, `Production`) replaced by aliases in 3.x — use
  `client.set_registered_model_alias(name, alias, version)` instead of
  `transition_model_version_stage()`

---

## Decision 4: Kubeflow Pipelines Standalone on Docker Desktop

**Decision**: Kubeflow Pipelines (KFP) standalone v2, deployed via `kubectl apply`,
on Docker Desktop Kubernetes. KFP SDK v2 for pipeline authoring.

**Rationale**: Full Kubeflow (with Istio, KServe, Katib) requires 8GB+ RAM. KFP
standalone installs only the Pipelines component and its dependencies (~2GB), which
fits Docker Desktop's default Kubernetes allocation.

**Installation path** (one-time setup):
```bash
# Enable Kubernetes in Docker Desktop Settings → Kubernetes → Enable Kubernetes
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources/base?ref=2.2.0"
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=2.2.0"
# UI available at: kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8888:80
```

**KFP v2 SDK pattern**:
```python
from kfp import dsl, compiler

@dsl.component(base_image="mlops-portfolio:latest")
def validate_stage(input_path: str, output_path: str) -> str:
    import subprocess
    result = subprocess.run(
        ["python", "-m", "src.validate.run"],
        env={"INPUT_PATH": input_path, "OUTPUT_PATH": output_path},
        check=True
    )
    return output_path

@dsl.pipeline(name="crash-severity-pipeline")
def crash_severity_pipeline():
    ingest = ingest_stage(...)
    validate = validate_stage(...).after(ingest)
    featurize = featurize_stage(...).after(validate)
    ...

compiler.Compiler().compile(crash_severity_pipeline, "pipeline.yaml")
```

**Alternatives considered**:
- Full Kubeflow: Too heavy for Docker Desktop. Not chosen.
- Argo Workflows directly: No Python SDK; harder to learn alongside Airflow. Not chosen.
- Prefect: Excellent tool but not on the portfolio tool list. Not chosen.

---

## Decision 5: Docker Base Image Strategy

**Decision**: Single base image `python:3.12-slim` + `uv` + project dependencies.
All six stage scripts are baked into the same image; entry point is selected per-stage
via `CMD` or `ENTRYPOINT` override.

**Rationale**: Building six separate images multiplies build time and registry space.
A single image with all deps installed means any stage's `run.py` is available. Stage
selection is done at runtime via the Docker command or KFP component definition.

**Dockerfile approach**:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen
COPY src/ ./src/
COPY great_expectations/ ./great_expectations/
COPY params.yaml ./
ENV PYTHONPATH=/app
```

**Alternatives considered**:
- One Dockerfile per stage: Maximum isolation but 6× build overhead. Overkill for a
  learning portfolio. Not chosen.
- `python:3.12` (full image): ~1GB vs ~200MB for slim. Not chosen.

---

## Decision 6: Airflow DAG Rewrite Strategy

**Decision**: Rewrite `airflow/dags/crash_ml_pipeline.py` using the TaskFlow API
(`@task` decorator). Each task calls the corresponding `src/<stage>/run.py` via
`subprocess.run` or direct Python import.

**Rationale**: TaskFlow API (Airflow 2.0+) passes data between tasks via XCom
automatically and produces cleaner code than classic `PythonOperator`. The existing
tutorial DAGs in `airflow/dags/` are kept for reference — a new `crash_ml_pipeline.py`
is written from scratch (no modification of existing files).

**Task structure**:
```python
from airflow.decorators import dag, task
from datetime import datetime
import subprocess, os

@dag(schedule=None, start_date=datetime(2026,1,1), catchup=False, tags=["mlops"])
def crash_severity_pipeline():

    @task()
    def ingest():
        subprocess.run(["python", "-m", "src.ingest.run"], check=True)

    @task()
    def validate():
        subprocess.run(["python", "-m", "src.validate.run"], check=True)

    ingest() >> validate() >> ...

crash_severity_pipeline()
```

---

## Decision 7: PyTorch MLP Architecture

**Decision**: ShallowMLP — Input(d) → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3)
→ Linear(64) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(1). Maximum 3 hidden layers
(constitution IV).

**Loss**: `BCEWithLogitsLoss(pos_weight=torch.tensor([2.74]))` for training;
`BCEWithLogitsLoss()` (no weight) for validation loss tracking.

**Training controls**: Adam optimiser, lr=1e-3; early stopping with patience=10
monitoring validation loss; batch_size=256 for training, 512 for validation/test.

**Rationale**: Sample complexity for N=74k, d≈50-80 is insufficient for deep networks.
Shallow MLP with Dropout + BatchNorm + Early Stopping provides SLT-sound generalisation
control. Matches constitution IV exactly.

**Per-epoch MLflow logging**:
```python
for epoch in range(max_epochs):
    ein_loss, ein_f1 = train_one_epoch(...)
    eout_loss, eout_f1 = evaluate(...)
    mlflow.log_metrics({
        "ein_loss": ein_loss, "eout_loss": eout_loss,
        "gap_f1": eout_f1 - ein_f1, "eout_macro_f1": eout_f1
    }, step=epoch)
```

---

## Decision 8: Statistical A/B Test Design

**Decision**: Train each model N=10 times across different random seeds. Collect the
distribution of `eout_macro_f1` scores per model. Run a **Welch's t-test** to test
whether the difference in means is statistically significant. Report Cohen's d and
95% CIs. Simpler model (ML/PyCaret) wins ties.

**Rationale**: A single training run is one sample from a random variable. Variance
due to weight initialisation (DL) and model selection randomness (ML) can be large
enough to flip apparent winners. A proper statistical test gives a defensible,
reproducible conclusion — exactly what separates a portfolio piece from a tutorial.
Constitution principle "Always report effect size and CIs alongside p-values" mandates
this approach.

**Why Welch's t-test** (not paired, not Mann-Whitney):
- Welch's handles unequal variances between the two groups (likely here — DL has higher
  variance from weight initialisation)
- N=10 is small but sufficient for a t-test with expected effect sizes > 0.5 (Cohen's d)
- Paired t-test would require the same data perturbation across both models — not
  applicable since the test set is fixed and seeds affect training only
- Mann-Whitney U is non-parametric but less powerful for N=10; Welch's is preferred
  when approximate normality holds (macro F1 scores tend to be approximately normal)

**Statistical procedure**:
```python
from scipy import stats
import numpy as np

scores_ml = [run.data.metrics["eout_macro_f1"] for run in ml_runs]  # N floats
scores_dl = [run.data.metrics["eout_macro_f1"] for run in dl_runs]  # N floats

t_stat, p_value = stats.ttest_ind(scores_ml, scores_dl, equal_var=False)

# Cohen's d
pooled_std = np.sqrt((np.std(scores_ml)**2 + np.std(scores_dl)**2) / 2)
cohens_d = (np.mean(scores_ml) - np.mean(scores_dl)) / pooled_std

# 95% CIs
n = len(scores_ml)
ci_ml = (np.mean(scores_ml) - 1.96*np.std(scores_ml)/np.sqrt(n),
         np.mean(scores_ml) + 1.96*np.std(scores_ml)/np.sqrt(n))
ci_dl = (np.mean(scores_dl) - 1.96*np.std(scores_dl)/np.sqrt(n),
         np.mean(scores_dl) + 1.96*np.std(scores_dl)/np.sqrt(n))

alpha = params["ab_test"]["alpha"]  # 0.05
if p_value < alpha:
    winner = "ml" if np.mean(scores_ml) > np.mean(scores_dl) else "dl"
    significant = True
else:
    winner = "ml"   # simpler model wins ties
    significant = False
```

**N=10 seeds justification**: With expected effect size d≈0.5 and alpha=0.05, N=10
gives ~40% statistical power — low, but acceptable for a learning portfolio where the
goal is to demonstrate the methodology, not claim production-grade inference. A note in
the report documents this limitation.

**Alternatives considered**:
- Single-run comparison: Fast but unreliable — one lucky seed can reverse the ranking.
  Not chosen.
- Cross-validation instead of seeds: Proper alternative but breaks the DVC pipeline
  design (featurize produces fixed splits). Not chosen for this portfolio version.
- Bootstrap resampling: More statistically sound but complex to explain. Not chosen.

---

## Decision 9: params.yaml Structure

**Decision**: Single `params.yaml` at repo root, read by DVC stages and importable
by `src/` scripts. KFP components and Airflow tasks read the same file.

```yaml
data:
  raw_path: data/raw/CGR_Crash_Data.csv
  processed_dir: data/processed/
  test_size: 0.2
  random_state: 42
  sentinel_value: 999          # DRIVER1AGE/DRIVER2AGE → NaN

model:
  class_weight_neg: 0.61
  class_weight_pos: 2.74
  n_select: 3
  macro_f1_threshold: 0.55
  minority_recall_threshold: 0.40

dl:
  epochs: 100
  patience: 10
  batch_size: 256
  lr: 0.001
  hidden_1: 128
  hidden_2: 64
  dropout: 0.3

mlflow:
  experiment_name_ml: crash-severity-ml
  experiment_name_dl: crash-severity-dl
  model_name: crash-severity
  tracking_uri: mlruns/

great_expectations:
  suite_name: crash_data_suite
  datasource_name: crash_data

ab_test:
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # N=10 runs per model
  alpha: 0.05                               # significance level
  tiebreak: ml                              # simpler model wins on p >= alpha
```
