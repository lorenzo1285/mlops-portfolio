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

## Decision 6: Airflow DAG Strategy ~~SUPERSEDED by Decision 10~~

> **Superseded 2026-04-26**: Airflow is eliminated from the active pipeline scope.
> See Decision 10 for the KFP-only rationale. The `airflow/` directory is kept as
> tutorial/learning reference material only; no active `crash_ml_pipeline.py` DAG
> will be created.

---

## Decision 7: PyTorch MLP Architecture ~~SUPERSEDED by Decision 13~~

> **Superseded 2026-04-26**: The MLP now operates on 32-dimensional Z vectors from the
> frozen VAE encoder, not on raw preprocessed features. EvoTorch NAS removed. Target is
> 3-class (PDO / Injury / Fatal). See Decision 13 for the updated architecture.
> Loss is now `CrossEntropyLoss(weight=computed_class_weights)` (not BCEWithLogitsLoss).

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

## Decision 9: params.yaml Structure ~~SUPERSEDED by Decision 15~~

> **Superseded 2026-04-26**: New `vae.*` and `encode.*` sections added. `model.*`
> updated for multi-class (n_classes=3, new thresholds). `dl.*` updated for Z-vector
> input (input_dim=32). Class weights now computed at runtime from training split.
> See Decision 15 for the updated structure.

---

## Decision 11: Denoising β-VAE Architecture (train_vae stage)

**Decision**: Encoder `[256, 128, 64] → latent_dim=32` (configurable dims, fixed latent).
Decoder mirrors encoder. Input corruption via `nn.Dropout(p=0.15)` (neural inpainting).
β-weighted ELBO loss: `ELBO = Reconstruction_loss + β × KL_divergence`.

**Architecture**:
```python
class Encoder(nn.Module):
    # Input → Linear(256) → LayerNorm → ReLU → Linear(128) → LayerNorm → ReLU
    # → Linear(64) → LayerNorm → ReLU → [Linear(32) μ, Linear(32) log_σ²]
    # Reparameterization: z = μ + ε × σ, ε ~ N(0,1)

class Decoder(nn.Module):
    # z(32) → Linear(64) → ReLU → Linear(128) → ReLU → Linear(256) → ReLU → Linear(d)

class DenoisingBetaVAE(nn.Module):
    def forward(self, x):
        x_corrupted = F.dropout(x, p=self.dropout_p, training=True)  # neural inpainting
        μ, log_σ² = self.encoder(x_corrupted)
        z = self.reparameterize(μ, log_σ²)
        x_hat = self.decoder(z)
        return x_hat, μ, log_σ², z
```

**ELBO loss**:
```python
reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')  # target is clean x, not corrupted
kl_loss = -0.5 * torch.mean(1 + log_σ² - μ.pow(2) - log_σ².exp())
elbo = reconstruction_loss + beta * kl_loss
```

**Training on full X** (no Y — unsupervised): `X = np.concatenate([X_train, X_val, X_test])`.
The VAE cannot overfit to the test target because target labels are never provided.

**MLflow experiment**: `crash-severity-vae`. Single run per pipeline execution.
Per-epoch: `vae_elbo`, `vae_reconstruction_loss`, `vae_kl_loss` with `step=epoch`.
Best checkpoint (lowest validation ELBO) saved to `models/vae_encoder.pth`, `models/vae_decoder.pth`.

**Rationale**: LayerNorm preferred over BatchNorm for the VAE because batch statistics
can be unstable during reconstruction of tabular data with mixed scales. Dropout input
corruption is the key denoising mechanism — the VAE must reconstruct clean X from
corrupted input, which forces it to learn robust latent representations.

**Alternatives considered**:
- VQ-VAE: Discrete latent space less suited to continuous tabular interpolation. Not chosen.
- AE (no KL): No generative capability — LSA would be meaningless without the smooth
  latent space guaranteed by the KL term. Not chosen.
- β=1 fixed: Useful baseline but β is the main HPO target — must be configurable.

---

## Decision 12: Latent-Space Augmentation (encode stage)

**Decision**: After encoding all splits with the frozen VAE encoder, apply LSA to
`Z_train` only. Generate synthetic fatal Z vectors by sampling Gaussian noise around
the mean of real fatal Z vectors in Z_train, until fatal class reaches `lsa_target_ratio`
(default 0.05 = 5% of Z_train rows).

**Algorithm**:
```python
fatal_mask = (y_train == 2)  # Fatal class index
z_fatal = Z_train[fatal_mask]
if len(z_fatal) < min_fatal_samples:
    raise RuntimeError(f"Too few fatal samples: {len(z_fatal)} < {min_fatal_samples}")

fatal_mean = z_fatal.mean(axis=0)   # shape (32,)
fatal_std  = z_fatal.std(axis=0)    # shape (32,) — per-dimension

n_current = len(Z_train)
n_target_fatal = int(n_current * lsa_target_ratio)
n_synthetic = max(0, n_target_fatal - len(z_fatal))

synthetic_z = fatal_mean + np.random.randn(n_synthetic, 32) * fatal_std
Z_train_augmented = np.vstack([Z_train, synthetic_z])
y_train_augmented = np.hstack([y_train, np.full(n_synthetic, 2)])
```

**Critical constraint**: `Z_val` and `Z_test` are NEVER augmented — they retain the
true class distribution for unbiased evaluation.

**Rationale**: Linear interpolation / Gaussian noise in the VAE latent space is
semantically meaningful because the VAE's KL term forces the latent space to be
approximately Gaussian and continuous. This is the property that makes LSA sound:
sampling near a fatal Z centroid produces plausible-but-novel fatal crash representations.
Raw-feature SMOTE lacks this property because the original feature space is mixed-type
and high-dimensional.

**Alternatives considered**:
- Per-seed SMOTE in raw feature space: Constitutionally prohibited (Principle III v3.1.0).
- Weighted loss only (no augmentation): Viable, but extremely low fatal frequency (<1%)
  makes gradient signal too sparse even with high class weight. LSA addresses both
  frequency and gradient signal. Chosen as complement to class weights.

---

## Decision 13: XGBoost Multi-Class Classifier (replaces PyCaret)

**Decision**: `xgboost.XGBClassifier(objective='multi:softprob', num_class=3)` trained
directly on `Z_train_augmented`. Class weights applied via `sample_weight` computed
from training split class distribution: `w_c = N / (3 × class_count_c)`.

**Key parameters** (in `params.yaml model.*` or passed via XGBoost config):
```python
clf = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=seed,
    eval_metric='mlogloss',
    early_stopping_rounds=10,
)
clf.fit(
    Z_train_augmented, y_train_augmented,
    sample_weight=compute_sample_weights(y_train_augmented),
    eval_set=[(Z_val, y_val)],
    verbose=False
)
```

**MLflow logging** (`mlflow.sklearn.autolog()` disabled per constitution):
```python
mlflow.log_params({...})  # XGBoost hyperparams
mlflow.log_metrics({
    "ein_macro_f1": ..., "eout_macro_f1": ...,
    "eout_fatal_recall": ..., "generalisation_gap": ...
})
mlflow.log_artifact("per_class_matrix.json")  # per-class P/R/F1
mlflow.sklearn.log_model(clf, "model")
```

**Rationale**: XGBoost on 32-dim Z vectors is an extremely well-posed problem — tree
splits on a smooth, disentangled latent space are highly effective. PyCaret's `compare_models`
is replaced because (a) the model family is now fixed (XGBoost) and (b) PyCaret's
autolog conflicts with the manual metric tracking required by constitution V.

**Alternatives considered**:
- PyCaret (original plan): compare_models selects the best sklearn estimator. No longer
  appropriate since the model selection question is now XGBoost vs MLP, not which
  sklearn estimator. Not chosen.
- RandomForest on Z: Valid alternative but XGBoost is more portfolio-relevant.

---

## Decision 14: Multi-Class Target Encoding

**Decision**: Map CRASHSEVER string values to integers at the `featurize` stage:
- `"Property Damage Only"` → `0` (PDO)
- `"Injury"` → `1`
- `"Fatal"` → `2`

Encoding is applied to `y_train`, `y_val`, `y_test` (saved as `.npy` int arrays).
The mapping is hardcoded in `Featurizer` as the canonical encoding for this dataset —
not in `params.yaml`, because the string-to-int mapping is a dataset invariant, not
a hyperparameter.

**Class distribution estimate** (from ~74,309 rows):
- PDO: ~60,000 rows (~81%)
- Injury: ~13,000 rows (~17.5%)
- Fatal: ~1,300 rows (~1.7%)  ← LSA target: augment to 5% of Z_train

**Rationale**: Native 3-class encoding avoids the information loss of binary collapse
(Injury+Fatal). The Fatal class is the safety-critical outcome; a model that ignores it
is constitutionally gated (fatal recall > 0.30, Principle VI v3.1.0).

---

## Decision 15: Updated params.yaml Structure

**Decision**: Add `vae.*` and `encode.*` sections. Update `model.*` for multi-class.
Update `dl.*` for Z-vector input. Class weights computed at runtime, not hardcoded.

```yaml
data:
  raw_path: data/raw/CGR_Crash_Data.csv
  processed_dir: data/processed/
  train_size: 0.70
  val_size: 0.15
  test_size: 0.15
  random_state: 42
  sentinel_value: 999

vae:
  encoder_dims: [256, 128, 64]
  latent_dim: 32                # fixed — not tunable via Katib
  beta: 1.0                     # runtime default; overwritten by tune.best_params.beta
  dropout_p: 0.15               # neural inpainting corruption rate
  epochs: 200
  patience: 20
  batch_size: 512
  lr: 0.001
  experiment_name: crash-severity-vae

encode:
  lsa_target_ratio: 0.05        # augment fatal class to 5% of Z_train
  min_fatal_samples: 10         # halt if fewer real fatal samples

model:
  n_classes: 3
  macro_f1_threshold: 0.45      # was 0.55
  fatal_recall_threshold: 0.30  # replaces minority_recall_threshold: 0.40

dl:
  input_dim: 32                 # Z-vector dimensionality
  hidden_dim: 64
  dropout: 0.3
  epochs: 100
  patience: 10
  batch_size: 256
  lr: 0.001

mlflow:
  experiment_name_ml: crash-severity-ml
  experiment_name_dl: crash-severity-dl
  experiment_name_vae: crash-severity-vae
  experiment_name_tune: crash-severity-tune
  model_name: crash-severity
  tracking_uri: mlruns/

great_expectations:
  suite_name: crash_data_suite
  datasource_name: crash_data

ab_test:
  seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  alpha: 0.05
  tiebreak: ml

tune:
  best_params:
    beta: null                  # written by tune stage after Katib completes
```

---

## Decision 10: KFP as the Sole Orchestrator (supersedes Decision 6)

**Decision**: Kubeflow Pipelines (KFP) v2 is the sole production orchestrator.
Apache Airflow is removed from the active implementation scope.

**Rationale**: KFP enforces container-native discipline that Airflow standalone
mode does not — each stage runs in an isolated, reproducible pod on Kubernetes.
The key portfolio demonstrations (container isolation, pod-level logging, Kubernetes
deployment, DVC caching inside a pod) are all delivered by KFP alone. Maintaining
a parallel Airflow DAG adds 5 tasks of implementation overhead without introducing
new MLOps concepts beyond what KFP already demonstrates.

The `airflow/` directory is retained as tutorial/learning reference material
(tutorial DAGs 01, 02, 03) so Airflow concepts remain visible in the repo, but
no active `crash_ml_pipeline.py` DAG is created or maintained.

**Alternatives considered**:
- Keep both Airflow + KFP (original plan): Demonstrates both orchestrators side by
  side, making trade-offs concrete. Not chosen — single-machine complexity without
  proportional portfolio learning value at this stage.
- Replace KFP with Airflow only: Simpler setup but loses container-native/Kubernetes
  demonstration, which is the harder and more production-relevant skill. Not chosen.

**Impact on constitution**: Principle IX amended from "parity: Airflow and Kubeflow"
to "KFP-only" in constitution v3.0.0 (2026-04-26).
