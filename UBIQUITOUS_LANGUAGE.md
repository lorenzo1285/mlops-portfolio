# Ubiquitous Language

## Pipeline Structure

| Term | Definition | Aliases to avoid |
|------|------------|------------------|
| **ML Pipeline** | The full 10-stage workflow: validate → ingest → featurize → train_vae → encode → train_ml → train_dl → evaluate → tune → register. | "workflow", "project pipeline", "seven-stage pipeline", "eight-stage pipeline" |
| **Pipeline Stage** | A named, self-contained unit of work with declared inputs, outputs, and an executable command. Runs identically under local DVC and Kubeflow execution. | "step" (use only for Kubeflow UI context), "job", "process" |
| **Preprocessing Pipeline** | The sklearn `ColumnTransformer` object (imputer → encoder → scaler) fitted on the Train Split only and serialised to `preprocessing_pipeline.joblib`. | "pipeline" (ambiguous — always qualify as "preprocessing pipeline") |
| **Stage Script** | A `src/<stage>/run.py` file that implements one pipeline stage. Accepts configuration via environment variables; exits 0 on success, non-zero on failure. | "stage module", "runner", "script" |

## Orchestration

| Term | Definition | Aliases to avoid |
|------|------------|------------------|
| **KFP Pipeline** | The Kubeflow Pipelines representation of the ML Pipeline — 10 containerised components with the same logical dependency structure as the DVC DAG. The sole active orchestrator. | "Kubeflow DAG", "KFP workflow" |
| **KFP Component** | A single containerised unit in a KFP Pipeline, corresponding to one pipeline stage. Decorated with `@dsl.component`. Calls `dvc repro <stage>` with the project root mounted via PVC. | "pod", "step" (acceptable in Kubeflow UI context only) |
| **Airflow DAG** | An Airflow-based representation of the ML Pipeline, retained in `airflow/` as tutorial and learning reference material only. Not used for the active crash severity pipeline. | "active DAG", "production DAG" |
| **Airflow Task** | A single node in an Airflow DAG. Used only in tutorial DAGs — not part of the active pipeline. | "step", "job" |

## VAE & Generative Learning

| Term | Definition | Aliases to avoid |
|------|------------|------------------|
| **Denoising β-VAE** | The variational autoencoder trained unsupervised on the full feature matrix. Uses input dropout for neural inpainting, a β-weighted KL term for disentanglement, and learns a compressed latent representation of crash records. | "DVAE", "VAE", "autoencoder", "generative model" (always use full name on first reference) |
| **ELBO** | Evidence Lower Bound — the VAE training objective: `Reconstruction Loss + β × KL Divergence`. A converging (decreasing) ELBO indicates the VAE is learning the data distribution. | "VAE loss", "training loss" (always say "ELBO" for the combined objective) |
| **Reconstruction Loss** | The MSE between the decoder's output and the original (uncorrupted) input. One component of the ELBO. | "recon loss", "MSE loss" |
| **KL Divergence** | The Kullback-Leibler term that regularises the latent space to be approximately Gaussian. Weighted by β in the ELBO. | "KL loss", "KL term" (acceptable shorthand in code comments) |
| **β (beta)** | The scalar weight on the KL Divergence term in the ELBO. Controls the disentanglement-reconstruction trade-off. Tuned by Katib over `[0.5, 1.0, 2.0, 4.0, 8.0]`. | "beta weight", "KL weight" |
| **Neural Inpainting** | The technique of applying `nn.Dropout(p=0.15)` to the VAE's input during training, forcing the encoder to reconstruct clean data from corrupted observations. | "input dropout", "input corruption", "denoising" (denoising is the effect; neural inpainting is the mechanism) |
| **Encoder** | The neural network sub-module that compresses a preprocessed feature vector into a Latent Vector Z (mean μ and log-variance log σ²). Part of the Denoising β-VAE. | "compression network", "embedding network" |
| **Decoder** | The neural network sub-module that reconstructs a feature vector from a Latent Vector Z. Used during VAE training and Katib trials; not used by classifiers at inference. | "reconstruction network", "generative network" |
| **Reparameterization Trick** | The sampling technique `z = μ + ε × σ` (ε ~ N(0,1)) that allows gradients to flow through the stochastic latent sampling step during backpropagation. | "latent sampling", "VAE sampling" |

## Latent Representation

| Term | Definition | Aliases to avoid |
|------|------------|------------------|
| **Latent Space** | The 32-dimensional continuous space in which the Encoder embeds crash records. Points that are close together correspond to similar crashes. The Denoising β-VAE's KL term ensures the space is approximately Gaussian. | "embedding space", "Z space" (acceptable shorthand in code) |
| **Latent Vector (Z)** | A single 32-dimensional point in the Latent Space — the compressed representation of one crash record produced by the frozen Encoder. The input to both classifiers. | "embedding", "Z vector", "latent code" |
| **latent_dim** | The fixed dimensionality of every Latent Vector Z. Set to 32 in `params.yaml`; not tunable. | "embedding size", "Z dimension", "bottleneck size" |
| **Z_train_augmented** | The set of Latent Vectors for the Train Split after LSA has added synthetic Fatal-class vectors. Used to train both classifiers. | "augmented Z", "Z train" (Z_train is the pre-augmentation version — always qualify) |
| **Z_val** | The set of Latent Vectors for the Validation Split. Never augmented. Used for early stopping and Katib trial fitness. | "val Z", "validation latent vectors" |
| **Z_test** | The set of Latent Vectors for the Test Split. Never augmented. Used only in the `evaluate` stage for final A/B test results and constitutional gate assertions. | "test Z", "test latent vectors", "held-out Z" |

## Class Imbalance & Augmentation

| Term | Definition | Aliases to avoid |
|------|------------|------------------|
| **Latent-Space Augmentation (LSA)** | The technique of generating synthetic Fatal-class Latent Vectors by sampling Gaussian noise around the mean of real Fatal Z vectors in Z_train, until Fatal reaches the configured target ratio. Applied to Z_train only. | "SMOTE" (SMOTE is raw-feature oversampling — LSA is Z-space augmentation), "oversampling", "synthetic augmentation" |
| **PDO** | Property Damage Only — crash severity class 0. Accidents with no injuries, representing approximately 81% of the dataset. | "no-injury", "class 0", "non-injury crash" |
| **Injury** | Crash severity class 1. Accidents involving at least one injury but no fatalities, representing approximately 17.5% of the dataset. | "class 1", "non-fatal injury" |
| **Fatal class** | Crash severity class 2. Accidents involving at least one fatality, representing approximately 1.7% of the dataset. The safety-critical minority class guarded by the fatal recall constitutional gate (> 0.30). | "fatal crash", "class 2", "fatality class" |
| **Class Weight** | A per-class scalar computed as `N / (n_classes × class_count_c)` from the training split class distribution. Applied during classifier training to compensate for class imbalance. Computed at runtime — never hardcoded. | "sample weight", "loss weight" (acceptable in code context where `weight=` is the argument name) |

## Experiment Tracking

| Term | Definition | Aliases to avoid |
|------|------------|------------------|
| **Experiment** | An MLflow named experiment grouping all runs of one model family (`crash-severity-vae`, `crash-severity-ml`, `crash-severity-dl`, `crash-severity-tune`). | "project", "trial group" |
| **Experiment Run** | A single training execution with logged parameters, metrics, and model artifact. Belongs to one Experiment. Tagged with `seed=<value>` and `model_type`. | "training run", "MLflow run" |
| **Training Loop** | The N-seed iteration inside `train_ml` or `train_dl` that produces one Experiment Run per seed, each trained on Z_train_augmented. | "seed loop", "experiment loop" |
| **HPO Trial** | A single hyperparameter configuration (one β value) evaluated as a Kubernetes pod during a Katib Experiment. Retrains the Denoising β-VAE, re-encodes, trains the winner classifier, and reports `val_macro_f1` to Katib. | "Optuna trial", "tune run", "search iteration" |
| **Katib Experiment** | A Kubernetes CRD submitted by the `tune` stage that defines the β HPO search: objective metric (`val_macro_f1`), algorithm (Bayesian), search space `[0.5, 1.0, 2.0, 4.0, 8.0]`, and trial template. Not to be confused with an MLflow Experiment. | "Katib job", "HPO experiment", "tune job" |

## Models & Artifacts

| Term | Definition | Aliases to avoid |
|------|------------|------------------|
| **VAE Artifact** | The pair of serialised files `models/vae_encoder.pth` + `models/vae_decoder.pth`. DVC-tracked output of the `train_vae` stage. | "VAE model", "VAE checkpoint" |
| **Model Artifact** | A serialised trained classifier: `.pkl` for the XGBoost model, `.pth` for the MLP classifier. DVC-tracked output of a training stage. | "model", "checkpoint" (reserve for intermediate `.pth` saves during training) |
| **MLP Classifier** | The shallow PyTorch MLP operating on Z vectors: `Linear(32, 64) → ReLU → Dropout → Linear(64, 3)`. Produces 3-class predictions. Not to be confused with the Encoder or Decoder. | "ShallowMLP" (retired), "FlexMLP" (retired), "DL model", "neural network" |
| **Registered Model** | A model artifact promoted to the MLflow Model Registry, identified by name and alias (e.g. `models:/crash-severity@champion`). | "deployed model", "production model" |
| **Champion** | The alias assigned to the winning Registered Model version in the MLflow Model Registry. | "best model", "winner model" |
| **Winner** | The model family (ML/XGBoost or DL/MLP) declared superior by the statistical A/B test, or ML/XGBoost by default when p ≥ 0.05. | "best model", "champion model" (champion is the registry alias, winner is the A/B test outcome) |

## Data & Validation

| Term | Definition | Aliases to avoid |
|------|------------|------------------|
| **Expectation Suite** | A versioned, committed set of data quality rules applied to the raw dataset before any processing. Stored in `great_expectations/gx/expectations/crash_data_suite.json`. | "validation rules", "GE suite", "data contract" |
| **Validation Result** | The outcome of running an Expectation Suite against a dataset — pass or fail with per-expectation detail. | "validation output", "GE result" |
| **Data Docs** | The HTML report generated by Great Expectations after a validation run. Saved as a pipeline artifact. | "GE report", "validation report", "quality report" |
| **Feature Set** | The pre-crash observable columns selected during the `featurize` stage as model inputs. | "features", "input columns", "selected columns" |
| **Train Split** | The 70% portion of the dataset used to fit model weights and the Preprocessing Pipeline. Used also (concatenated with Val and Test, no labels) for unsupervised VAE pre-training. | "training set", "train data" |
| **Validation Split** | The 15% portion used for classifier early stopping and Katib HPO fitness scoring (via Z_val). Never used for final A/B test evaluation. | "dev set", "val data", "hold-out" |
| **Test Split** | The 15% portion strictly reserved for the `evaluate` stage A/B test (via Z_test). Must never be seen during classifier training, HPO, or Katib trials. | "test set", "held-out set" |
| **per-class P/R/F1 matrix** | A JSON artifact logged by `train_ml` and `train_dl` per run, reporting precision, recall, and F1 for each of the three severity classes (PDO, Injury, Fatal). | "classification report", "confusion matrix" (different concept) |

## Statistical Evaluation

| Term | Definition | Aliases to avoid |
|------|------------|------------------|
| **A/B Test** | The Welch's t-test comparing the distribution of `eout_macro_f1` scores across all seeds for ML (XGBoost) vs DL (MLP). Produces p-value, Cohen's d, and 95% CIs. | "model comparison", "evaluation", "benchmark" |
| **Generalisation Gap** | `eout_macro_f1 − ein_macro_f1`. Negative value indicates overfitting. Mandatory MLflow metric for every Experiment Run. | "gap", "train-test gap", "overfit score" |
| **Fatal Recall Gate** | The constitutional requirement that the winner's fatal class recall exceeds 0.30 on Z_test. Blocks registration if not met. Complements the macro F1 gate (> 0.45). | "recall threshold", "minority recall" (retired — now specifically "fatal recall") |

## Relationships

- The **ML Pipeline** is expressed as a DVC `dvc.yaml` DAG (local execution) and a **KFP Pipeline** (Kubeflow execution) — the same logic, two execution contexts.
- A **Pipeline Stage** is implemented as a **Stage Script**, wrapped as a **KFP Component**.
- The `train_vae` stage produces a **VAE Artifact** (encoder + decoder). The `encode` stage uses the frozen **Encoder** to produce **Z_val**, **Z_test**, and **Z_train_augmented**.
- **LSA** operates on **Z_train** (before augmentation) and produces **Z_train_augmented** by adding synthetic **Fatal class** Latent Vectors. **Z_val** and **Z_test** are never modified.
- The **MLP Classifier** and **XGBoost** classifier both take **Z_train_augmented** as training input and **Z_test** as the final evaluation input.
- A **Katib Experiment** runs N **HPO Trials**; each trial uses **Z_val** for fitness and logs one **Experiment Run** to `crash-severity-tune`.
- An **A/B Test** compares `crash-severity-ml` vs `crash-severity-dl` **Experiments** and declares a **Winner**.
- The **Winner** must pass the macro F1 gate (> 0.45) and the **Fatal Recall Gate** (> 0.30) before being registered as the **Champion** **Registered Model**.

## Example Dialogue

> **Dev:** "The VAE trains on all data — does that mean it's seeing the test set?"
> **Domain expert:** "Yes, but only the features (X) — never the labels. The **Denoising β-VAE** is unsupervised. You can't leak a label that isn't provided. The **Test Split** contamination rule applies to supervised classifier training: **Z_test** must never be used to train the **MLP Classifier** or XGBoost."

> **Dev:** "So Z_test is always the true held-out set?"
> **Domain expert:** "Exactly. **Z_val** is for early stopping and **HPO Trial** fitness — that's what Katib reads via `val_macro_f1`. **Z_test** is only touched in the `evaluate` stage for the **A/B Test** and the **Fatal Recall Gate**."

> **Dev:** "What does LSA do exactly, and why isn't it just SMOTE?"
> **Domain expert:** "**Latent-Space Augmentation (LSA)** samples Gaussian noise around the mean of real **Fatal class** **Latent Vectors** in **Z_train**. It works because the **Latent Space** is smooth and continuous — Gaussian interpolation near a **Fatal class** centroid produces plausible, novel fatal crash representations. SMOTE interpolates in the raw feature space, where mixing a continuous speed limit with a categorical weather code is meaningless. LSA is principled; raw SMOTE is not."

> **Dev:** "Is the per-class P/R/F1 matrix logged per seed or just once?"
> **Domain expert:** "Per seed — every **Experiment Run** in `crash-severity-ml` and `crash-severity-dl` logs its own **per-class P/R/F1 matrix** as a JSON artifact. The `evaluate` stage then computes the aggregated comparison. You can inspect any individual seed's **Fatal class** recall directly in MLflow."

> **Dev:** "What's the difference between the Winner and the Champion?"
> **Domain expert:** "The **Winner** is the A/B test outcome — the model family (XGBoost or MLP) with the higher mean `eout_macro_f1` and passing the **Fatal Recall Gate**. The **Champion** is the MLflow Model Registry alias pointing to the best-seed **Model Artifact** from the winning family. You can have a Winner without a Champion if the registration step fails."

## Flagged Ambiguities

- **"pipeline"** appears in three distinct senses: the full **ML Pipeline**, the sklearn **Preprocessing Pipeline**, and the **KFP Pipeline**. Always qualify — never say "the pipeline" without context.
- **"run"** is used for both an **Experiment Run** (MLflow) and a pipeline execution. Use "pipeline run" for the latter and "Experiment Run" for MLflow.
- **"model"** was used for both a **Model Artifact** (file on disk) and a trained model object in memory. Use "model artifact" when referring to serialised files.
- **"experiment"** is overloaded: an MLflow **Experiment** (a named run group) vs. a **Katib Experiment** (a Kubernetes CRD). Always qualify with "MLflow" or "Katib".
- **"task"** appears in two unrelated contexts: **Airflow Task** (DAG node, tutorial only) and implementation task (T-numbers in tasks.md). These are distinct.
- **"architecture"** appears in two senses: MLP layer configuration (now simply described as "architecture" in context since NAS is removed) and system/software architecture. Qualify when ambiguous.
- **"epoch"** must only refer to one pass over training data during weight optimisation. The NAS concept of "architecture generation" is retired along with NAS itself.
- **"Z_train" vs "Z_train_augmented"**: Z_train is the pre-LSA set of latent vectors for the Train Split; Z_train_augmented is after LSA. Only Z_train_augmented is passed to classifiers. Always use the full qualified name.
- **"FlexMLP"** and **"ShallowMLP"** are both retired. The current DL classifier is **MLP Classifier** — a fixed-architecture `Linear(32,64)→ReLU→Dropout→Linear(64,3)` operating on Z vectors. No NAS, no variable depth.
- **"val_macro_f1" vs "eout_macro_f1"**: `val_macro_f1` is the Katib fitness metric computed on **Z_val** inside **HPO Trials**. `eout_macro_f1` is the final test metric computed on **Z_test** in the `evaluate` stage. These are different numbers from different data splits — never use them interchangeably.
- **"minority class"** is retired as a standalone term. Now refers specifically to **Fatal class** (class 2). The **Fatal Recall Gate** replaces the generic "minority recall threshold".
