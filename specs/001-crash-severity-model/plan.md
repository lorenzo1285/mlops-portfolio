# Implementation Plan: Crash Severity Prediction Model

**Branch**: `001-crash-severity-model` | **Date**: 2026-03-30 | **Spec**: [spec.md](./spec.md)

---

## Summary

Build a binary crash severity classifier (no-injury vs injury+fatal) on the CGR Crash Data (74,309 records, Grand Rapids 2008–2017) using only pre-crash observable features. The ML phase uses PyCaret to compare the full hypothesis set in one shot. The DL phase uses a shallow PyTorch MLP with explicit SLT controls (Dropout, BatchNorm, Early stopping). The best model overall is selected by out-of-sample macro F1. Post-crash outcome columns are reserved for evaluation enrichment only.

---

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: pandas, numpy, scikit-learn, pycaret, torch, mlflow, matplotlib, seaborn, jupyter (all installed)
**Storage**: Local files — CSV input, joblib for pipeline/model artifacts, JSON for experiment logs
**Testing**: Metric thresholds on held-out test set (macro F1 > 0.55, minority class recall > 0.40)
**Target Platform**: Local Jupyter notebook environment
**Project Type**: ML/DL pipeline — notebook-based, end-to-end from raw CSV to saved model artifact
**Performance Goals**: Macro F1 > 0.55; minority class (injury+fatal) recall > 0.40
**Constraints**: MLP MUST be shallow (2–3 layers max) per sample complexity analysis; no data leakage — all preprocessing fit on train only; PyCaret handles ML phase, PyTorch handles DL phase
**Scale/Scope**: 74,309 records × ~50–80 features post-encoding; single machine, no distributed compute

---

## Constitution Check

Constitution is currently a blank template — no project-level principles defined. No gate violations.

---

## Learning Theory Framework

| Element | Definition |
|---|---|
| **X** | Pre-crash feature vectors ∈ Rᵈ (d ≈ 50–80 after encoding) |
| **Y** | {0, 1} — 0 = no-injury (PDO), 1 = injury+fatal |
| **f** | P(Y=1 \| X=x) — unknown true severity probability |
| **H₁…Hₙ** | Full ML hypothesis set explored via PyCaret `compare_models()` |
| **H_DL** | Shallow PyTorch MLP (2–3 layers) — explicit SLT controls |
| **L** | Weighted binary cross-entropy — `BCEWithLogitsLoss(pos_weight=2.74)` for MLP; class_weights={0:0.61, 1:2.74} for PyCaret ML models |
| **A_ML** | PyCaret handles optimiser per model (L-BFGS for LR, Bagging for RF, etc.) |
| **A_DL** | Mini-batch SGD + Adam + Backpropagation + Dropout + Early stopping |
| **Generalisation** | Eout ≤ Ein + Ω(H,N); stratified k-fold CV via PyCaret; train/val gap monitored for MLP |

---

## Project Structure

### Documentation

```text
specs/001-crash-severity-model/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
└── tasks.md             ← Phase 2 output (/speckit.tasks)
```

### Source Code

```text
notebooks/
├── 02_preprocessing.ipynb     ← feature selection, encoding, leakage audit
├── 03_ml_pycaret.ipynb        ← PyCaret setup() + compare_models() + tune best
├── 04_dl_pytorch.ipynb        ← PyTorch shallow MLP with full SLT controls
└── 05_evaluation.ipynb        ← ML vs DL comparison, enrichment analysis,
                                  feature importance

models/
├── preprocessing_pipeline.joblib   ← fit on train, used by DL notebook
├── best_ml_model.pkl                ← PyCaret save_model() output
└── mlp_model.pth                    ← PyTorch state_dict

mlruns/                              ← MLflow tracking store (auto-created)
mlflow.db                            ← SQLite backend (optional)
```

---

## Phase 0: Research

### Pre-crash Feature List

Features confirmed as observable before or at the moment of the crash:

| Group | Columns |
|---|---|
| Temporal | HOUR, DAYOFWEEK, MONTH, YEAR |
| Weather | WEATHER, SURFCOND, LIGHTING |
| Road geometry | SPEEDLIMIT, RDNUMLANES, RDWIDTH, ROUTECLASS, TRUNKLINE, RDSUBTYPE |
| Driver demographics | DRIVER1AGE, DRIVER1SEX, DRIVER2AGE, DRIVER2SEX |
| Vehicle | VEH1TYPE, VEH1USE, VEH2TYPE, VEH2USE |
| Crash circumstance | CRASHTYPE, TRAFCTLDEV, NONTRAFFIC |

**Excluded (post-crash leakage)**: NUMOFINJ, NUMOFKILL, GRTINJSEVE, V1DAMAGE, V2DAMAGE, V1HARMEVT*, V2HARMEVT*, V1VIOLATOR, V2VIOLATOR, D1INJURY, D2INJURY, NOATYPEINJ, NOBTYPEINJ, NOCTYPEINJ

### Target Encoding

```
CRASHSEVER → binary label
  "Property Damage Only"   → 0  (no-injury)
  "Injury" + "Fatal"       → 1  (injury+fatal)
```

Class weights: **w₀ = 0.61, w₁ = 2.74** (N / (2 × class_count))

### Missing Value Strategy

| Feature type | Strategy |
|---|---|
| Numeric sentinel 999 (driver age) | Replace with NaN → median imputation |
| Numeric missing | Median imputation |
| Categorical missing | Most-frequent imputation |

### MLP Architecture (Sample Complexity Constrained)

N=74,309 is insufficient for deep architectures — MLP capped at 3 layers:

```
Input layer:   d features (~50–80)
Hidden layer 1: 128 units → BatchNorm → ReLU → Dropout(0.3)
Hidden layer 2:  64 units → BatchNorm → ReLU → Dropout(0.3)
Output layer:    1 unit   → Sigmoid
```

Optimiser: Adam (lr=1e-3, weight_decay=1e-4)
Early stopping: patience=10 epochs monitoring validation loss

---

## Phase 1: Design

### Data Model

**CrashRecord**
- Input: raw row from CGR_Crash_Data.csv
- Pre-crash feature subset selected per FR-010
- Sentinel values (999) recoded to NaN before pipeline

**SeverityLabel**
- Derived from CRASHSEVER
- Binary: 0 = Property Damage Only, 1 = Injury + Fatal
- Class distribution: 81.8% / 18.2%

**PreprocessingPipeline**
- Fit exclusively on training split (FR-001)
- Serialised alongside every model artifact
- Shared between ML and DL notebooks for consistency

**TrainedModel**
- ML: PyCaret model bundle (pipeline + estimator) saved via `save_model()`
- DL: PyTorch `state_dict` + separate preprocessing pipeline joblib

**ExperimentLog**
- JSON entries via `experiment_tracker.py`
- Fields: model name, hyperparameters, macro_f1, weighted_f1, recall_class1, auc_roc

### Notebook Contracts

**02_preprocessing.ipynb — outputs:**
```python
X_train, X_test, y_train, y_test   # numpy arrays, stratified 80/20 split
pipeline                            # fitted sklearn Pipeline → joblib
feature_names                       # list of post-encoding column names
```

**03_ml_pycaret.ipynb — key calls:**
```python
import mlflow
from pycaret.classification import *

mlflow.set_experiment("crash-severity-ml")
mlflow.sklearn.autolog()             # auto-captures params, metrics, model

exp = setup(
    data          = train_df,
    target        = "SEVERITY_BINARY",
    train_size    = 0.8,
    fold          = 5,
    fold_strategy = "stratifiedkfold",
    class_weights = {0: 0.61, 1: 2.74},
    metric        = "F1",
    session_id    = 42,
    log_experiment= True,            # PyCaret → MLflow native integration
    experiment_name="crash-severity-ml",
)

best_models = compare_models(n_select=3, sort="F1")
tuned       = tune_model(best_models[0])

# Log Ein/Eout explicitly for every top model
for model in best_models:
    with mlflow.start_run(run_name=type(model).__name__, nested=True):
        ein  = f1_score(y_train, model.predict(X_train), average="macro")
        eout = f1_score(y_test,  model.predict(X_test),  average="macro")
        mlflow.log_metrics({
            "ein_macro_f1":   ein,
            "eout_macro_f1":  eout,
            "generalisation_gap": eout - ein,   # negative = overfit
        })

save_model(tuned, "models/best_ml_model")
```

**04_dl_pytorch.ipynb — key components:**
```python
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

mlflow.set_experiment("crash-severity-dl")

# Loss function — weighted binary cross-entropy
# pos_weight = w1/w0 = 2.74/0.61 ≈ 4.49, but we pass w1 directly as pos_weight
# BCEWithLogitsLoss combines Sigmoid + BCE in one numerically stable op
pos_weight = torch.tensor([2.74])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# For evaluation: plain BCE (no pos_weight) gives interpretable loss values
eval_criterion = nn.BCEWithLogitsLoss()

with mlflow.start_run(run_name="shallow-mlp"):
    mlflow.log_params({
        "architecture": "128-64-1",
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "pos_weight": 2.74,
        "loss_fn": "BCEWithLogitsLoss",
        "early_stopping_patience": 10,
    })

    for epoch in range(MAX_EPOCHS):
        train_one_epoch(model, optimizer, train_loader, criterion)

        # Ein — in-sample cross-entropy loss + macro F1 (train set)
        ein_loss, ein_f1 = evaluate(model, X_train, y_train, eval_criterion)
        # Eout — out-of-sample cross-entropy loss + macro F1 (val set)
        eout_loss, eout_f1 = evaluate(model, X_val, y_val, eval_criterion)

        mlflow.log_metrics({
            "ein_loss":  ein_loss,   "ein_f1":  ein_f1,
            "eout_loss": eout_loss,  "eout_f1": eout_f1,
            "gap_f1":    eout_f1 - ein_f1,   # divergence = overfitting onset
        }, step=epoch)

        if early_stopping(eout_loss):
            break

    # Final test-set metrics
    test_f1, test_recall = evaluate(model, X_test, y_test)
    mlflow.log_metrics({"test_macro_f1": test_f1, "test_recall_class1": test_recall})
    mlflow.pytorch.log_model(model, "mlp_model",
                             registered_model_name="crash-severity-mlp")
```

**05_evaluation.ipynb — MLflow comparison query:**
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Pull best run from each experiment ranked by eout_macro_f1
ml_runs = client.search_runs("crash-severity-ml",
                              order_by=["metrics.eout_macro_f1 DESC"],
                              max_results=1)
dl_runs = client.search_runs("crash-severity-dl",
                              order_by=["metrics.test_macro_f1 DESC"],
                              max_results=1)

# Enrichment analysis: predicted 0 vs 1 → mean NUMOFINJ (SC-007)
# Feature importance from best ML model artifact
# Confusion matrix, AUC-ROC plots logged as mlflow.log_artifact()
```

### Implementation Sequence

| Step | Notebook | Tool | Outputs |
|---|---|---|---|
| 1 | `02_preprocessing.ipynb` | pandas + sklearn | pipeline.joblib, train/test arrays |
| 2 | `03_ml_pycaret.ipynb` | PyCaret | best_ml_model.pkl, experiment log entries |
| 3 | `04_dl_pytorch.ipynb` | PyTorch | mlp_model.pth, experiment log entry |
| 4 | `05_evaluation.ipynb` | pandas + matplotlib | comparison table, feature importance, enrichment analysis |

Each step is independently executable. Steps 2–4 depend on Step 1 output. Step 4 depends on all prior steps.

### Quickstart

```bash
# Install dependencies (already done)
uv sync

# Run notebooks in order
uv run jupyter notebook notebooks/02_preprocessing.ipynb
uv run jupyter notebook notebooks/03_ml_pycaret.ipynb
uv run jupyter notebook notebooks/04_dl_pytorch.ipynb
uv run jupyter notebook notebooks/05_evaluation.ipynb

# Launch MLflow UI to compare all runs (Ein vs Eout, learning curves)
uv run mlflow ui
# Open http://localhost:5000
```
