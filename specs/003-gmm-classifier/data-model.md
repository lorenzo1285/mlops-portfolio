# Data Model: GMM Classifier — Third Parallel Training Branch

**Branch**: `003-gmm-classifier` | **Date**: 2026-05-06

## New Entities

### GMMConfig (src/config.py)

Typed dataclass for GMM hyperparameters. Added to `ProjectConfig`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_components` | `dict[int, int]` | `{0: 1, 1: 1, 2: 2}` | Gaussians per class label (PDO=0, Injury=1, Fatal=2); Fatal defaults to 2 to capture potential bimodal Z-distribution |
| `covariance_type` | `str` | `"full"` | sklearn GaussianMixture covariance type |
| `reg_covar` | `float` | `1e-6` | Regularization added to diagonal of covariance |
| `max_iter` | `int` | `100` | Max EM iterations |
| `n_init` | `int` | `5` | Number of EM random initialisations per class; best log-likelihood is kept; eliminates degenerate component solutions when n_components > 1 |
| `fatal_prior_boost` | `float` | `1.0` | Scalar multiplier on log-prior of Fatal class at prediction time; values > 1.0 increase Fatal recall (analogous to `fatal_threshold` in XGBoost, `focal_loss_gamma` in MLP) |
| `experiment_name` | `str` | `"crash-severity-gmm"` | MLflow experiment name |

**params.yaml section**:
```yaml
gmm:
  n_components:
    0: 1   # PDO — unimodal in latent space
    1: 1   # Injury — unimodal in latent space
    2: 2   # Fatal — two Gaussians; captures bimodal Z-distribution after CTGAN augmentation
  covariance_type: "full"
  reg_covar: 1.0e-6
  max_iter: 100
  n_init: 5
  fatal_prior_boost: 1.0
  experiment_name: "crash-severity-gmm"
```

---

### GMMClassifier (src/train_gmm/trainer.py — internal)

Thin wrapper around per-class `GaussianMixture` instances. Exposes `predict(Z)` compatible with the `self._classifier.predict(Z)` call in `CrashSeverityPyfunc`. This is the object that gets pickled to `models/best_gmm_model.pkl`.

| Attribute | Description |
|-----------|-------------|
| `_gmms` | `dict[int, GaussianMixture]` — one fitted GMM per class label |
| `_log_priors` | `np.ndarray` — log P(class) computed from training class distribution; Fatal entry multiplied by `fatal_prior_boost` before argmax |
| `predict(Z)` | Returns `np.ndarray` of integer class labels (0=Fatal, 1=Injury, 2=PDO) |

---

### GMMTrainResult (src/train_gmm/trainer.py)

Structured result returned by `GMMTrainer.train()`.

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | MLflow run ID of best-seed run |
| `model_path` | `str` | Path to `models/best_gmm_model.pkl` |
| `best_seed` | `int` | Seed that produced the best validation macro F1 |
| `eout_macro_f1` | `float` | Validation macro F1 of best seed |
| `eout_fatal_recall` | `float` | Validation fatal recall of best seed |

---

## Modified Entities

### MLflowConfig (src/config.py) — MODIFIED

Added field:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiment_name_gmm` | `str` | `"crash-severity-gmm"` | MLflow experiment for GMM seeds |

**params.yaml addition** (under `mlflow:`):
```yaml
experiment_name_gmm: "crash-severity-gmm"
```

---

### ABTestConfig (src/config.py) — MODIFIED

| Field | Old Type | New Type | Notes |
|-------|----------|----------|-------|
| `tiebreak` | `str` | `list[str]` | Priority-ordered list; first element wins when all pairwise tests are non-significant |

**params.yaml change**:
```yaml
# Before
ab_test:
  tiebreak: "ml"

# After
ab_test:
  tiebreak: ["ml", "dl", "gmm"]
```

---

### EvaluationResult (src/evaluate/evaluator.py) — MODIFIED

Added fields (GMM metrics + per-pair statistics):

| Field | Type | Notes |
|-------|------|-------|
| `gmm_mean_f1` | `float` | Mean eout_macro_f1 across GMM seeds |
| `gmm_ci_low` | `float` | 95% CI lower bound |
| `gmm_ci_high` | `float` | 95% CI upper bound |
| `gmm_mean_fatal_recall` | `float` | Mean eout_fatal_recall across GMM seeds |
| `p_value_ml_dl` | `float` | Replaces old `p_value` |
| `p_value_ml_gmm` | `float` | Welch's t-test, Bonferroni-corrected |
| `p_value_dl_gmm` | `float` | Welch's t-test, Bonferroni-corrected |
| `cohens_d_ml_dl` | `float` | Replaces old `cohens_d` |
| `cohens_d_ml_gmm` | `float` | |
| `cohens_d_dl_gmm` | `float` | |

Removed fields: `p_value` (renamed), `cohens_d` (renamed).

**evaluation_report.json schema** (additions; `winner` and `gates_passed` keys unchanged):
```json
{
  "winner": "gmm",
  "gates_passed": true,
  "gmm_mean_f1": 0.42,
  "gmm_mean_fatal_recall": 0.61,
  "p_value_ml_dl": 0.18,
  "p_value_ml_gmm": 0.03,
  "p_value_dl_gmm": 0.09,
  ...
}
```

---

## Artifact Registry

| Artifact | Path | Stage | Tracked By |
|----------|------|-------|------------|
| GMM model (best seed) | `models/best_gmm_model.pkl` | `train_gmm` | DVC |
| Updated evaluation report | `docs/evaluation_report.json` | `evaluate` | DVC |
| Updated A/B/C comparison | `docs/ab_test_comparison.json` | `evaluate` | DVC |
| Registry receipt | `models/registry_receipt.json` | `register` | DVC |

---

## State Transitions

```
encode outputs (Z_train_augmented, Z_val, Z_test, y_*)
    │
    ├─── train_ml  ────────────────────────────────┐
    ├─── train_dl  ────────────────────────────────┤
    └─── train_gmm (NEW) ─── best_gmm_model.pkl ──┤
                                                    │
                                               evaluate
                                         (3-way pairwise t-tests)
                                                    │
                                    ┌───────────────┴───────────────┐
                                gates_passed=true             gates_passed=false
                                    │                               │
                                 register                        halt (non-zero exit)
                                    │
                          crash-severity@champion
```
