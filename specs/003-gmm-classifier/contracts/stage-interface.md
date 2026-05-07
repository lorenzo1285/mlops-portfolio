# Stage Interface Contract: train_gmm

**Branch**: `003-gmm-classifier` | **Date**: 2026-05-06

## train_gmm Stage

### Inputs (DVC deps)

| File | Source Stage | Description |
|------|-------------|-------------|
| `data/processed/Z_train_augmented.npy` | encode | Latent vectors for augmented training set (shape: [N_aug, 8]) |
| `data/processed/y_train_augmented.npy` | encode | Labels for augmented training set (shape: [N_aug]) |
| `data/processed/Z_val.npy` | encode | Latent vectors for validation set (shape: [N_val, 8]) |
| `data/processed/y_val.npy` | featurize | Validation labels (shape: [N_val]) |
| `data/processed/Z_test.npy` | encode | Latent vectors for test set (shape: [N_test, 8]) |
| `data/processed/y_test.npy` | featurize | Test labels (shape: [N_test]) |
| `src/train_gmm/run.py` | — | Entry point |
| `src/train_gmm/trainer.py` | — | Business logic |
| `src/config.py` | — | Config loader |

### Outputs (DVC outs)

| File | Description |
|------|-------------|
| `models/best_gmm_model.pkl` | Pickled `GMMClassifier` wrapper (best seed by val macro F1) |

### Params (dvc.yaml params)

```yaml
- gmm.n_components
- gmm.covariance_type
- gmm.reg_covar
- gmm.max_iter
- gmm.fatal_prior_boost
- gmm.experiment_name
- model.n_classes
- model.macro_f1_threshold
- model.fatal_recall_threshold
- mlflow.experiment_name_gmm
- mlflow.tracking_uri
- ab_test.seeds
```

### Environment Variables (run.py reads)

| Var | Default | Description |
|-----|---------|-------------|
| `PARAMS_PATH` | `params.yaml` | Path to params file |
| `MLFLOW_TRACKING_URI` | from params | Overrides params if set |

### MLflow Experiment

- **Experiment name**: `crash-severity-gmm` (from `gmm.experiment_name`)
- **Run naming**: `gmm_seed_{seed}` per seed
- **Mandatory metrics per run** (logged to MLflow):
  - `ein_macro_f1` — training set macro F1
  - `eout_macro_f1` — validation set macro F1
  - `generalisation_gap` — eout_macro_f1 − ein_macro_f1
  - `eout_fatal_recall` — validation set fatal class recall
- **Best-seed metric**: `eout_macro_f1` (validation set — never test)

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | At least one seed completed and best_gmm_model.pkl written |
| 1 | All seeds failed to converge or other fatal error |

---

## evaluate Stage — Interface Changes

### Additional Input (DVC dep)

| File | Description |
|------|-------------|
| `models/best_gmm_model.pkl` | Presence confirms train_gmm completed |

### Additional Params

```yaml
- mlflow.experiment_name_gmm
```

### Output Schema Change

`docs/evaluation_report.json` gains GMM fields and per-pair p-values (see data-model.md). The `winner` and `gates_passed` keys are unchanged in position and semantics.

---

## register Stage — Interface Changes

### Conditional Classifier Path

When `winner == "gmm"`, `run.py` must pass `models/best_gmm_model.pkl` as `classifier_path` to `ModelRegistrar.register()`. When `winner == "ml"`, passes `models/best_ml_model.pkl`. When `winner == "dl"`, passes `models/mlp_model.pth`.

### CrashSeverityPyfunc Dispatch

| winner value | load_context behaviour | predict behaviour |
|-------------|----------------------|------------------|
| `"ml"` | pickle load | `self._classifier.predict(Z)` |
| `"gmm"` | pickle load (same as ml) | `self._classifier.predict(Z)` |
| `"dl"` | torch load ShallowMLP | argmax of logits |

The `winner="gmm"` path is absorbed into the existing pickle branch with no separate dispatch required, as long as `GMMClassifier.predict(Z)` matches the sklearn predict interface.
