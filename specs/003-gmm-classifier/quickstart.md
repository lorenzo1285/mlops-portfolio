# Quickstart: GMM Classifier — Third Parallel Training Branch

**Branch**: `003-gmm-classifier` | **Date**: 2026-05-06

## Prerequisites

Pipeline must have completed through the `encode` stage:
```powershell
uv run dvc repro encode
```

## Run train_gmm only

```powershell
uv run dvc repro train_gmm
```

Expected output:
- `models/best_gmm_model.pkl` written
- MLflow runs appear in `crash-severity-gmm` experiment (10 seeds)

## Run 3-way evaluation

```powershell
uv run dvc repro evaluate
```

Expected output:
- `docs/evaluation_report.json` — includes `winner` (one of: ml, dl, gmm) and `gates_passed`
- `docs/ab_test_comparison.json` — 3-way pairwise statistics

## Run full pipeline from encode onward

```powershell
uv run dvc repro train_ml train_dl train_gmm evaluate register
```

Note: `train_ml`, `train_dl`, and `train_gmm` run in parallel (no inter-stage deps); DVC schedules them accordingly.

## View MLflow results

```powershell
uv run mlflow ui
```

Navigate to `http://localhost:5000` → experiment `crash-severity-gmm` to see per-seed metrics.

## Run tests

```powershell
uv run python -m pytest tests/test_train_gmm.py -v
uv run python -m pytest tests/test_evaluator.py -v
uv run python -m pytest tests/test_register.py -v
```

All tests require real pipeline artifacts (`data/processed/Z_train_augmented.npy`, etc.) to exist. Run `dvc repro encode` first.

## Params reference

GMM parameters are in `params.yaml` under the `gmm:` key:

```yaml
gmm:
  n_components:
    0: 1                   # PDO — unimodal
    1: 1                   # Injury — unimodal
    2: 2                   # Fatal — two Gaussians for potential bimodal Z-distribution
  covariance_type: "full"  # full covariance matrix per class
  reg_covar: 1.0e-6        # diagonal regularization
  max_iter: 100            # max EM iterations
  n_init: 5               # random EM restarts per class; best kept (eliminates degenerate components)
  fatal_prior_boost: 1.0   # >1.0 boosts Fatal recall (same role as fatal_threshold/focal_loss_gamma)
  experiment_name: "crash-severity-gmm"
```

Tiebreak order (3-way) is configured under `ab_test:`:

```yaml
ab_test:
  tiebreak: ["ml", "dl", "gmm"]
```
