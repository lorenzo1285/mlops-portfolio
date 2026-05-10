# Source Code (`src/`)

This directory contains the core implementation logic for the Road Traffic Accidents Severity Classification project. The pipeline is modular, with each stage having its own submodule and a shared set of utilities.

## Directory Structure

| Module | Description |
| :--- | :--- |
| `augment/` | Synthetic data generation using CTGAN to address class imbalance. |
| `encode/` | Transformation of raw features into VAE latent space (Z-space). |
| `evaluate/` | A/B testing and statistical validation of model performance. |
| `featurize/` | Data preprocessing, encoding, and engineering. |
| `ingest/` | Raw data loading and initial cleaning. |
| `register/` | Model versioning and MLflow Model Registry interaction. |
| `train_dl/` | Deep Learning pipeline (Shallow MLP) using PyTorch. |
| `train_ml/` | Machine Learning pipeline (XGBoost) using Scikit-Learn. |
| `train_vae/` | Variational Autoencoder training for dimensionality reduction. |
| `validate/` | Data contract validation using Great Expectations. |

## Shared Utilities

### `metrics.py`
Contains shared evaluation helpers:
- `compute_class_weights`: Calculates inverse frequency weights for cost-sensitive learning.
- `per_class_matrix`: Generates precision, recall, and F1 scores per class.
- `focal_loss_grad_hess`: Custom objective function for XGBoost to support focal loss.

### `losses.py`
Contains PyTorch loss function implementations:
- `BalancedFocalLoss`: Implements Focal Loss with class-specific alpha weighting to focus on hard samples and minority classes (Fatal).

### `config.py`
Centralized configuration management using `ProjectConfig` dataclasses, loading parameters from `params.yaml`.

---

## Latest Model Results (Phase 4B)

Results captured as of April 2026. The pipeline targets three classes: **PDO** (Property Damage Only), **Injury**, and **Fatal**.

### Variational Autoencoder (VAE)
| Metric | Value |
| :--- | :--- |
| ELBO | -2.532 |
| Reconstruction Loss | 1.659 |
| KL Loss | 1.746 |
| Latent Dim | 8 |

### MLP (Shallow Neural Network)
Summary across 10 random seeds on the test set (`eout`):

| Metric | Mean | Best |
| :--- | :--- | :--- |
| `macro_f1` | 0.3251 | **0.3341** |
| `fatal_recall` | 0.5500 | 0.5625 |
| `generalisation_gap` | 0.1182 | — |

**Per-Class Breakdown (Best Seed):**
| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| PDO | 0.853 | 0.640 | 0.731 | 9,115 |
| Injury | 0.235 | 0.294 | 0.261 | 2,016 |
| **Fatal** | **0.005** | **0.562** | **0.010** | **16** |

---

## Performance Gates

The project defines strict performance gates in the `evaluate` stage (Constitution VI):
1. **Macro F1 Gate**: Must be `> 0.35`. (Current: `0.334` ❌)
2. **Fatal Recall Gate**: Must be `> 0.50`. (Current: `0.563` ✅)

The **Fatal** class scarcity (only 16 rows in test) is identified as the primary bottleneck for macro F1 performance.
