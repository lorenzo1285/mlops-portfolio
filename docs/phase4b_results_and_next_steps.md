# Phase 4B Results & Next Steps
*Captured: 2026-04-29 — Branch: 002-mlops-portfolio*

---

## Pipeline state at capture

Stages complete (DVC-tracked, artifacts on disk):

| Stage | Status |
|---|---|
| validate | ✅ |
| ingest | ✅ |
| featurize | ✅ |
| train_vae | ✅ |
| augment | ✅ |
| encode | ✅ |
| train_dl | ✅ `models/mlp_model.pth` written |
| train_ml | ❌ not run |
| evaluate | ❌ blocked on train_ml |
| tune / register | ❌ not run |

---

## Artifact inventory

| Artifact | Shape | Notes |
|---|---|---|
| `X_train_augmented.npy` | (54 676, 28) | CTGAN Fatal rows included |
| `y_train_augmented.npy` | (54 676,) | PDO 77.8 % · Injury 17.2 % · Fatal 5.0 % |
| `X_val.npy` | (11 147, 28) | Never augmented |
| `X_test.npy` | (11 147, 28) | Never augmented |
| `y_val.npy` | (11 147,) | PDO 81.8 % · Injury 18.1 % · **Fatal 0.14 % (16 rows)** |
| `y_test.npy` | (11 147,) | PDO 81.8 % · Injury 18.1 % · **Fatal 0.14 % (16 rows)** |
| `Z_train_augmented.npy` | (54 676, 8) | μ-path, deterministic |
| `Z_val.npy` | (11 147, 8) | |
| `Z_test.npy` | (11 147, 8) | |
| `models/mlp_model.pth` | — | Best seed by eout_macro_f1 |

---

## VAE results (`crash-severity-vae`, latest run)

| Metric | Value |
|---|---|
| ELBO (final epoch) | −2.532 |
| Reconstruction loss | 1.659 |
| KL loss | 1.746 |
| kl_beta (final) | 0.500 |
| latent_dim | 8 |
| epochs | 200 |

The KL loss is nearly as large as the reconstruction loss, suggesting the encoder
is being pushed hard against the prior and the posterior may be close to N(0,1)
for all inputs — a sign the 8-dim bottleneck is tight relative to 28 input features.

---

## MLP results (`crash-severity-dl`, 10 seeds)

| Metric | Mean | Best | Worst |
|---|---|---|---|
| `eout_macro_f1` | 0.3251 | **0.3341** | 0.3166 |
| `eout_fatal_recall` | 0.5500 | 0.5625 | 0.5000 |
| `ein_macro_f1` | 0.4434 | — | — |
| `generalisation_gap` | 0.1182 | — | 0.1277 |
| `best_val_loss` | 0.9078 | 0.8903 | 0.9211 |

Random-baseline CrossEntropy on 3 classes ≈ ln(3) = **1.099**.
Best val loss 0.890 is only ~0.21 nats below random — limited signal.

### Per-class breakdown (best seed, test set)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PDO | 0.853 | 0.640 | 0.731 | 9 115 |
| Injury | 0.235 | 0.294 | 0.261 | 2 016 |
| **Fatal** | **0.005** | **0.562** | **0.010** | **16** |

Fatal precision of 0.005 means the model predicts "Fatal" on ~1 800 rows to catch
9 of the 16 actual Fatal cases. The macro F1 of 0.334 is the average of
0.731 + 0.261 + 0.010 — almost entirely dragged down by the Fatal F1 of 0.010.

A majority-class classifier (always predict PDO) scores **81.8 % accuracy**.
The MLP achieves roughly 60–70 % accuracy — worse than the trivial baseline.
Accuracy is the wrong metric here; it highlights why macro F1 was chosen.

### Constitution VI gate assessment

| Gate | Threshold | Best value | Pass? |
|---|---|---|---|
| `eout_macro_f1` | > 0.35 | 0.334 | ❌ |
| `eout_fatal_recall` | > 0.50 | 0.563 | ✅ |

The macro F1 gate will fail at the `evaluate` stage as currently implemented.
All 10 seeds are below 0.35; this is not a seed-variance issue.

---

## Root cause analysis

Three compounding problems identified — listed by severity.

### Problem 1 — Fatal class data scarcity (primary)

The raw dataset contains only **~106 Fatal rows** out of 74 309 (0.14 %).
With a stratified 70/15/15 split (already confirmed — `train_test_split(..., stratify=y)`
is in `featurizer.py`), this distributes as:

| Split | Total rows | Fatal rows |
|---|---|---|
| Train | ~52 016 | ~74 |
| Val | ~11 147 | **16** |
| Test | ~11 147 | **16** |

16 Fatal rows in val/test is not a splitting strategy problem — it is a **data
availability problem**. No split strategy (including cross-validation) meaningfully
changes this; 5-fold CV would give ~21 Fatal rows per eval fold, not enough for
reliable class-level metrics and at 5× the compute cost.

### Problem 2 — Training/inference distribution mismatch (secondary)

CTGAN augmentation pushed the training Fatal proportion to **5 %**.
Val and test remain at the **true 0.14 %**. The model is calibrated to predict
Fatal ~5 % of the time, so at inference it floods ~1 800 rows with Fatal
predictions — of which 1 791 are wrong. This is why Fatal precision = 0.005.

The model is not confused; it learned the wrong prior.

### Problem 3 — Z-space bottleneck (tertiary)

All 10 seeds converge to the same narrow val loss band (0.890–0.921). The VAE
KL loss (1.746) equals the reconstruction loss (1.659), meaning the encoder is
heavily regularised toward N(0,1) regardless of input. Class-discriminative
structure in the 28 original features is likely erased by the tight 8-dim
bottleneck + high β penalty.

---

## Why cross-validation is not the fix

Cross-validation was considered and ruled out for three reasons:

1. **Data scarcity is the root cause.** 5-fold CV gives ~21 Fatal eval rows
   instead of 16 — not a meaningful improvement for Fatal-class metrics.

2. **Pipeline incompatibility.** The VAE trains unsupervised on X_all for 200
   epochs. Correct CV would require retraining the VAE on each fold's training
   set, multiplying compute by K. CTGAN augmentation must also run inside each
   fold. The DVC DAG is built around single numpy array artifacts.

3. **Constitution conflict.** Principle II defines a fixed 3-way split. CV
   would require an amendment and a production deployment strategy for the
   final model (CV produces K models, not one).

---

## Is it worth training XGBoost next?

**Yes — run it**, for three reasons:

1. **Required for the A/B test.** `evaluate` (T046–T051) runs Welch's t-test
   on the macro F1 distributions of XGBoost vs MLP. Both need to exist.

2. **XGBoost may outperform MLP on this Z-space.** Tree-based models partition
   the 8-dim latent cube directly without needing to learn linear embeddings.
   On low-dimensional structured data, XGBoost typically extracts more signal.

3. **Diagnostic value.** If XGBoost also plateaus around 0.33–0.34 macro F1,
   the Z-space bottleneck (Problem 3) is confirmed. If it scores materially
   higher (≥ 0.38), the MLP config is the constraint, not the representation.

---

## Recommendations to fix the problems

### Fix 1 — Threshold calibration (quick win, no retraining)

The model's predicted probabilities are calibrated to a 5 % Fatal prior.
Shift the decision threshold for Fatal downward at inference time so that
only the highest-confidence Fatal predictions are kept.

```python
# In evaluate/run.py — instead of argmax:
probs = model.predict_proba(Z_test)          # shape (N, 3)
threshold = 0.30                              # tune on Z_val
y_pred = np.where(probs[:, 2] >= threshold, 2, probs[:, :2].argmax(axis=1))
```

Expected effect: Fatal precision rises from 0.005 toward 0.3–0.5;
recall drops slightly; overall macro F1 improves.

### Fix 2 — Increase `latent_dim` (medium effort, re-run VAE)

Change `params.yaml`: `vae.latent_dim: 8` → `16`.
The 28→8 compression is aggressive. At 28 features, 16 dims preserves
more structure while still regularising. Downstream stages (encode,
train_ml, train_dl) all read `latent_dim` from config — no code changes needed.

DVC will invalidate `train_vae → encode → train_ml → train_dl → evaluate`.

### Fix 3 — Lower β (re-run VAE, pairs with Fix 2)

Change `params.yaml`: `vae.beta_max: 0.5` → `0.1` or `0.2`.
Lower β reduces the KL penalty, allowing the encoder to retain more
discriminative structure at the cost of less regularisation. This is
exactly what the Katib HPO stage (T052+) searches over — running it
manually first lets you validate the direction before committing to a
full HPO sweep.

### Fix 4 — Collect or surface more Fatal data (highest impact, slowest)

106 Fatal rows is a hard ceiling on evaluation reliability. Options:
- Request additional years of crash data from the same source
- Merge with a complementary crash dataset (NHTSA FARS for fatalities)
- Accept the limitation and document it: report Fatal metrics with an
  explicit low-support warning rather than treating 16 test rows as
  statistically significant

### Fix 5 — Acknowledge the gate threshold given data reality

The constitution VI gate `eout_macro_f1 > 0.35` was set without knowing
the true Fatal prevalence (0.14 %). With 16 Fatal test rows, the variance
on Fatal F1 is ±25 %. Consider amending the gate to:
- `eout_macro_f1 > 0.30` (relaxed, reflects data reality), or
- Add a `min_fatal_support` guard: skip the gate if Fatal test support < 50

---

## Recommended execution order

### Phase A — complete the pipeline skeleton (no fixes yet)

1. **T035–T037** — XGBoost `train_ml` (RED → GREEN → `dvc repro train_ml`)
2. **T046–T051** — `evaluate` A/B test — expected to fail macro F1 gate;
   confirms diagnosis and closes the pipeline loop end-to-end

### Phase B — fix the representation

3. Apply **Fix 2** (`latent_dim: 16`) + **Fix 3** (`beta_max: 0.1`)
   together — re-run `dvc repro train_vae encode train_ml train_dl evaluate`
4. Apply **Fix 1** (threshold calibration) in `evaluate` stage
5. Re-check constitution VI gates

### Phase C — HPO and productionisation

6. **T052–T060** — Katib β-HPO (after representation is stable)
7. **T061–T065** — `register` stage
8. **T066–T070** — Docker + Kubernetes packaging
