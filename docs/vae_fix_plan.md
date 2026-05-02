# VAE Fix Plan — Fatal Recall

**Status as of 2026-05-02**: gates still failing (`fatal_recall = 0.25`, gate = 0.50)  
**Best params from Katib**: `vae.beta_max=0.2`, `vae.latent_dim=32`

---

## History (summarised)

| Fix | Description | Result |
|-----|-------------|--------|
| KL annealing (T106/T107) | Ramp β 0 → beta_max over warmup_epochs | Collapsed dims 5→0, active dims 3→5. ✅ |
| Weighted sampler (T121) | Fatal rows up-weighted in VAE DataLoader | VAE sees Fatal proportionally. ✅ |
| Katib HPO (T052–T060) | Search beta_max ∈ [0.05–1.0], latent_dim ∈ [8,16,32] | F1 +0.009. Fatal recall unchanged at 0.25. ❌ |

**VAE audit after KL annealing + weighted sampler** (2026-04-29, MLflow run `36a3cd52`):

| Metric | Before | After | Target | Status |
|---|---|---|---|---|
| Active dims (σ²<0.5) | 3/8 | 5/8 | ≥4/8 | ✅ |
| Collapsed dims (σ²>0.9) | 5/8 | 0/8 | ≤3/8 | ✅ |
| Fatal-PDO sep >1σ dims | 1/8 | 1/8 | ≥2/8 | ⚠️ |
| PDO overlap in Fatal 2σ box | 47.06% | 47.88% | — | accepted |
| Downstream fatal recall | — | 0.25 | >0.50 | ❌ |

---

## Katib HPO Results (2026-05-02)

9/15 trials (skopt crashed on duplicate categorical suggestions; aborted at `maxFailedTrialCount`).  
Every trial scored `val_fitness = val_macro_f1 × 0.5` — the 0.5× fatal recall penalty applied universally.

| beta_max | latent_dim | val_fitness | implied F1 | fatal_recall |
|---|---|---|---|---|
| **0.2** | **32** | **0.1861** | **0.3723** | 0.25 |
| 0.05 | 16 | 0.1857 | 0.3713 | 0.25 |
| 0.2 | 8  | 0.1824 | 0.3648 | 0.25 |
| 0.2 | 16 | 0.1821 | 0.3642 | 0.25 |
| 0.5 | 16 | 0.1812 | 0.3624 | 0.25 |
| 0.1 | 8  | 0.1766 | 0.3533 | 0.25 |

Pre-Katib baseline: F1=0.364, fatal_recall=0.25. Net delta: +0.008 F1, 0 recall.

---

## Deep Diagnosis

### Finding 1 — 16 val fatal rows; recall is quantised in steps of 0.0625

```
val:  PDO 9,114 (81.76%) | Injury 2,017 (18.09%) | Fatal 16 (0.14%)
```

Recall = {0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, …, 0.50, …}.  
All 9 trials got exactly **4/16** val fatal correct. The same 4 are likely unambiguous fatals;
the remaining 12 sit in the Injury/PDO confusion zone. Going from 0.25 → 0.50 requires 4 more
correct predictions on those 12 hard rows. This is not a smooth gradient that HPO can move.

### Finding 2 — 34.8× train/val fatal prior mismatch

```
train_aug fatal: 5.00%  (2,734 / 54,676)
val fatal:       0.14%     (16 / 11,147)
mismatch:       34.8×
```

XGBoost learns a decision boundary calibrated for 5% fatal. At inference (0.14% fatal), the
argmax threshold is miscalibrated — the model needs P(Fatal|X) > 50% to predict Fatal, which
rarely happens on the hard rows. Class weights (`w_fatal = 6.67`) correct the loss but not
the inference threshold.

### Finding 3 — 97.3% of training fatals are CTGAN synthetics

```
real Fatal: 73 rows (2.7%)  |  CTGAN synthetic: 2,661 rows (97.3%)
```

CTGAN learns from 73 real fatal rows. The 16 val fatals are held-out samples from the same pool.
Synthetics capture average marginal patterns but miss the specific multivariate signatures of the
12 hard val fatals. Increasing `target_fatal_ratio` generates more synthetics but does not add new
information — it deepens the 97% synthetic dependence.

### Finding 4 — beta_max / latent_dim are the wrong levers

Neither parameter changes the prior mismatch, adds real fatal rows, or shifts the inference
threshold. HPO exhausted its search space without moving fatal recall because the root cause
is upstream of the VAE hyperparameters entirely.

---

## Data Centric Approach

<!-- fixes that change what the model trains on -->

---

## Model Centric Approach

<!-- fixes that change how the model makes predictions -->

---

## Notes

- **Katib skopt crash**: crash-loops after exhausting diversity in a small categorical space.
  Switch `algorithmName: bayesianoptimization` → `random` in `k8s/katib/vae_experiment.yaml`
  for future runs.
- Fix C (Bayesian prior correction) reduces Fatal probability by 35.7× — counterproductive for
  recall; only useful for calibration audits.
- `target_fatal_ratio` already updated 0.05 → 0.15 in `params.yaml` (queued for T060).
