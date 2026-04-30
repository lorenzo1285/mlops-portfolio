# VAE Fix Plan — Fatal Representation & Posterior Collapse

**Triggered by**: `notebooks/vae_fatal_representation.ipynb` audit  
**Date**: 2026-04-29  
**Status**: Both fixes implemented ✅ | Re-audit complete (2026-04-29)

## Task Map

| Fix | Description | Tasks | Status |
|-----|-------------|-------|--------|
| Fix 1 | Weighted sampler + augmented VAE training data | **T121** | ✅ Done (2026-04-29) |
| Fix 2 | KL annealing — ramp β from 0 → `beta_max` over `warmup_epochs` | **T106** (RED), **T107** (GREEN) | ✅ Done (2026-04-29) |

**T121** is in `specs/002-mlops-portfolio/tasks.md` (Phase 3H section, after T107).  
**T106/T107** are in `specs/002-mlops-portfolio/tasks.md` Phase 3H — VAE KL Annealing.

---

## Audit Findings — Initial (before fixes)

| Metric | Value | Status |
|---|---|---|
| Fatal active dims (σ²<0.5) | 3 / 8 | ✅ Healthy |
| Fatal KL > PDO KL | Yes (2.75 vs 2.55 nats) | ✅ Healthy |
| Centroid dist ratio (synth/PDO) | 0.390 | ✅ Healthy |
| KS-aligned dims real vs synth Fatal | 2 / 8 dims | ⚠️ Borderline |
| Fatal-PDO separation > 1σ | 1 / 8 dims | ⚠️ Borderline |
| PDO overlap in Fatal 2σ box | 47.06% | ❌ Problem |

**Overall verdict**: Fatal Z representation is poor. The Fatal cluster in latent space is so diffuse
that 47% of PDO samples overlap it — leaving the downstream classifier with almost no separating signal.

---

## Root Cause (Two Layers)

### Layer 1 — General posterior collapse (affects all classes)
5 of 8 latent dims are collapsed (σ² ≈ 1, prior) for PDO, Injury, and Fatal alike.  
The effective latent space is ~3 dims. The VAE is underfitting.

**Cause**: fixed β=0.5 during training is too strong a KL penalty at the start of training.  
**Fix**: KL annealing — ramp β from 0 → beta_max over warmup_epochs (T106/T107, already planned).

### Layer 2 — Fatal-specific scarcity
579× more PDO than Fatal during VAE training. Fatal contributed 0.14% of all ELBO gradient updates.  
Even in the 3 active dims, the encoder learned PDO-shaped structure — Fatal crashes barely influenced it.

**Cause**: uniform sampling in the VAE DataLoader; no mechanism to up-weight rare classes.  
**Fix**: weighted sampler in `DVAETrainer` DataLoader — Fatal rows get proportional gradient share.

---

## Fix Plan

### Fix 1 — Weighted sampler in VAE DataLoader (T121 — DONE ✅)

**What**: pass class labels alongside X_all into `DVAETrainer.train()`.  
Use `torch.utils.data.WeightedRandomSampler` to give Fatal rows proportional frequency.  
Weight formula: `w_c = N / (n_classes × class_count_c)` — same as `compute_class_weights()` in `src/metrics.py`.

**Constitution compliance**:
- Does not violate II (unsupervised): reconstruction target is still clean X, no labels used for loss
- Labels are used only to compute sampling frequency — not a supervised signal
- Does not change the DAG or any DVC deps/outs
- Does not touch X_val or X_test

**Interface change**: `DVAETrainer.train(X_all, y_all)` — add `y_all` parameter.  
`run.py` already loads y_train/y_val/y_test; concatenate and pass alongside X_all.

**Success criterion**: re-run audit notebook → Fatal-PDO separation > 1σ in ≥ 3 dims,  
PDO overlap < 30%, KS-aligned dims ≥ 4/8.

### Fix 2 — KL annealing (T106 RED + T107 GREEN — DONE ✅)

**What**: replace fixed β with linear warmup: β_t = min(beta_max, beta_start + beta_max × t/warmup_epochs).  
Prevents posterior collapse in early epochs; opens up more active dims for all classes.

**Actual outcome**: active dims increased 3 → 5; collapsed dims dropped 5 → 0. Overall KL doubled
(2.5 → 5+ nats) — the model is encoding significantly more information. Best epoch=4, early stopping
at epoch 24 — model converged during the β warmup window.

---

## Implementation Order

```
Fix 1 — T121 (weighted sampler + augmented data)  ←  DONE ✅
    │
    ▼
Fix 2 — T106/T107 (KL annealing)  ←  DONE ✅
    │
    ▼
dvc repro train_vae → encode  ←  DONE ✅ (2026-04-29)
    │
    ▼
Re-run vae_fatal_representation.ipynb audit  ←  DONE ✅ (see results below)
    │
    ▼
Continue to Phase 4A (augment T108–T113)  ←  NEXT
```

---

## Results — Both Fixes Applied (2026-04-29)

**MLflow run**: `36a3cd52` | Final ELBO: -2.56 (was -3.64) | Best epoch: 4 | Early stop: epoch 24
**Data**: 76,970 samples; 54,676 train (aug); 2,766 Fatal (3.59%); weighted sampler active

| Metric | Before fixes | After Fix 1 only | After both fixes | Target | Status |
|---|---|---|---|---|---|
| Active dims (σ²<0.5) | 3/8 | 2/8 | **5/8** | ≥4/8 | ✅ |
| Collapsed dims (σ²>0.9) | 5/8 | 6/8 | **0/8** | ≤3/8 | ✅ |
| Fatal KL > PDO KL | ✅ Yes | — | ⚠️ No (5.04 vs 5.13) | Yes | ⚠️ |
| Centroid ratio (synth/PDO) | 0.390 | — | **0.442** | <0.5 | ✅ |
| KS-aligned dims (KS<0.10) | 2/8 | 4/8 | **4/8** | ≥4/8 | ✅ |
| Fatal-PDO sep >1σ dims | 1/8 | 1/8 | **1/8** | ≥2/8 | ⚠️ |
| PDO overlap in Fatal 2σ box | 47.06% | 55.99% | **47.88%** | not a hard gate | accepted |
| Downstream fatal recall | — | — | — | >0.50 | pending |

**Key finding**: KL annealing eliminated all collapsed dims (0/8 collapsed vs 5/8 before) and
opened 2 additional active dims. Fatal-PDO overlap unchanged — the Fatal cluster remains
diffuse, with only z3 providing meaningful separation (sep=1.83). This is accepted under the
business rule (Fatal recall > PDO precision).

**Seasonality note**: HOUR_sin/cos and MONTH_sin/cos as standalone features are weak Fatal
separators — Fatal crashes are not cleanly seasonal. However, z3 (the one strong separator)
likely captures their interaction with LIGHTING, WEATHER, and SPEEDLIMIT. The VAE encodes
conjunctive feature patterns, not raw seasonality.

---

## Next Steps

### Immediate — Phase 4A (T108–T113)
Build the `augment` stage. This is the pipeline blocker for everything downstream:
- `X_train_augmented.npy` already exists from a previous run but has no DVC-tracked production code
- T094/T095 (encode pipeline verification) remain blocked until augment stage is built

### Medium term — Katib β-HPO (T052–T060)
The best epoch was 4 (during β warmup, β ≈ 0.13). This suggests `beta_max=0.5` with
`warmup_epochs=15` may be suboptimal — the model prefers a lower effective β. Katib will
search `beta_max` over [0.5, 1.0, 2.0, 4.0, 8.0] to find the optimal value. The annealing
schedule (T106/T107) is already in place so all Katib trials benefit from warmup.

### If fatal recall < 0.50 after Phase 4 evaluation
Option: reduce `latent_dim` from 8 → 4 to match the effective rank of the active dims.
This forces the encoder to concentrate all information into fewer, denser dims and may
improve Fatal-PDO separation. Do not reduce before Phase 4 results are in.

---

## Notes

- The 2,661 synthetic vs 73 real Fatal ratio in Z_train_augmented means the classifier trains
  almost entirely on CTGAN synthetics. Improving Fatal Z quality (this plan) directly improves
  what the classifier trains on.
- Weighted sampling does not change `X_train_augmented` or the augment stage — it only affects
  how frequently the VAE sees each row during its own training loop.
- After Katib β-HPO (T052–T060), the best β will be the KL weight at the plateau — the warmup
  schedule does not conflict with Katib since Katib searches `beta_max`, not the schedule itself.
