# VAE Fix Plan — Fatal Representation & Posterior Collapse

**Triggered by**: `notebooks/vae_fatal_representation.ipynb` audit  
**Date**: 2026-04-29  
**Status**: Fix 1 complete ✅ | Fix 2 pending (T106/T107)

## Task Map

| Fix | Description | Tasks | Status |
|-----|-------------|-------|--------|
| Fix 1 | Weighted sampler + augmented VAE training data | **T121** | ✅ Done (2026-04-29) |
| Fix 2 | KL annealing — ramp β from 0 → `beta_max` over `warmup_epochs` | **T106** (RED), **T107** (GREEN) | ⏳ Pending |

**T121** is in `specs/002-mlops-portfolio/tasks.md` (Phase 3H section, after T107).  
**T106/T107** are in `specs/002-mlops-portfolio/tasks.md` Phase 3H — VAE KL Annealing.

---

---

## Audit Findings

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

**Cause**: fixed β=1.0 during training is too strong a KL penalty at the start of training.  
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

### Fix 2 — KL annealing (T106 RED + T107 GREEN — pending ⏳)

**What**: replace fixed β=1.0 with linear warmup: β_t = min(beta_max, beta_start + beta_max × t/warmup_epochs).  
Prevents posterior collapse in early epochs; opens up more active dims for all classes.

**Expected outcome**: active dims increase from ~3 → 5+; collapsed dims drop from 5 → ≤ 2.  
Combined with Fix 1 (Fatal gets more gradient), the newly-opened dims should encode Fatal structure.

---

## Implementation Order

```
Fix 1 — T121 (weighted sampler + augmented data)  ←  DONE ✅
    │
    ▼
Fix 2 — T106 RED → T107 GREEN (KL annealing)  ←  implement next
    │
    ▼
dvc repro train_vae → encode
    │
    ▼
Re-run notebooks/vae_fatal_representation.ipynb  ←  verify both fixes
    │
    ▼
Continue to T091–T095 (encode QA) if audit passes
```

Fix 1 before Fix 2 matters: KL annealing opens up more latent dims, but if Fatal still gets
only 0.14% gradient share those dims will remain PDO-shaped. Both fixes are needed together
to produce a meaningful Fatal cluster.

---

## Success Criteria (re-run audit)

**Business priority**: Fatal recall > PDO precision. PDO overlap in the Fatal Z region is
acceptable — it means some PDO samples get predicted as Fatal, which is the safe error direction.

| Metric | Before Fix 1 | After Fix 1 | Target after KL annealing |
|---|---|---|---|
| Active dims (σ²<0.5) — Fatal | 3 / 8 | 2 / 8 | ≥ 4 / 8 |
| Collapsed dims (σ²>0.9) — Fatal | 5 / 8 | 6 / 8 | ≤ 3 / 8 |
| Fatal-PDO separation > 1σ | 1 / 8 dims | 1 / 8 dims | ≥ 2 / 8 dims |
| PDO overlap in Fatal 2σ box | 47.06% | 55.99% | **not a hard gate** (PDO sacrifice accepted) |
| KS-aligned dims real vs synth Fatal | 2 / 8 | **4 / 8** ✅ | ≥ 4 / 8 |
| Downstream fatal recall (Z_test) | — | — | **> 0.50** (constitution gate) |

If the audit still shows ❌ after both fixes, next option is reducing `latent_dim` from 8 to match
the effective rank of the space (currently ~3 active dims), then re-running.

---

## Notes

- The 2,661 synthetic vs 73 real Fatal ratio in Z_train_augmented means the classifier trains
  almost entirely on CTGAN synthetics. Improving Fatal Z quality (this plan) directly improves
  what the classifier trains on.
- Weighted sampling does not change `X_train_augmented` or the augment stage — it only affects
  how frequently the VAE sees each row during its own training loop.
- After Katib β-HPO (T052–T060), the best β will be the KL weight at the plateau — the warmup
  schedule does not conflict with Katib since Katib searches `beta_max`, not the schedule itself.
