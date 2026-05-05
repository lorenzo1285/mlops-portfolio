# VAE Fix Plan — Fatal Recall

**Status as of 2026-05-04**: val gates PASSING; test recall gate still failing  
**Active params**: `vae.beta_max=0.2`, `vae.latent_dim=32`, `model.fatal_threshold=0.17`, `augment.target_fatal_ratio=0.05`  
**Active HPO engine**: Optuna (replaced Katib 2026-05-04 — see Phase O3.6)

### Current metrics (XGBoost, τ=0.17, 60/20/20 split, MLflow run `2f652b5d`)

| Set | Macro F1 | Fatal Recall | Gate |
|-----|----------|--------------|------|
| Val | 0.3501 | 0.5238 (11/21) | ✅ PASS |
| Test | 0.3481 | 0.3333 (7/21) | ❌ FAIL |

A/B evaluate (mean over 10 seeds): XGBoost F1=0.3514, recall=0.3919 — test recall gate not met.  
Gap remaining: 4 more correct Fatal predictions needed out of 21 test Fatals (33.3% → 52.4%).

---

## History (summarised)

| Fix | Description | Result |
|-----|-------------|--------|
| KL annealing (T106/T107) | Ramp β 0 → beta_max over warmup_epochs | Collapsed dims 5→0, active dims 3→5. ✅ |
| Weighted sampler (T121) | Fatal rows up-weighted in VAE DataLoader | VAE sees Fatal proportionally. ✅ |
| Katib HPO (T052–T060) | Search beta_max ∈ [0.05–1.0], latent_dim ∈ [8,16,32] | F1 +0.009. Fatal recall unchanged at 0.25. ❌ |
| **Phase 1** | | |
| Fix B — target_fatal_ratio 0.05→0.15 | Triple synthetic Fatal in augment stage | 7,794 synthetics generated; Fatal fraction 0.14%→15.0% in train_aug. ✅ |
| Val set repair — 60/20/20 split | Changed from 70/15/15 to increase val Fatal rows 16→21 | Recall quantization step 6.25%→4.76%; gate now observable at 11/21=52.4%. ✅ |
| dl.input_dim bug fix | Corrected `dl.input_dim: 8→32` in params.yaml | MLP now trains on correct 32-dim Z-space. ✅ |
| Fix A — threshold τ=0.17 | Grid-scanned τ∈[0.15,0.50] on Z_val; τ=0.17 is first dual-pass point | Val: F1=0.3501 ✅, recall=0.5238 ✅. Test recall=0.3333 ❌. |
| DVC param tracking | Added `model.fatal_threshold` to dvc.yaml train_ml params | DVC now correctly invalidates train_ml when τ changes. ✅ |
| **Phase 2 (2026-05-04)** | | |
| Fix B revert — target_fatal_ratio 0.15→0.05 | Prior mismatch grew to 107× (train 15% vs real 0.14%); no recall lift observed; synthetics from 63 real rows add volume not information | Reverted. ✅ |
| Katib → Optuna (Phase O3.6) | Replaced Katib as active HPO engine; expanded search to 5 params + pruning | In progress — T126–T132. |

**VAE audit after KL annealing + weighted sampler** (2026-04-29, MLflow run `36a3cd52`):

| Metric | Before | After | Target | Status |
|---|---|---|---|---|
| Active dims (σ²<0.5) | 3/8 | 5/8 | ≥4/8 | ✅ |
| Collapsed dims (σ²>0.9) | 5/8 | 0/8 | ≤3/8 | ✅ |
| Fatal-PDO sep >1σ dims | 1/8 | 1/8 | ≥2/8 | ⚠️ |
| PDO overlap in Fatal 2σ box | 47.06% | 47.88% | — | accepted |
| Downstream fatal recall | — | 0.25 | >0.50 | ❌ |

---

## Katib HPO Results (2026-05-02) — Historical Reference Only

> **Note**: Katib replaced by Optuna as active HPO engine (2026-05-04, Phase O3.6). Expanded search space: 5 continuous/categorical params + MedianPruner. See research.md Decision 17.

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

### Finding 1 — Val fatal rows; recall quantisation ✅ RESOLVED (Phase 1)

**Before (70/15/15):**
```
val:  PDO 9,114 (81.76%) | Injury 2,017 (18.09%) | Fatal 16 (0.14%)
Recall step = 6.25% — gate (≥50%) unreachable; max observed = 4/16 = 25%
```

**After (60/20/20):**
```
val:  PDO 12,170 (81.88%) | Injury 2,671 (17.97%) | Fatal 21 (0.14%)
Recall step = 4.76% — gate observable at 11/21 = 52.4%
```

With τ=0.17, XGBoost now achieves **11/21 = 52.4%** val recall, clearing the gate. The
remaining gap on test (7/21 = 33.3%) points to distribution shift between val and test on the
hard Fatal rows — not a quantization artefact.

### Finding 2 — Train/val fatal prior mismatch (updated, partially mitigated)

**Before (target_fatal_ratio=0.05):**
```
train_aug fatal: 5.00%  (2,734 / 54,676)
val fatal:       0.14%     (16 / 11,147)
mismatch:       34.8×
```

**After (target_fatal_ratio=0.15, 60/20/20):**
```
train_aug fatal: 15.0%  (7,857 / 52,379)
val fatal:        0.14%    (21 / 14,862)
mismatch:       107×
```

Fix B tripled the synthetic fatal count, but the mismatch actually grew because the denominator
(real val data) is still 0.14% fatal. The threshold correction τ=0.17 partially compensates at
the decision boundary, but the gap between training distribution (15% fatal) and inference
distribution (0.14% fatal) remains the core calibration problem.

### Finding 3 — 97.3% of training fatals are CTGAN synthetics

```
real Fatal: 73 rows (2.7%)  |  CTGAN synthetic: 2,661 rows (97.3%)
```

CTGAN learns from 63 real fatal rows (60/20/20 split). The 21 val fatals are held-out samples
from the same pool. Synthetics capture average marginal patterns but miss the specific multivariate
signatures of the hard val fatals. At target_fatal_ratio=0.15, real fatals are 63/7,857+63 = 0.8%
of augmented fatal rows — 99.2% synthetic. Increasing the ratio adds volume but not new information.

### Finding 4 — beta_max / latent_dim are the wrong levers

Neither parameter changes the prior mismatch, adds real fatal rows, or shifts the inference
threshold. HPO exhausted its search space without moving fatal recall because the root cause
is upstream of the VAE hyperparameters entirely.

---

## Data Centric Approach

Fixes that change what the model trains on or how features are represented.

- **Cyclical KL Annealing (Upgrade)**: Transition from monotonic to cyclical annealing (Fu et al., 2019). By periodically resetting $\beta$ to 0, we "re-open" the latent bottleneck multiple times. This forces the encoder to escape local minima and find more discriminative patterns for the Fatal class, increasing Mutual Information between $X$ and $Z$.
- **Fix B — Increase `target_fatal_ratio` (0.05 → 0.15)**: Triple the fatal proportion in training data (~8,940 synthetics). Higher fatal training pressure to shift the boundary.
- **Generative Refinement (TVAE/CTGAN)**:
    - **Conditional Sampling**: Instead of training on isolated Fatal rows, train on the full dataset using CTGAN's conditional matrix. This allows the model to learn fatal offsets relative to the global crash manifold.
    - **Latent-Space Cleaning**: Apply a $k$-Nearest Neighbor filter to synthetics. Delete any synthetic Fatal sample whose neighborhood is dominated by real PDO/Injury samples to prevent "boundary noise" from confusing the classifier.
- **Tomek Link Cleaning**: Use `imblearn.combine.SMOTETomek` to identify and remove ambiguous synthetic samples that overlap with PDO/Injury clusters. This "clears the space" in latent space around the rare fatal signatures.
- **Hybrid Sampling Strategy**: Combine Generative Oversampling (TVAE) with Targeted Undersampling (Tomek Links). While TVAE provides the volume needed for gradient updates, Tomek Links sharpen the decision boundary by surgically removing PDO samples that "encroach" on the Fatal manifold. This creates a clear margin without the information loss associated with random undersampling.
- **Library Selection — `imbalanced-learn`**: We will utilize the `imbalanced-learn` (imblearn) library as the primary engine for boundary sharpening.
    - **1. Sharp Decision Boundaries (Tomek Links)**: We will use `imblearn.under_sampling.TomekLinks`. It finds pairs of samples that are nearest neighbors but different classes. By removing the majority class (PDO) from these pairs, we "clean" the boundary. Application: apply to the Latent Space ($Z$) before training the classifier.
    - **2. Hybrid Cleaning (SMOTE-Tomek)**: Even though we use TVAE for generation, we can use `imblearn.combine.SMOTETomek`'s cleaning logic. We use the TomekLinks part of this library to clean up the TVAE synthetics that were generated in "noisy" locations.
    - **3. Pipeline Integration**: `imblearn` provides a specialized `Pipeline` object. Why: Standard `sklearn` pipelines don't handle samplers that change the number of rows. Action: Use the `imblearn` pipeline in `src/train_ml/trainer.py` to ensure Tomek cleaning only happens during training and not during inference.
- **Danger Index Feature Engineering**: Create interaction features and "Danger Labels" before the VAE to provide explicit conjunctive signals that are currently being "blurred" by the latent bottleneck.
    - **Solo High-Speed Label**: `Label = 1 if (NUMOFVEHIC == 1 AND SPEEDLIMIT >= 45) else 0`. (Addresses the -26.4% change in vehicle count for fatals).
    - **Survival Ratio (Soft Label)**: `Score = 1 - (NUMOFUNINJ / Total_Occupants)`. (Addresses the -68.6% drop in uninjured persons in fatals).
    - **Vulnerability Interaction**: `Label = 1 if (DRIVER1AGE < 25 OR DRIVER1AGE > 70) AND (SPEEDLIMIT > 40)`.
    - **⚠️ Leakage Warning**: Drop `NUMOFKILL` (98.3% correlation) and similar outcome-based features before training.

---

## Model Centric Approach

Fixes that change how the model learns or makes predictions.

- **Balanced Focal Loss (α + γ)**: Combine cost-sensitive weights ($\alpha$) with difficulty-based focusing ($\gamma$). By adding a $(1 - p_t)^\gamma$ modulating factor, we down-weight the gradient contribution from the 9,000+ "easy" PDO samples and force the optimizer to focus on the 12 "hard" Fatal rows that the model currently misses.
- **Fix A — Fatal Decision Threshold Optimization**: Replace the hard `argmax` decision rule with a tuned probability threshold $\tau \in [0.05, 0.50]$ on `Z_val`. Pick $\tau^*$ that satisfies the 0.50 recall gate.
- **Fix D — Supervised Latent Loss**: Add a cross-entropy branch to the VAE ELBO ($L_{total} = L_{rec} + \beta L_{KL} + \gamma L_{CE}$). This forces the encoder to preserve discriminative class structure in the latent space.

---

## Technical Implementation Strategy

### 1. Balanced Focal Loss
We will implement a unified loss function in `src/metrics.py`:
```python
# Balanced Focal Loss = alpha * (1 - p)**gamma * CrossEntropy
```
This handles both class imbalance (via $\alpha$) and sample difficulty (via $\gamma$).

### 2. Custom XGBoost Objective
For `MLTrainer`, we will implement the first and second derivatives of the Balanced Focal Loss to allow XGBoost to optimize for "hard" fatal rows directly.
- **Fix D — Supervised Latent Loss**: Add a cross-entropy branch to the VAE ELBO ($L_{total} = L_{rec} + \beta L_{KL} + \gamma L_{CE}$). This forces the encoder to preserve discriminative class structure in the latent space.

---

## Notes

- **Katib skopt crash**: crash-loops after exhausting diversity in a small categorical space.
  Switch `algorithmName: bayesianoptimization` → `random` in `k8s/katib/vae_experiment.yaml`
  for future runs.
- Fix C (Bayesian prior correction) reduces Fatal probability by 35.7× — counterproductive for
  recall; only useful for calibration audits.
- `target_fatal_ratio` already updated 0.05 → 0.15 in `params.yaml` (queued for T060).

---

## Priority Matrix — Complexity vs. Impact

**Impact** = expected lift on `eout_fatal_recall` toward the 0.50 gate.  
**Complexity** = estimated implementation effort (code changes, retraining cost, risk of regression).

### Summary Table

| Fix | Category | Complexity | Impact | Rationale |
|-----|----------|-----------|--------|-----------|
| ~~**Fix A — Threshold Optimization**~~ ✅ | Model | Low | High | **DONE** — τ=0.17; val gates pass. Test recall=0.3333. |
| ~~**Fix B — target_fatal_ratio 0.15**~~ ↩ reverted | Data | Low | Low | **REVERTED 2026-05-04** — prior mismatch grew to 107×; no recall lift; back to 0.05. |
| **Optuna HPO** (replaces Katib) | Model | Low | High | **IN PROGRESS** — T126–T132; 5-param search + pruning; no pod overhead. See Decision 17. |
| **MLP Balanced Focal Loss (α + γ)** | Model | Low | Medium | **QUEUED** — T133a–d; drop-in replacement for `CrossEntropyLoss` in `DLTrainer`; no upstream cascade. |
| **Cyclical KL Annealing** | Model | Medium | Medium | **QUEUED** — T135a–d; upgrade to existing KL mechanism; may increase active dims beyond 5/8. |
| **Danger Index Feature Engineering** | Data | Medium | High | **QUEUED** — T123a–e; vulnerability interaction (DRIVER1AGE×SPEEDLIMIT) implementable now; NUMOFVEHIC/NUMOFUNINJ need leakage audit. |
| **Tomek Link Cleaning** | Data | Low | Medium | ⛔ **BLOCKED** — Constitution III violation: 4th imbalance mechanism. Requires amendment before implementation. |
| **XGBoost Focal Loss (custom obj)** | Model | High | High | **QUEUED** — T125a–d; last resort; requires manual grad/hess derivation. |
| **Fix D — Supervised Latent Loss** | Model | High | High | ⛔ **BLOCKED** — Constitution II conflict: CE branch requires labels; VAE trains on X_all unsupervised. Requires amendment (T136) before implementation (T137a–d). |
| **CTGAN Conditional Sampling** | Data | High | Medium | Deferred — bounded by 63 real Fatal rows regardless of sampling strategy. |
| **Latent-Space Cleaning (kNN filter)** | Data | Medium | Medium | Deferred — lower priority than cyclical KL and danger index. |

### 3×3 Grid

```
            │  Low Impact        │  Medium Impact               │  High Impact
────────────┼────────────────────┼──────────────────────────────┼──────────────────────────────
Low         │  Fix B             │  Tomek Link Cleaning         │  Fix A (Threshold Opt.)  ★
Complexity  │  (target_fatal     │                              │
            │   ratio ↑)        │                              │
────────────┼────────────────────┼──────────────────────────────┼──────────────────────────────
Medium      │                    │  Cyclical KL Annealing       │  Danger Index Features   ★
Complexity  │                    │  Hybrid Sampling             │  Balanced Focal Loss     ★
            │                    │  Latent-Space Cleaning       │
────────────┼────────────────────┼──────────────────────────────┼──────────────────────────────
High        │                    │  CTGAN Conditional Sampling  │  Fix D (Supervised       ★
Complexity  │                    │                              │   Latent Loss)
            │                    │                              │  Custom XGBoost Obj.     ★
```

### Recommended Execution Order (updated 2026-05-04)

1. ~~**Fix A**~~ ✅ — τ=0.17; val gates pass; test recall=0.3333.
2. ~~**Fix B**~~ ↩ — reverted 0.15→0.05; no recall lift; prior mismatch worse at 107×.
3. **Optuna HPO** ← *active* — T126–T132; 5-param search (beta_max, latent_dim, warmup_epochs, lr, dropout_p) + MedianPruner; replaces Katib.
4. **MLP Balanced Focal Loss** — T133a–d; no upstream cascade; run alongside or after Optuna.
5. **Cyclical KL Annealing** — T135a–d; triggers VAE retrain; piggyback on Optuna-forced retrain.
6. **Danger Index Features** — T123a–e; 6-stage cascade; vulnerability interaction feature implementable immediately.
7. **XGBoost Focal Loss** — T125a–d; last resort before Fix D.
8. **Fix D (Supervised Latent Loss)** — ⛔ blocked; draft constitution II amendment (T136) first; implement T137a–d only after amendment accepted.
