# Train/Val/Test Split Strategy Analysis

**Date:** 2026-05-02  
**Context:** Phase O3.5 threshold tuning revealed validation set limitations

## Current State

### Distribution
```
Total: 74,309 crashes (after dropna)
Fatal: 105 crashes (0.141% of dataset)

Split (70/15/15):
- Train: 52,015 samples (73 Fatal = 0.140%)
- Val:   11,147 samples (16 Fatal = 0.143%)  ← PROBLEM
- Test:  11,147 samples (16 Fatal = 0.143%)
```

### Current Method
- **Stratified train_test_split** (sklearn)
- Two-stage split:
  1. Split off test (15%)
  2. Split train/val from remainder (70/15 relative)
- `stratify=y` maintains class proportions
- `random_state=42` for reproducibility

## The Problem

### Issue 1: Recall Quantization
With **16 Fatal rows in validation**:
- Recall is discretized: 0/16, 1/16, 2/16, ..., 16/16
- Step size: **6.25%** (1/16)
- Observed maximum: **5/16 = 31.25%**
- Constitutional gate: **≥50% (8/16 = 50%)**
- **Gap: Cannot observe recalls between 31.25% and 37.5%**

### Issue 2: Unreliable Metrics
- **Standard error of recall** ∝ 1/√n → SE ∝ 1/4 = 25%
- Threshold scan showed:
  - Val recall peaked at 31.25% (5/16 Fatal identified)
  - Test recall at τ=0.15: **62.5%** (10/16 Fatal identified)
  - **31% gap** suggests val set is not representative

### Issue 3: Constitutional Gate Violation
Constitution Principle VI requires:
- Fatal recall > 0.50 ✅ (on test, using τ=0.15)
- Macro F1 > 0.35 ✅ (on test, using τ=0.23)

**But validation cannot verify this!** Max val recall = 31.25% < 50%.

## Recommended Sampling Techniques

### Option 1: Stratified K-Fold Cross-Validation ⭐ **RECOMMENDED**

**Method:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # Train model, compute metrics
```

**Benefits:**
- Each fold: ~84 Fatal rows (train) + ~21 Fatal rows (val)
- Val recall step: 1/21 = **4.76%** (vs 6.25% currently)
- Can observe 10/21 = 47.6% recall (closer to 50% gate)
- Average metrics across 5 folds → **more stable estimates**
- Uses 100% of data (vs 70% currently)
- Industry standard for imbalanced datasets

**Costs:**
- Requires pipeline refactor: train 5 models instead of 1
- Longer training time (~5x)
- More complex DVC stage dependency graph
- Katib HPO needs modification (outer CV loop)

**Fatal Distribution per Fold (5-fold):**
```
Fold 1: train=84, val=21
Fold 2: train=84, val=21
Fold 3: train=84, val=21
Fold 4: train=84, val=21
Fold 5: train=84, val=21
```

---

### Option 2: Adjusted Split Ratios

**Method:**
```python
# Change from 70/15/15 to 60/20/20
train_size: 0.60
val_size: 0.20
test_size: 0.20
```

**Benefits:**
- Simpler: no pipeline refactor needed
- Each set gets ~21 Fatal rows (60% = 63, 20% = 21, 20% = 21)
- Val recall step: 1/21 = **4.76%**
- Can observe 10/21 = 47.6% or 11/21 = 52.4% recall

**Costs:**
- Reduces training data by 10% (52k → 44.5k samples)
- Lower sample complexity ratio (may violate Constitution IV)
- Still single-fold risk (no averaging)

**Fatal Distribution (60/20/20):**
```
Train: 44,586 samples (63 Fatal = 0.141%)
Val:   14,862 samples (21 Fatal = 0.141%)
Test:  14,861 samples (21 Fatal = 0.141%)
```

---

### Option 3: Repeated Stratified K-Fold

**Method:**
```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
# 15 total train/val splits
```

**Benefits:**
- Same as K-Fold but with **variance reduction**
- 15 models → robust CI on metrics
- Can detect if single-fold results are outliers
- Gold standard for model selection

**Costs:**
- 15x training time
- Significant pipeline complexity
- MLflow experiment explosion (15 runs per HPO trial)

---

### Option 4: Nested Cross-Validation

**Method:**
```python
# Outer loop: model evaluation (5-fold)
# Inner loop: HPO on each outer fold (3-fold)
# Total: 5 × 3 = 15 HPO experiments
```

**Benefits:**
- Prevents optimistic bias in HPO
- True generalization estimate
- Best practice for model selection with rare classes

**Costs:**
- 15x compute cost for HPO
- Very complex implementation
- Overkill for portfolio demonstration

---

### Option 5: Manual Stratification with Minimum Counts

**Method:**
```python
def stratified_split_min_samples(X, y, test_size, min_minority=20):
    # Custom logic: ensure each split has ≥20 Fatal rows
    # May require iterative resampling or ratio adjustment
```

**Benefits:**
- Guarantees minimum Fatal count per split
- Flexible control over rare class representation

**Costs:**
- Custom code (not sklearn standard)
- May violate exact train/val/test ratio
- Harder to reproduce across datasets

---

## Recommendation Matrix

| Technique | Fatal/Val | Recall Step | Pipeline Impact | Training Time | Recommendation |
|-----------|-----------|-------------|-----------------|---------------|----------------|
| Current (70/15/15) | 16 | 6.25% | None | 1x | ❌ Insufficient |
| **5-Fold CV** | **21** | **4.76%** | **High** | **5x** | **⭐ BEST** |
| 60/20/20 split | 21 | 4.76% | Low | 1x | ✅ Quick fix |
| Repeated 5-Fold | 21 | 4.76% | Very High | 15x | ⚠️ Overkill |
| Nested CV | 21 | 4.76% | Very High | 15x | ⚠️ Overkill |
| Manual strat | 20-30 | 3.33-5% | Medium | 1x | ⚠️ Non-standard |

---

## Implementation Plan

### Path A: Quick Fix (60/20/20 Split) — **1-2 hours**

**Steps:**
1. Update `params.yaml`:
   ```yaml
   data:
     train_size: 0.60
     val_size: 0.20
     test_size: 0.20
   ```
2. Run `dvc repro featurize` (invalidates all downstream)
3. Run threshold scan again with 21 Fatal val rows
4. Check if 10/21 or 11/21 recall satisfies gate

**Pros:** Minimal code change, fast iteration  
**Cons:** Still single-fold risk, reduced training data

---

### Path B: Full K-Fold CV — **8-16 hours** ⭐

**Steps:**
1. **Constitution amendment**: Allow K-fold as exception to 3-way split (Principle II)
2. **Refactor featurize stage**:
   - Add `split_strategy: simple | kfold` param
   - When `kfold`: output `folds/fold_{i}/X_train.npy` etc.
3. **Refactor train_ml/train_dl**:
   - Loop over folds
   - Train one model per fold
   - Log per-fold metrics to MLflow
   - Average metrics across folds for final report
4. **Refactor evaluate**:
   - Aggregate fold results
   - Report mean ± std for each metric
5. **Update Katib HPO**:
   - Each trial runs full 5-fold CV
   - Fitness = mean(val_macro_f1) across folds
6. **Update tests**:
   - Mock fold structure
   - Verify averaging logic

**Pros:** Industry best practice, robust metrics, uses all data  
**Cons:** High implementation complexity, longer training time

---

## Decision Criteria

Choose **Path A (60/20/20)** if:
- Need quick validation of threshold approach
- Want minimal pipeline disruption
- Portfolio timeline is tight

Choose **Path B (K-Fold CV)** if:
- Want production-grade solution
- Demonstrating MLOps best practices is priority
- Have time for proper architecture refactor
- Planning to showcase in portfolio/resume

---

## Next Steps

**User Decision Required:**
1. Which path to take? (A: quick fix, B: full refactor)
2. If Path A: proceed with 60/20/20 split
3. If Path B: start with Constitution amendment + spec update

**Immediate Action (Path A):**
```bash
# Update params.yaml
# Run pipeline
uv run dvc repro featurize
uv run python scripts/tune_threshold.py
uv run dvc repro train_ml evaluate
```

**Immediate Action (Path B):**
1. Update `.specify/memory/constitution.md` Principle II
2. Create `specs/002-mlops-portfolio/003-kfold-cv/spec.md`
3. Run `/speckit.plan` workflow
