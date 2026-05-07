# Research: GMM Classifier — Third Parallel Training Branch

**Branch**: `003-gmm-classifier` | **Date**: 2026-05-06

## Decision 1: Statistical Test for 3-Way Comparison

**Decision**: Pairwise Welch's t-tests with Bonferroni correction (3 pairs: ml↔dl, ml↔gmm, dl↔gmm).

**Rationale**: The existing `ABEvaluator` already uses Welch's t-test and the infrastructure (scipy.stats.ttest_ind) is in place. Extending to airwise tests reuses the same mechanism with a corrected alpha (α/3). One-way ANOVA would also work but requires a post-hoc test anyway (Tukey's HSD), adding complexity without benefit at n=10 seeds. Bonferroni is conservative but appropriate for a 3-comparison family.

**Alternatives considered**:
- One-way ANOVA + Tukey HSD — valid but requires `statsmodels`; adds a dependency for marginal gain at 3 groups
- Kruskal-Wallis (non-parametric) — appropriate for small n but unnecessary given Welch's t-test already assumes unequal variance; F1 distributions are approximately normal across seeds

**Winner selection from 3-way pairwise result**: Find the classifier with the highest mean `eout_macro_f1`. If it is statistically significantly better (p < α/3) than at least one other, it wins. If no classifier is significantly better than all others (i.e., all pairwise comparisons p ≥ α/3), apply the tiebreak list.

---

## Decision 2: GMM Prediction Strategy

**Decision**: One `GaussianMixture(n_components=1)` fitted per class on that class's Z_train_augmented rows. Prediction = `argmax_c [log P(Z | class=c) + log P(class=c)]` — i.e., the maximum a-posteriori class under equal or weighted priors.

**Rationale**: Per-class fitting is equivalent to Linear Discriminant Analysis when covariance_type="full" and priors are uniform, but more expressive when n_components > 1 (multi-modal class distributions). Sklearn's `GaussianMixture.score_samples()` returns log-likelihood; adding log-prior gives MAP estimate. This is the standard discriminative use of GMM.

**Alternatives considered**:
- Single GMM with n_components=3 (unsupervised) — doesn't guarantee class alignment; requires label-to-component mapping step
- sklearn `BayesianGaussianMixture` — automatic component count but slower and less reproducible across seeds

**Class weights for GMM**: Class weights are not passed as `sample_weight` to GMM (sklearn's GaussianMixture does not accept it). Instead, unequal class priors are handled by computing log-prior probabilities from the training class distribution and adding them to the log-likelihoods at prediction time. This is constitutionally compliant: CTGAN augmentation already balances the Fatal class in Z_train_augmented; runtime prior adjustment accounts for any residual imbalance.

---

## Decision 3: tiebreak Configuration Shape

**Decision**: Change `ab_test.tiebreak` from `str` to `list[str]` in `params.yaml` and `ABTestConfig`. Default: `["ml", "dl", "gmm"]`. When all pairwise tests are non-significant, the first element of the tiebreak list that has a trained artifact wins.

**Rationale**: A single string cannot express a 3-way priority order. A list is the minimal extension and maintains backward semantic compatibility (first element is the default winner, same as the current `"ml"` string default).

**Migration**: `ABTestConfig.tiebreak: str → list[str]`. `params.yaml` entry changes from `tiebreak: "ml"` to `tiebreak: ["ml", "dl", "gmm"]`. The evaluate stage logic changes from `winner = self._ab_test_config.tiebreak` to iterating the list.

---

## Decision 4: EvaluationResult Structure

**Decision**: Extend `EvaluationResult` with GMM fields in-place. Rename `p_value` → `p_value_ml_dl`; add `p_value_ml_gmm`, `p_value_dl_gmm`, `cohens_d_ml_dl`, `cohens_d_ml_gmm`, `cohens_d_dl_gmm`. Add `gmm_mean_f1`, `gmm_ci_low`, `gmm_ci_high`, `gmm_mean_fatal_recall`.

**Rationale**: Extending the existing dataclass is simpler than creating a new `ABCEvaluationResult` type. The `evaluation_report.json` schema evolves (backward-incompatible for the old 2-field p_value), but `register/registrar.py` only reads `gates_passed` and `winner` — unaffected by the schema extension.

**Alternatives considered**:
- New `MultiEvaluationResult` class — cleaner semantically but requires touching more files (run.py, register/run.py, any JSON consumers)
- Keeping old field names and adding new ones — avoided because `p_value` would become ambiguous (which pair?)

---

## Decision 5: CrashSeverityPyfunc Extension for GMM

**Decision**: Add a third branch in `CrashSeverityPyfunc.load_context()` for `winner == "gmm"`: load the pkl file (same as XGBoost). Prediction path: same as ML branch — `self._classifier.predict(Z)` via a thin wrapper that calls the per-class GMM's MAP prediction.

**Rationale**: GMM persists as a pickle (sklearn model), identical to XGBoost. The only difference from the XGBoost path is the internal predict logic, which is encapsulated inside the pickled `GMMClassifier` wrapper class (see data-model.md). No changes to the artifact bundling or MLflow pyfunc signature.

**Note**: The `GMMClassifier` wrapper (not sklearn's `GaussianMixture` directly) is what gets pickled — it exposes a `predict(Z)` method compatible with the `self._classifier.predict(Z)` call site in `CrashSeverityPyfunc.predict()`. This means `winner == "gmm"` falls into the same `if winner == "ml"` branch logically, and the `load_context` dispatch simply loads pkl for both ml and gmm while loading `.pth` for dl.

---

## Decision 6: MLflowConfig Extension

**Decision**: Add `experiment_name_gmm: str` field to `MLflowConfig` dataclass and corresponding `mlflow.experiment_name_gmm` key to `params.yaml`. Default value: `"crash-severity-gmm"`.

**Rationale**: Follows exact same pattern as `experiment_name_ml` and `experiment_name_dl`. The registrar must resolve the GMM experiment to look up the champion run_id.

---

## Decision 7: fatal_prior_boost Tuning — Optuna HPO with Winner Dispatch

**Decision**: Add `fatal_prior_boost` to the Optuna search space AND fix `OptunaTuner._objective()` to dispatch on the current A/B/C winner — training `MLTrainer`, `DLTrainer`, or `GMMTrainer` depending on what `evaluation_report.json` reports as winner.

**Rationale**: The `OptunaTuner` currently hardcodes `MLTrainer` in every trial. This means VAE hyperparameters are always optimized for XGBoost fitness, even when DL or GMM is the champion — undermining the HPO rationale. Adding GMM to the search space makes this pre-existing gap visible and too costly to ignore. Winner-dispatch fixes it for all three classifiers simultaneously.

**Alternatives considered**:
- Train all three classifiers per trial (B) — 3× trial cost with marginal benefit; overkill for 8-dim Z-space
- Defer (C) — leaves tune stage systematically wrong for non-ML winners

**Implementation**:
- `OptunaSearchSpace` gains `fatal_prior_boost_low: float = 1.0` and `fatal_prior_boost_high: float = 5.0`
- `params.yaml tune.optuna.search_space` gains those two fields
- `OptunaTuner.__init__()` reads `evaluation_report.json` to know the current winner; accepts `gmm_config: GMMConfig`
- `_objective()` branches on winner: suggests and uses only the relevant imbalance param (fatal_threshold for ml, focal_loss_gamma for dl, fatal_prior_boost for gmm); trains the corresponding trainer with a single seed for trial speed
- `tune/run.py` passes `gmm_config` to `OptunaTuner`
