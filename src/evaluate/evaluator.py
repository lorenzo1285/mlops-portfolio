from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
from scipy import stats


@dataclass
class EvaluationResult:
    winner: str                      # "ml", "dl", or "gmm"
    p_value_ml_dl: float             # pairwise ml vs dl
    p_value_ml_gmm: float            # pairwise ml vs gmm
    p_value_dl_gmm: float            # pairwise dl vs gmm
    cohens_d_ml_dl: float            # effect size ml vs dl
    cohens_d_ml_gmm: float           # effect size ml vs gmm
    cohens_d_dl_gmm: float           # effect size dl vs gmm
    ml_mean_f1: float
    dl_mean_f1: float
    gmm_mean_f1: float
    ml_ci_low: float
    ml_ci_high: float
    dl_ci_low: float
    dl_ci_high: float
    gmm_ci_low: float
    gmm_ci_high: float
    ml_mean_fatal_recall: float
    dl_mean_fatal_recall: float
    gmm_mean_fatal_recall: float
    gates_passed: bool               # True iff winner clears both constitutional thresholds


class ABEvaluator:
    """3-way A/B/C comparison of ML vs DL vs GMM using Welch's t-tests with Bonferroni correction.

    Queries per-seed eout_macro_f1 and eout_fatal_recall from all three MLflow experiments,
    runs three pairwise Welch's t-tests with Bonferroni-corrected alpha (alpha/3), and selects
    winner based on: (1) candidate set of classifiers significantly better than at least one
    other, (2) highest mean F1 among candidates, or (3) tiebreak rule if no significant
    differences. Asserts constitutional gates: macro_f1 > threshold AND fatal_recall > threshold.

    Public interface
    ----------------
    evaluate() -> EvaluationResult
    """

    def __init__(self, mlflow_config, ab_test_config, model_config) -> None:
        self._mlflow_config = mlflow_config
        self._ab_test_config = ab_test_config
        self._model_config = model_config

    def evaluate(self) -> EvaluationResult:
        """Run 3-way A/B/C test on ML vs DL vs GMM experiments; check constitutional gates."""
        mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)
        
        # Query MLflow for per-seed metrics (all three experiments)
        ml_f1, ml_recall = self._get_metrics(self._mlflow_config.experiment_name_ml)
        dl_f1, dl_recall = self._get_metrics(self._mlflow_config.experiment_name_dl)
        gmm_f1, gmm_recall = self._get_metrics(self._mlflow_config.experiment_name_gmm)
        
        # Three pairwise Welch's t-tests with Bonferroni correction
        alpha_bonf = self._ab_test_config.alpha / 3.0  # alpha/3 for 3 pairwise comparisons
        
        _, p_value_ml_dl = stats.ttest_ind(ml_f1, dl_f1, equal_var=False)
        _, p_value_ml_gmm = stats.ttest_ind(ml_f1, gmm_f1, equal_var=False)
        _, p_value_dl_gmm = stats.ttest_ind(dl_f1, gmm_f1, equal_var=False)
        
        # Handle NaN p-values (occurs when distributions are identical/zero variance)
        # NaN means no meaningful difference -> treat as p=1.0 (not significant)
        p_value_ml_dl = 1.0 if np.isnan(p_value_ml_dl) else float(p_value_ml_dl)
        p_value_ml_gmm = 1.0 if np.isnan(p_value_ml_gmm) else float(p_value_ml_gmm)
        p_value_dl_gmm = 1.0 if np.isnan(p_value_dl_gmm) else float(p_value_dl_gmm)
        
        # Compute Cohen's d for all pairs
        cohens_d_ml_dl = self._cohens_d(ml_f1, dl_f1)
        cohens_d_ml_gmm = self._cohens_d(ml_f1, gmm_f1)
        cohens_d_dl_gmm = self._cohens_d(dl_f1, gmm_f1)
        
        # Compute 95% CIs
        ml_ci_low, ml_ci_high = self._confidence_interval(ml_f1)
        dl_ci_low, dl_ci_high = self._confidence_interval(dl_f1)
        gmm_ci_low, gmm_ci_high = self._confidence_interval(gmm_f1)
        
        # Mean metrics
        ml_mean_f1 = float(np.mean(ml_f1))
        dl_mean_f1 = float(np.mean(dl_f1))
        gmm_mean_f1 = float(np.mean(gmm_f1))
        ml_mean_fatal_recall = float(np.mean(ml_recall))
        dl_mean_fatal_recall = float(np.mean(dl_recall))
        gmm_mean_fatal_recall = float(np.mean(gmm_recall))
        
        # Winner selection algorithm (3-way with Bonferroni correction)
        # Step 1: Build candidate set = classifiers significantly better than at least one other
        candidates = set()
        
        # ML is a candidate if it beats DL or GMM significantly
        if p_value_ml_dl < alpha_bonf and ml_mean_f1 > dl_mean_f1:
            candidates.add("ml")
        if p_value_ml_gmm < alpha_bonf and ml_mean_f1 > gmm_mean_f1:
            candidates.add("ml")
        
        # DL is a candidate if it beats ML or GMM significantly
        if p_value_ml_dl < alpha_bonf and dl_mean_f1 > ml_mean_f1:
            candidates.add("dl")
        if p_value_dl_gmm < alpha_bonf and dl_mean_f1 > gmm_mean_f1:
            candidates.add("dl")
        
        # GMM is a candidate if it beats ML or DL significantly
        if p_value_ml_gmm < alpha_bonf and gmm_mean_f1 > ml_mean_f1:
            candidates.add("gmm")
        if p_value_dl_gmm < alpha_bonf and gmm_mean_f1 > dl_mean_f1:
            candidates.add("gmm")
        
        # Step 2: Select winner
        if candidates:
            # At least one classifier is significantly better -> pick highest mean F1 among candidates
            candidate_f1s = {
                "ml": ml_mean_f1 if "ml" in candidates else -np.inf,
                "dl": dl_mean_f1 if "dl" in candidates else -np.inf,
                "gmm": gmm_mean_f1 if "gmm" in candidates else -np.inf,
            }
            winner = max(candidate_f1s, key=candidate_f1s.get)
        else:
            # No significant differences -> apply tiebreak (first in priority list)
            winner = self._ab_test_config.tiebreak[0]
        
        # Check constitutional gates on winner
        if winner == "ml":
            winner_f1 = ml_mean_f1
            winner_recall = ml_mean_fatal_recall
        elif winner == "dl":
            winner_f1 = dl_mean_f1
            winner_recall = dl_mean_fatal_recall
        else:  # gmm
            winner_f1 = gmm_mean_f1
            winner_recall = gmm_mean_fatal_recall
        
        gates_passed = (
            winner_f1 > self._model_config.macro_f1_threshold
            and winner_recall > self._model_config.fatal_recall_threshold
        )
        
        return EvaluationResult(
            winner=winner,
            p_value_ml_dl=p_value_ml_dl,
            p_value_ml_gmm=p_value_ml_gmm,
            p_value_dl_gmm=p_value_dl_gmm,
            cohens_d_ml_dl=cohens_d_ml_dl,
            cohens_d_ml_gmm=cohens_d_ml_gmm,
            cohens_d_dl_gmm=cohens_d_dl_gmm,
            ml_mean_f1=ml_mean_f1,
            dl_mean_f1=dl_mean_f1,
            gmm_mean_f1=gmm_mean_f1,
            ml_ci_low=ml_ci_low,
            ml_ci_high=ml_ci_high,
            dl_ci_low=dl_ci_low,
            dl_ci_high=dl_ci_high,
            gmm_ci_low=gmm_ci_low,
            gmm_ci_high=gmm_ci_high,
            ml_mean_fatal_recall=ml_mean_fatal_recall,
            dl_mean_fatal_recall=dl_mean_fatal_recall,
            gmm_mean_fatal_recall=gmm_mean_fatal_recall,
            gates_passed=gates_passed,
        )

    def _get_metrics(self, experiment_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Query MLflow for eout_macro_f1 and eout_fatal_recall from the most recent seed runs."""
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_name}' not found in MLflow")

        n_seeds = len(self._ab_test_config.seeds)
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED'",
            output_format="pandas",
            order_by=["start_time DESC"],
            max_results=n_seeds,
        )

        if runs.empty:
            raise ValueError(f"No finished runs found in experiment '{experiment_name}'")

        f1_col = "metrics.eout_macro_f1"
        recall_col = "metrics.eout_fatal_recall"
        runs = runs.dropna(subset=[f1_col, recall_col])

        if runs.empty:
            raise ValueError(f"No runs with complete metrics in experiment '{experiment_name}'")

        return runs[f1_col].values, runs[recall_col].values

    def _cohens_d(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(
            ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof
        )
        return float((np.mean(x) - np.mean(y)) / pooled_std)

    def _confidence_interval(self, x: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
        """Compute confidence interval using t-distribution."""
        n = len(x)
        mean = np.mean(x)
        se = stats.sem(x)
        margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return float(mean - margin), float(mean + margin)
