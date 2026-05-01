from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
from scipy import stats


@dataclass
class EvaluationResult:
    winner: str                  # "ml" or "dl"
    p_value: float
    cohens_d: float
    ml_mean_f1: float
    dl_mean_f1: float
    ml_ci_low: float
    ml_ci_high: float
    dl_ci_low: float
    dl_ci_high: float
    ml_mean_fatal_recall: float
    dl_mean_fatal_recall: float
    gates_passed: bool           # True iff winner clears both constitutional thresholds


class ABEvaluator:
    """Welch's t-test comparing macro F1 distributions of ML vs DL seeds.

    Queries per-seed eout_macro_f1 and eout_fatal_recall from MLflow, runs
    Welch's t-test, and returns a structured result. Falls back to the
    tiebreak rule (default: ml) when p >= alpha. Asserts constitutional
    gates: macro_f1 > threshold AND fatal_recall > threshold.

    Public interface
    ----------------
    evaluate() → EvaluationResult
    """

    def __init__(self, mlflow_config, ab_test_config, model_config) -> None:
        self._mlflow_config = mlflow_config
        self._ab_test_config = ab_test_config
        self._model_config = model_config

    def evaluate(self) -> EvaluationResult:
        """Run A/B test on ML vs DL experiments; check constitutional gates."""
        mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)
        
        # Query MLflow for per-seed metrics
        ml_f1, ml_recall = self._get_metrics(self._mlflow_config.experiment_name_ml)
        dl_f1, dl_recall = self._get_metrics(self._mlflow_config.experiment_name_dl)
        
        # Welch's t-test on F1 distributions
        t_stat, p_value = stats.ttest_ind(ml_f1, dl_f1, equal_var=False)
        
        # Determine winner
        if p_value < self._ab_test_config.alpha:
            # Statistically significant difference
            winner = "ml" if np.mean(ml_f1) > np.mean(dl_f1) else "dl"
        else:
            # Not significant → apply tiebreak
            winner = self._ab_test_config.tiebreak
        
        # Compute Cohen's d
        cohens_d = self._cohens_d(ml_f1, dl_f1)
        
        # Compute 95% CIs
        ml_ci_low, ml_ci_high = self._confidence_interval(ml_f1)
        dl_ci_low, dl_ci_high = self._confidence_interval(dl_f1)
        
        # Mean metrics
        ml_mean_f1 = float(np.mean(ml_f1))
        dl_mean_f1 = float(np.mean(dl_f1))
        ml_mean_fatal_recall = float(np.mean(ml_recall))
        dl_mean_fatal_recall = float(np.mean(dl_recall))
        
        # Check constitutional gates on winner
        winner_f1 = ml_mean_f1 if winner == "ml" else dl_mean_f1
        winner_recall = ml_mean_fatal_recall if winner == "ml" else dl_mean_fatal_recall
        
        gates_passed = (
            winner_f1 > self._model_config.macro_f1_threshold
            and winner_recall > self._model_config.fatal_recall_threshold
        )
        
        return EvaluationResult(
            winner=winner,
            p_value=float(p_value),
            cohens_d=cohens_d,
            ml_mean_f1=ml_mean_f1,
            dl_mean_f1=dl_mean_f1,
            ml_ci_low=ml_ci_low,
            ml_ci_high=ml_ci_high,
            dl_ci_low=dl_ci_low,
            dl_ci_high=dl_ci_high,
            ml_mean_fatal_recall=ml_mean_fatal_recall,
            dl_mean_fatal_recall=dl_mean_fatal_recall,
            gates_passed=gates_passed,
        )

    def _get_metrics(self, experiment_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Query MLflow for eout_macro_f1 and eout_fatal_recall from all runs."""
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_name}' not found in MLflow")
        
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED'",
            output_format="pandas",
        )
        
        if runs.empty:
            raise ValueError(f"No finished runs found in experiment '{experiment_name}'")
        
        f1_values = runs["metrics.eout_macro_f1"].values
        recall_values = runs["metrics.eout_fatal_recall"].values
        
        return f1_values, recall_values

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
