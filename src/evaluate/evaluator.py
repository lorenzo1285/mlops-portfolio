from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationResult:
    winner: str          # "ml" or "dl"
    p_value: float
    cohens_d: float
    ml_mean_f1: float
    dl_mean_f1: float
    ml_ci_low: float
    ml_ci_high: float
    dl_ci_low: float
    dl_ci_high: float


class ABEvaluator:
    """Welch's t-test comparing macro F1 distributions of ML vs DL seeds.

    Loads per-seed eout_macro_f1 values from the two MLflow experiments,
    runs the test, and returns a structured result. Falls back to the
    tiebreak rule (default: ml) when p >= alpha.

    Public interface
    ----------------
    evaluate(X_test, y_test) → EvaluationResult
        Runs held-out evaluation for both model families and performs A/B test.
        Asserts constitutional gates: F1 > 0.55, minority recall > 0.40.
    """

    def __init__(self, mlflow_config, ab_test_config) -> None:
        self._mlflow_config = mlflow_config
        self._ab_test_config = ab_test_config

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> EvaluationResult:
        raise NotImplementedError
