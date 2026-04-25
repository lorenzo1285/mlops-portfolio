from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TuneResult:
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    best_run_id: str


class HyperparamTuner:
    """Katib Bayesian HPO on the winning model family with MLflow per-trial logging.

    Submits a Katib Experiment CRD to Kubernetes; each trial runs as a pod
    executing src/tune/trial.py. Search space is defined in k8s/katib/*.yaml.
    For DL models, the MLP architecture is fixed by the prior EvoTorch NAS run.

    Public interface
    ----------------
    tune(X_train, y_train, X_val, y_val) → TuneResult
        Submits Katib Experiment; polls until Succeeded; returns best params.
    """

    def __init__(self, mlflow_config, tune_config, winner: str) -> None:
        self._mlflow_config = mlflow_config
        self._tune_config = tune_config
        self._winner = winner

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TuneResult:
        raise NotImplementedError
