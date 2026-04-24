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
    """Optuna TPE search on the winning model family with MLflow per-trial logging.

    Each trial is one MLflow run in the crash-severity-tune experiment.
    Search space is read from params.yaml under tune.ml_search_space or
    tune.dl_search_space depending on winner. Pruning halts unpromising
    DL trials early based on per-epoch val loss.

    Public interface
    ----------------
    tune(X_train, y_train, X_val, y_val) → TuneResult
        Runs n_trials Optuna trials; best params written back to params.yaml.
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
