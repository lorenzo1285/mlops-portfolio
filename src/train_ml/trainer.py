from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MLTrainResult:
    run_ids: list[str]
    best_run_id: str
    mean_macro_f1: float
    mean_minority_recall: float
    model_path: str


class MLTrainer:
    """PyCaret multi-seed training with MLflow tracking via mlflow.evaluate().

    Trains one model per seed in config.ab_test.seeds, logs all metrics
    to the crash-severity-ml experiment, and saves the best-seed model.
    autolog is explicitly disabled; all metrics go through mlflow.evaluate().

    Public interface
    ----------------
    train(X_train, y_train, X_val, y_val) → MLTrainResult
    """

    def __init__(self, mlflow_config, model_config, seeds: list[int]) -> None:
        self._mlflow_config = mlflow_config
        self._model_config = model_config
        self._seeds = seeds

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> MLTrainResult:
        raise NotImplementedError
