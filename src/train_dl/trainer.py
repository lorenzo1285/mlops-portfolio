from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DLTrainResult:
    run_ids: list[str]
    best_run_id: str
    mean_macro_f1: float
    mean_minority_recall: float
    model_path: str


class DLTrainer:
    """PyTorch ShallowMLP multi-seed training with MLflow tracking.

    Trains one MLP per seed with BCEWithLogitsLoss, early stopping on
    val loss, and class-weighted sampling. Evaluates via a pyfunc wrapper
    so mlflow.evaluate() is used consistently with the ML trainer.

    Public interface
    ----------------
    train(X_train, y_train, X_val, y_val) → DLTrainResult
    """

    def __init__(self, mlflow_config, dl_config, seeds: list[int]) -> None:
        self._mlflow_config = mlflow_config
        self._dl_config = dl_config
        self._seeds = seeds

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> DLTrainResult:
        raise NotImplementedError
