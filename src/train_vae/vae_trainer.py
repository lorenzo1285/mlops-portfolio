from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import MLflowConfig, VAEConfig


@dataclass
class VAETrainResult:
    best_epoch: int
    final_elbo: float
    encoder_path: str
    decoder_path: str
    run_id: str


class DVAETrainer:
    def __init__(self, vae_config: VAEConfig, mlflow_config: MLflowConfig) -> None:
        self._vae_config = vae_config
        self._mlflow_config = mlflow_config

    def train(self, X_all: np.ndarray) -> VAETrainResult:
        raise NotImplementedError
