from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import EncodeConfig


@dataclass
class EncodeResult:
    Z_train_augmented: np.ndarray
    Z_val: np.ndarray
    Z_test: np.ndarray
    y_train_augmented: np.ndarray
    n_real_fatal: int
    n_synthetic: int


class LatentEncoder:
    def __init__(
        self,
        encoder_path: str,
        encode_config: EncodeConfig,
        latent_dim: int,
    ) -> None:
        self._encoder_path = encoder_path
        self._encode_config = encode_config
        self._latent_dim = latent_dim

    def encode(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
    ) -> EncodeResult:
        raise NotImplementedError
