from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from src.config import EncodeConfig
from src.train_vae.vae_trainer import Encoder


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

    def _load_encoder(self) -> Encoder:
        checkpoint = torch.load(self._encoder_path, weights_only=True)
        encoder = Encoder(
            input_dim=checkpoint["input_dim"],
            encoder_dims=checkpoint["encoder_dims"],
            latent_dim=checkpoint["latent_dim"],
        )
        encoder.load_state_dict(checkpoint["state_dict"])
        encoder.eval()
        return encoder

    def _encode_split(self, encoder: Encoder, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            mu, _ = encoder(torch.tensor(X, dtype=torch.float32))
            return mu.numpy()

    def _apply_lsa(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        z_fatal = Z_train[y_train == 2]
        n_real_fatal = len(z_fatal)

        fatal_mean = z_fatal.mean(axis=0)
        # ddof=1 for unbiased std; floor at 1e-6 so synthetics still spread when
        # all real fatal vectors share a value in a given dimension
        fatal_std = np.maximum(z_fatal.std(axis=0, ddof=1), 1e-6)

        target = self._encode_config.lsa_target_ratio
        N_train = len(Z_train)

        if target >= 1.0:
            n_synthetic = 0
        else:
            n_synthetic = max(
                0,
                int(math.ceil((target * N_train - n_real_fatal) / (1.0 - target))),
            )

        if n_synthetic == 0:
            return Z_train, y_train, 0

        rng = np.random.default_rng(self._encode_config.random_state)
        synthetic_z = (
            fatal_mean + rng.standard_normal((n_synthetic, self._latent_dim)) * fatal_std
        ).astype(Z_train.dtype)

        return (
            np.vstack([Z_train, synthetic_z]),
            np.hstack([y_train, np.full(n_synthetic, 2, dtype=y_train.dtype)]),
            n_synthetic,
        )

    def encode(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
    ) -> EncodeResult:
        # Raises RuntimeError before any encoding if y_train lacks enough fatal rows,
        # so downstream LSA has a real centroid to sample from.
        n_fatal = int((y_train == 2).sum())
        if n_fatal < self._encode_config.min_fatal_samples:
            raise RuntimeError(
                f"Insufficient fatal samples for LSA: {n_fatal} < "
                f"{self._encode_config.min_fatal_samples}"
            )

        encoder = self._load_encoder()

        Z_train = self._encode_split(encoder, X_train)
        Z_val = self._encode_split(encoder, X_val)
        Z_test = self._encode_split(encoder, X_test)

        Z_train_augmented, y_train_augmented, n_synthetic = self._apply_lsa(Z_train, y_train)

        return EncodeResult(
            Z_train_augmented=Z_train_augmented,
            Z_val=Z_val,
            Z_test=Z_test,
            y_train_augmented=y_train_augmented,
            n_real_fatal=n_fatal,
            n_synthetic=int(n_synthetic),
        )
