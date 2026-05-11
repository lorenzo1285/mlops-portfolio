from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.train_vae.vae_trainer import Encoder

if TYPE_CHECKING:
    from src.config import DriftConfig


@dataclass
class EncodeResult:
    """Result of latent space encoding (pure projection, no augmentation)."""
    Z_train_augmented: np.ndarray
    Z_val: np.ndarray
    Z_test: np.ndarray
    drift_reference_path: str | None = field(default=None)


class LatentEncoder:
    def __init__(
        self,
        encoder_path: str,
        latent_dim: int,
        drift: DriftConfig | None = None,
    ) -> None:
        self._encoder_path = encoder_path
        self._latent_dim = latent_dim
        self._drift = drift

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
        """Encode X to latent Z via encoder μ path (deterministic)."""
        with torch.no_grad():
            mu, _ = encoder(torch.tensor(X, dtype=torch.float32))
            return mu.numpy()

    def encode(
        self,
        X_train_augmented: np.ndarray,
        y_train_augmented: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        X_train: np.ndarray | None = None,
    ) -> EncodeResult:
        """Project augmented X splits to latent space Z via frozen encoder.

        Pure projection only — no LSA. Augmentation was handled by upstream
        augment stage (CTGAN). Encoder uses μ path (deterministic, not sampled z).

        Args:
            X_train_augmented: Training features (already augmented by augment stage)
            y_train_augmented: Training labels (unused; run.py saves as pass-through)
            X_val: Validation features (never augmented)
            X_test: Test features (never augmented)
            X_train: Real pre-augmentation training features; required when drift config
                is set. Used to build the in-distribution reference for MMD drift detection.

        Returns:
            EncodeResult with Z_train_augmented, Z_val, Z_test, drift_reference_path
        """
        encoder = self._load_encoder()

        Z_train_augmented = self._encode_split(encoder, X_train_augmented)
        Z_val = self._encode_split(encoder, X_val)
        Z_test = self._encode_split(encoder, X_test)

        drift_ref_path = None
        if self._drift is not None and X_train is not None:
            drift_ref_path = self._save_drift_reference(encoder, X_train)

        return EncodeResult(
            Z_train_augmented=Z_train_augmented,
            Z_val=Z_val,
            Z_test=Z_test,
            drift_reference_path=drift_ref_path,
        )

    def _save_drift_reference(self, encoder: Encoder, X_train: np.ndarray) -> str:
        from src.metrics import compute_reference_sample_size

        n_ref = compute_reference_sample_size(
            len(X_train), X_train.shape[1], self._drift.min_ratio
        )
        rng = np.random.default_rng(self._drift.random_state)
        idx = rng.choice(len(X_train), size=n_ref, replace=False)
        Z_ref = self._encode_split(encoder, X_train[idx])

        ref_path = self._drift.reference_path
        os.makedirs(os.path.dirname(os.path.abspath(ref_path)), exist_ok=True)
        np.savez(
            ref_path,
            Z_ref=Z_ref,
            Z_mean=Z_ref.mean(axis=0),
            Z_std=Z_ref.std(axis=0),
            n_samples=np.array(n_ref),
            latent_dim=np.array(self._latent_dim),
        )
        return ref_path
