from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.train_vae.vae_trainer import Encoder


@dataclass
class EncodeResult:
    """Result of latent space encoding (pure projection, no augmentation)."""
    Z_train_augmented: np.ndarray
    Z_val: np.ndarray
    Z_test: np.ndarray


class LatentEncoder:
    def __init__(
        self,
        encoder_path: str,
        latent_dim: int,
    ) -> None:
        self._encoder_path = encoder_path
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
    ) -> EncodeResult:
        """Project augmented X splits to latent space Z via frozen encoder.
        
        Pure projection only — no LSA. Augmentation was handled by upstream
        augment stage (CTGAN). Encoder uses μ path (deterministic, not sampled z).
        
        Args:
            X_train_augmented: Training features (already augmented by augment stage)
            y_train_augmented: Training labels (unused; run.py saves as pass-through)
            X_val: Validation features (never augmented)
            X_test: Test features (never augmented)
        
        Returns:
            EncodeResult with Z_train_augmented, Z_val, Z_test
        """
        encoder = self._load_encoder()
        
        Z_train_augmented = self._encode_split(encoder, X_train_augmented)
        Z_val = self._encode_split(encoder, X_val)
        Z_test = self._encode_split(encoder, X_test)
        
        return EncodeResult(
            Z_train_augmented=Z_train_augmented,
            Z_val=Z_val,
            Z_test=Z_test,
        )
