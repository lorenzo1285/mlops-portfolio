"""CTGAN/TVAE augmentation for imbalanced Fatal class."""
from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd
from ctgan import TVAE

from src.config import AugmentConfig


@dataclass
class AugmentResult:
    """Result of CTGAN augmentation."""
    X_augmented: np.ndarray
    y_augmented: np.ndarray
    n_real_fatal: int
    n_synthetic: int


class CTGANAugmenter:
    """Generate synthetic Fatal class samples using TVAE."""
    
    def __init__(self, augment_config: AugmentConfig):
        """Initialize augmenter with config.
        
        Args:
            augment_config: Configuration with tvae_epochs, target_fatal_ratio, random_state
        """
        self.config = augment_config
    
    def augment(self, X_train: np.ndarray, y_train: np.ndarray) -> AugmentResult:
        """Augment Fatal class to reach target ratio using TVAE.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) with values {0, 1, 2}
        
        Returns:
            AugmentResult with X_augmented, y_augmented, metadata
        
        Raises:
            RuntimeError: If Fatal samples < 10
        """
        # Validate Fatal sample count
        fatal_mask = (y_train == 2)
        n_fatal = fatal_mask.sum()
        
        if n_fatal < 10:
            raise RuntimeError(
                f"Insufficient Fatal samples ({n_fatal}) for TVAE training — "
                f"minimum 10 required for reliable generative modeling"
            )
        
        # Extract Fatal rows as DataFrame (TVAE expects DataFrame)
        X_fatal = X_train[fatal_mask]
        n_features = X_train.shape[1]
        X_fatal_df = pd.DataFrame(X_fatal, columns=[f"f{i}" for i in range(n_features)])
        
        # Fit TVAE on Fatal class
        tvae = TVAE(
            epochs=self.config.tvae_epochs,
            cuda=False,  # CPU-only for portability
        )
        tvae.fit(X_fatal_df)
        
        # Compute number of synthetic samples needed
        # Formula: n_synthetic = (target × N - n_fatal) / (1 - target)
        # where N = len(X_train) is the original training set size
        n_total = len(X_train)
        target_ratio = self.config.target_fatal_ratio
        
        n_synthetic_float = (target_ratio * n_total - n_fatal) / (1 - target_ratio)
        n_synthetic = max(0, ceil(n_synthetic_float))
        
        if n_synthetic == 0:
            # Already at or above target ratio
            return AugmentResult(
                X_augmented=X_train.copy(),
                y_augmented=y_train.copy(),
                n_real_fatal=int(n_fatal),
                n_synthetic=0,
            )
        
        # Sample synthetic Fatal rows
        synthetic_df = tvae.sample(n_synthetic)
        synthetic_np = synthetic_df.to_numpy().astype(X_train.dtype)
        
        # Stack: original + synthetic
        X_augmented = np.vstack([X_train, synthetic_np])
        y_augmented = np.hstack([y_train, np.full(n_synthetic, 2, dtype=y_train.dtype)])
        
        return AugmentResult(
            X_augmented=X_augmented,
            y_augmented=y_augmented,
            n_real_fatal=int(n_fatal),
            n_synthetic=n_synthetic,
        )
