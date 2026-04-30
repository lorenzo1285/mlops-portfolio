"""Tests for CTGAN augment stage."""
import os
import tempfile
from math import ceil

import numpy as np
import pytest

from src.augment.augmenter import AugmentResult, CTGANAugmenter
from src.config import AugmentConfig


@pytest.fixture
def augment_config():
    """Minimal config for testing."""
    return AugmentConfig(tvae_epochs=2, target_fatal_ratio=0.15, random_state=42)


@pytest.fixture
def synthetic_train_data():
    """Synthetic train data: 200 rows, 8 cols; 10 Fatal (5%), 190 non-Fatal."""
    np.random.seed(42)
    X_train = np.random.randn(200, 8).astype(np.float32)
    y_train = np.array([2] * 10 + [0] * 100 + [1] * 90, dtype=np.int64)
    return X_train, y_train


class TestCTGANAugmenter:
    """Boundary tests for CTGANAugmenter."""

    def test_augment_increases_fatal_fraction(self, augment_config, synthetic_train_data):
        """Augment should increase Fatal fraction to at least target ratio."""
        X_train, y_train = synthetic_train_data
        
        # Verify Fatal fraction is below target before augmentation
        initial_fatal_fraction = (y_train == 2).mean()
        assert initial_fatal_fraction < augment_config.target_fatal_ratio, (
            f"Test setup error: initial Fatal fraction {initial_fatal_fraction:.3f} "
            f">= target {augment_config.target_fatal_ratio}"
        )
        
        augmenter = CTGANAugmenter(augment_config)
        result = augmenter.augment(X_train, y_train)
        
        # Check returned dataclass
        assert isinstance(result, AugmentResult)
        assert result.X_augmented.shape[1] == 8
        assert result.n_real_fatal == 10
        assert result.n_synthetic > 0
        
        # Fatal fraction should reach target
        fatal_fraction = (result.y_augmented == 2).mean()
        assert fatal_fraction >= augment_config.target_fatal_ratio, (
            f"Fatal fraction {fatal_fraction:.3f} < target {augment_config.target_fatal_ratio}"
        )
    
    def test_non_fatal_rows_unchanged(self, augment_config, synthetic_train_data):
        """Non-Fatal rows should not be modified."""
        X_train, y_train = synthetic_train_data
        
        augmenter = CTGANAugmenter(augment_config)
        result = augmenter.augment(X_train, y_train)
        
        # Count non-Fatal rows
        non_fatal_count_before = (y_train != 2).sum()
        non_fatal_count_after = (result.y_augmented != 2).sum()
        
        assert non_fatal_count_after == non_fatal_count_before, (
            "Non-Fatal row count should remain unchanged"
        )
    
    def test_raises_error_when_too_few_fatal_samples(self, augment_config):
        """Should raise RuntimeError when Fatal samples < 10."""
        np.random.seed(42)
        X_train = np.random.randn(50, 8).astype(np.float32)
        y_train = np.array([2] * 5 + [0] * 45, dtype=np.int64)  # Only 5 Fatal
        
        augmenter = CTGANAugmenter(augment_config)
        
        with pytest.raises(RuntimeError, match="Insufficient Fatal samples"):
            augmenter.augment(X_train, y_train)
