"""Tests for configuration loading and infrastructure."""
import pytest
from pathlib import Path

from src.config import load_config, FeatureSelectionConfig
from src.metrics import make_eval_dataset
import numpy as np


class TestConfig:
    """Test that params.yaml loads correctly with all sections."""

    def test_load_config_reads_all_sections(self):
        """Config loads successfully with all required sections."""
        config = load_config()
        
        assert config.features is not None
        assert config.data is not None
        assert config.model is not None
        assert config.dl is not None
        assert config.mlflow is not None
        assert config.ab_test is not None
        assert config.feature_selection is not None

    def test_feature_selection_config_has_defaults(self):
        """Feature selection config has expected defaults."""
        config = load_config()
        
        assert config.feature_selection.method == "none"
        assert config.feature_selection.n_features == 10
        assert config.feature_selection.threshold == 0.95

    def test_feature_selection_config_has_valid_method(self):
        """Feature selection method is one of allowed values."""
        config = load_config()
        
        valid_methods = {"none", "mutual_info", "rfe", "correlation", "vif"}
        assert config.feature_selection.method in valid_methods

    def test_config_has_ordinal_columns_dict(self):
        """Ordinal columns are loaded as dict with category lists."""
        config = load_config()

        assert isinstance(config.features.ordinal_columns, dict)
        # DAYOFWEEK and MONTH are now cyclical — neither should be ordinal
        assert "DAYOFWEEK" not in config.features.ordinal_columns
        assert "MONTH" not in config.features.ordinal_columns

        # HOUR, MONTH, DAYOFWEEK are all cyclical
        assert "MONTH" in config.features.cyclical_columns
        assert "HOUR" in config.features.cyclical_columns
        assert "DAYOFWEEK" in config.features.cyclical_columns
        assert config.features.cyclical_columns["MONTH"] == 12
        assert config.features.cyclical_columns["HOUR"] == 24
        assert config.features.cyclical_columns["DAYOFWEEK"] == 7


class TestMetrics:
    """Test MLflow evaluation dataset helpers."""

    def test_make_eval_dataset_creates_dataset(self):
        """make_eval_dataset returns a valid MLflow Dataset."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        dataset = make_eval_dataset(y_true, y_pred)
        
        # MLflow dataset should have a source property
        assert hasattr(dataset, 'source')

    def test_make_eval_dataset_handles_binary_labels(self):
        """Dataset creation works with binary 0/1 labels."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        dataset = make_eval_dataset(y_true, y_pred)
        assert dataset is not None

    def test_make_eval_dataset_accepts_feature_names(self):
        """Dataset creation accepts optional feature_names parameter."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        feature_names = ["feature1", "feature2", "feature3"]
        
        # Should not raise
        dataset = make_eval_dataset(y_true, y_pred, feature_names=feature_names)
        assert dataset is not None
