"""Tests for train_gmm stage - GMM classifier on Z-space."""
from pathlib import Path

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from src.config import ABTestConfig, GMMConfig, MLflowConfig, ModelConfig
from src.train_gmm.trainer import GMMClassifier, GMMTrainResult, GMMTrainer


class TestGMMClassifier:
    """Boundary tests for GMMClassifier wrapper (TDD Slice A)."""

    @pytest.fixture(scope="module")
    def real_Z_splits(self):
        data_dir = Path("data/processed")
        return (
            np.load(data_dir / "Z_train_augmented.npy"),
            np.load(data_dir / "y_train_augmented.npy"),
            np.load(data_dir / "Z_val.npy"),
            np.load(data_dir / "y_val.npy"),
        )

    @pytest.fixture(scope="module")
    def fitted_gmms(self, real_Z_splits):
        Z_train, y_train, _, _ = real_Z_splits
        gmms = {}
        for class_label in [0, 1, 2]:
            mask = y_train == class_label
            Z_class = Z_train[mask]
            n_components = 2 if class_label == 2 else 1
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                reg_covar=1e-6,
                max_iter=100,
                n_init=5,
                random_state=42,
            )
            gmm.fit(Z_class)
            gmms[class_label] = gmm
        return gmms

    @pytest.fixture(scope="module")
    def log_priors(self, real_Z_splits):
        _, y_train, _, _ = real_Z_splits
        priors = np.array([
            (y_train == 0).sum() / len(y_train),
            (y_train == 1).sum() / len(y_train),
            (y_train == 2).sum() / len(y_train),
        ])
        return np.log(priors)

    def test_gmm_classifier_predict_returns_valid_shape_and_labels(
        self, fitted_gmms, log_priors, real_Z_splits
    ):
        _, _, Z_val, _ = real_Z_splits

        classifier = GMMClassifier(
            gmms=fitted_gmms,
            log_priors=log_priors,
            fatal_prior_boost=1.0,
        )
        y_pred = classifier.predict(Z_val)

        assert y_pred.shape == (len(Z_val),), f"Expected shape {(len(Z_val),)}, got {y_pred.shape}"
        assert set(y_pred).issubset({0, 1, 2}), f"Predictions contain invalid labels: {set(y_pred)}"
        assert y_pred.dtype in [np.int32, np.int64, int], f"Expected integer dtype, got {y_pred.dtype}"

    def test_gmm_classifier_fatal_prior_boost_increases_fatal_predictions(
        self, fitted_gmms, log_priors, real_Z_splits
    ):
        _, _, Z_val, _ = real_Z_splits

        # Baseline: no boost
        classifier_baseline = GMMClassifier(
            gmms=fitted_gmms,
            log_priors=log_priors,
            fatal_prior_boost=1.0,
        )
        y_pred_baseline = classifier_baseline.predict(Z_val)
        fatal_fraction_baseline = (y_pred_baseline == 2).sum() / len(Z_val)

        # Boosted: 3x boost on Fatal prior
        classifier_boosted = GMMClassifier(
            gmms=fitted_gmms,
            log_priors=log_priors,
            fatal_prior_boost=3.0,
        )
        y_pred_boosted = classifier_boosted.predict(Z_val)
        fatal_fraction_boosted = (y_pred_boosted == 2).sum() / len(Z_val)

        assert (
            fatal_fraction_boosted > fatal_fraction_baseline
        ), f"Boosted fatal fraction ({fatal_fraction_boosted:.3f}) not greater than baseline ({fatal_fraction_baseline:.3f})"


class TestGMMTrainer:
    """Boundary tests for GMMTrainer (TDD Slice B)."""

    @pytest.fixture(scope="module")
    def real_Z_y_splits(self):
        """Load real Z-space and label artifacts from data/processed/."""
        data_dir = Path("data/processed")
        return (
            np.load(data_dir / "Z_train_augmented.npy"),
            np.load(data_dir / "y_train_augmented.npy"),
            np.load(data_dir / "Z_val.npy"),
            np.load(data_dir / "y_val.npy"),
            np.load(data_dir / "Z_test.npy"),
            np.load(data_dir / "y_test.npy"),
        )

    @pytest.fixture
    def gmm_config(self):
        """GMM config with sensible defaults."""
        return GMMConfig(
            n_components={0: 1, 1: 1, 2: 2},
            covariance_type="full",
            reg_covar=1e-6,
            max_iter=100,
            n_init=5,
            fatal_prior_boost=1.0,
            experiment_name="test-gmm",
        )

    @pytest.fixture
    def model_config(self):
        """Model config for 3-class classification."""
        return ModelConfig(
            n_classes=3,
            n_select=10,
            macro_f1_threshold=0.35,
            fatal_recall_threshold=0.50,
        )

    @pytest.fixture
    def mlflow_config(self):
        """MLflow config pointing to test tracking URI."""
        return MLflowConfig(
            tracking_uri="mlruns/",
            experiment_name_ml="test-ml",
            experiment_name_dl="test-dl",
            experiment_name_gmm="test-gmm",
            experiment_name_vae="test-vae",
            experiment_name_tune="test-tune",
            model_name="test-model",
        )

    @pytest.fixture
    def ab_test_config(self):
        """AB test config with single seed for fast testing."""
        return ABTestConfig(seeds=[0], alpha=0.05, tiebreak=["ml", "dl", "gmm"])

    def test_gmm_trainer_train_returns_gmm_train_result(
        self,
        gmm_config,
        model_config,
        mlflow_config,
        ab_test_config,
        real_Z_y_splits,
    ):
        """GMMTrainer.train() returns GMMTrainResult with non-empty run_id, model_path, and eout_macro_f1 > 0."""
        Z_train, y_train, Z_val, y_val, Z_test, y_test = real_Z_y_splits

        trainer = GMMTrainer(
            gmm_config=gmm_config,
            model_config=model_config,
            mlflow_config=mlflow_config,
            ab_test_config=ab_test_config,
        )

        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)

        # Assert: Returns GMMTrainResult with required fields
        assert isinstance(result, GMMTrainResult), f"Expected GMMTrainResult, got {type(result)}"
        assert result.run_id is not None and len(result.run_id) > 0, "run_id should be set"
        assert result.model_path is not None and len(result.model_path) > 0, "model_path should be set"
        assert Path(result.model_path).exists(), f"Model file {result.model_path} should exist"
        assert result.eout_macro_f1 > 0.0, f"eout_macro_f1 should be > 0, got {result.eout_macro_f1}"
        assert result.eout_fatal_recall >= 0.0, f"eout_fatal_recall should be >= 0, got {result.eout_fatal_recall}"
