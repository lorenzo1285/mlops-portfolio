"""Tests for encode stage - Latent space encoding with LSA augmentation."""
import numpy as np
import pytest
import torch
from pathlib import Path

from src.config import EncodeConfig, MLflowConfig, VAEConfig
from src.encode.encoder import EncodeResult, LatentEncoder
from src.train_vae.vae_trainer import DVAETrainer


class TestLatentEncoder:
    """Boundary tests for latent space encoding with LSA."""

    # --- fixtures ---

    @pytest.fixture(scope="module")
    def minimal_vae_config(self):
        """Minimal VAE config for fast encoder training."""
        return VAEConfig(
            encoder_dims=[16, 8],
            latent_dim=4,
            beta=1.0,
            dropout_p=0.15,
            epochs=5,
            patience=3,
            batch_size=32,
            lr=0.001,
            experiment_name="test-encode-vae",
        )

    @pytest.fixture(scope="module")
    def trained_encoder_path(self, minimal_vae_config, tmp_path_factory):
        """Train a real minimal VAE and return encoder checkpoint path.

        Uses tmp_path_factory (session-scoped) so the VAE trains once per
        module and artifacts land in a temp dir, not models/.
        """
        mlflow_dir = tmp_path_factory.mktemp("mlruns")
        output_dir = tmp_path_factory.mktemp("models")

        mlflow_cfg = MLflowConfig(
            # file:// URI required on Windows (bare paths fail registry init)
            tracking_uri=mlflow_dir.as_uri(),
            experiment_name_ml="test-ml",
            experiment_name_dl="test-dl",
            experiment_name_vae="test-encode-vae",
            experiment_name_tune="test-tune",
            model_name="test-model",
        )

        np.random.seed(42)
        X_all = np.random.randn(100, 10).astype(np.float32)

        trainer = DVAETrainer(minimal_vae_config, mlflow_cfg)
        result = trainer.train(X_all, output_dir=output_dir)

        return result.encoder_path

    @pytest.fixture
    def encode_config(self):
        """Standard encode config (min_fatal_samples=10)."""
        return EncodeConfig(lsa_target_ratio=0.05, min_fatal_samples=10)

    @pytest.fixture
    def encode_config_low_min(self):
        """Encode config with low min_fatal_samples for tests that need LSA to trigger.

        With N=200 and ratio=0.05, the LSA threshold is 10 samples (5% of 200).
        min_fatal_samples=10 makes it impossible to trigger LSA without also
        triggering the RuntimeError guard.  This config sets min_fatal_samples=3
        so tests can use n_fatal=4 (2%) to exercise augmentation logic.
        """
        return EncodeConfig(lsa_target_ratio=0.05, min_fatal_samples=3)

    @pytest.fixture
    def dummy_splits_with_fatal(self):
        """Synthetic train/val/test splits with 15 fatal samples (7.5% > 5% target)."""
        np.random.seed(42)

        X_train = np.random.randn(200, 10).astype(np.float32)
        y_train = np.zeros(200, dtype=np.int64)
        y_train[:15] = 2   # 15 fatal (7.5%)
        y_train[15:50] = 1  # 35 injury

        X_val = np.random.randn(50, 10).astype(np.float32)
        y_val = np.zeros(50, dtype=np.int64)
        y_val[:5] = 2
        y_val[5:15] = 1

        X_test = np.random.randn(50, 10).astype(np.float32)
        y_test = np.zeros(50, dtype=np.int64)
        y_test[:5] = 2
        y_test[5:15] = 1

        return X_train, y_train, X_val, y_val, X_test, y_test

    # --- tests ---

    def test_latent_encoder_returns_encode_result(
        self, trained_encoder_path, encode_config, dummy_splits_with_fatal
    ):
        """Given valid splits with 15 fatal samples (above threshold), encode() returns
        EncodeResult with correct shapes and zero synthetic augmentation."""
        X_train, y_train, X_val, _, X_test, _ = dummy_splits_with_fatal
        encoder = LatentEncoder(
            encoder_path=trained_encoder_path,
            encode_config=encode_config,
            latent_dim=4,
        )

        result = encoder.encode(X_train, y_train, X_val, X_test)

        assert isinstance(result, EncodeResult)
        assert result.Z_train_augmented.shape[1] == 4
        assert result.Z_val.shape[1] == 4
        assert result.Z_test.shape[1] == 4
        # 15 fatals already above 5% target - no augmentation expected
        assert result.n_real_fatal == 15
        assert result.n_synthetic == 0

    def test_lsa_augments_z_train_to_target_ratio(
        self, trained_encoder_path, encode_config_low_min, dummy_splits_with_fatal
    ):
        """LSA synthesizes fatal samples in Z_train until fatal fraction >= lsa_target_ratio.

        Uses encode_config_low_min (min_fatal_samples=3) so that n_fatal=4 (2% of 200)
        is above the RuntimeError guard but below the 5% target, exercising LSA.
        """
        X_train, _, X_val, _, X_test, _ = dummy_splits_with_fatal

        y_train = np.zeros(200, dtype=np.int64)
        y_train[:4] = 2   # 4/200 = 2% — below target, above min_fatal_samples=3
        y_train[4:40] = 1

        encoder = LatentEncoder(
            encoder_path=trained_encoder_path,
            encode_config=encode_config_low_min,
            latent_dim=4,
        )

        result = encoder.encode(X_train, y_train, X_val, X_test)

        fatal_fraction = (result.y_train_augmented == 2).sum() / len(result.y_train_augmented)
        assert fatal_fraction >= encode_config_low_min.lsa_target_ratio, (
            f"Expected fatal fraction >= {encode_config_low_min.lsa_target_ratio}, "
            f"got {fatal_fraction:.4f}"
        )
        assert len(result.Z_train_augmented) > len(X_train), (
            f"Expected augmented train size > {len(X_train)}, "
            f"got {len(result.Z_train_augmented)}"
        )
        assert result.n_synthetic > 0
        assert result.n_real_fatal == 4

    def test_z_val_and_z_test_not_augmented(
        self, trained_encoder_path, encode_config, dummy_splits_with_fatal
    ):
        """Z_val and Z_test rows match original split sizes regardless of LSA on Z_train."""
        X_train, y_train, X_val, _, X_test, _ = dummy_splits_with_fatal

        encoder = LatentEncoder(
            encoder_path=trained_encoder_path,
            encode_config=encode_config,
            latent_dim=4,
        )

        result = encoder.encode(X_train, y_train, X_val, X_test)

        assert result.Z_val.shape[0] == len(X_val), (
            f"Z_val must not be augmented: expected {len(X_val)} rows, "
            f"got {result.Z_val.shape[0]}"
        )
        assert result.Z_test.shape[0] == len(X_test), (
            f"Z_test must not be augmented: expected {len(X_test)} rows, "
            f"got {result.Z_test.shape[0]}"
        )
        # y_train_augmented length must be consistent with Z_train_augmented
        assert len(result.y_train_augmented) == len(result.Z_train_augmented), (
            "y_train_augmented and Z_train_augmented must have the same number of rows"
        )

    def test_encoder_fails_when_insufficient_fatal_samples(
        self, trained_encoder_path, encode_config, dummy_splits_with_fatal
    ):
        """Raises RuntimeError when y_train has fewer than min_fatal_samples fatal cases."""
        X_train, _, X_val, _, X_test, _ = dummy_splits_with_fatal
        y_train = np.zeros(200, dtype=np.int64)
        y_train[:5] = 2  # 5 fatals < min_fatal_samples=10

        encoder = LatentEncoder(
            encoder_path=trained_encoder_path,
            encode_config=encode_config,
            latent_dim=4,
        )

        with pytest.raises(RuntimeError, match="(?i)fatal.*sample"):
            encoder.encode(X_train, y_train, X_val, X_test)

    def test_encoder_uses_mu_not_sampled_z(
        self, trained_encoder_path, encode_config, dummy_splits_with_fatal
    ):
        """Encoding is deterministic: uses mu directly, not stochastic reparameterized z."""
        X_train, y_train, X_val, _, X_test, _ = dummy_splits_with_fatal
        encoder = LatentEncoder(
            encoder_path=trained_encoder_path,
            encode_config=encode_config,
            latent_dim=4,
        )

        result1 = encoder.encode(X_train, y_train, X_val, X_test)
        result2 = encoder.encode(X_train, y_train, X_val, X_test)

        np.testing.assert_array_equal(
            result1.Z_val,
            result2.Z_val,
            err_msg="Z_val must be deterministic (mu, not sampled z)",
        )
        np.testing.assert_array_equal(
            result1.Z_test,
            result2.Z_test,
            err_msg="Z_test must be deterministic (mu, not sampled z)",
        )

    def test_no_augmentation_when_fatal_fraction_already_sufficient(
        self, trained_encoder_path, encode_config, dummy_splits_with_fatal
    ):
        """When real fatal fraction >= lsa_target_ratio, no synthetic samples are added."""
        X_train, _, X_val, _, X_test, _ = dummy_splits_with_fatal
        y_train = np.zeros(200, dtype=np.int64)
        y_train[:12] = 2   # 12/200 = 6% > 5% target
        y_train[12:50] = 1

        encoder = LatentEncoder(
            encoder_path=trained_encoder_path,
            encode_config=encode_config,
            latent_dim=4,
        )

        result = encoder.encode(X_train, y_train, X_val, X_test)

        assert result.n_synthetic == 0, (
            f"Expected n_synthetic=0 when fatal fraction already sufficient, "
            f"got {result.n_synthetic}"
        )
        assert len(result.Z_train_augmented) == len(X_train), (
            f"Expected no augmentation when fatal fraction sufficient, "
            f"got {len(result.Z_train_augmented)} vs {len(X_train)}"
        )
