"""Tests for encode stage - Pure projection of augmented splits to latent space."""
import numpy as np
import pytest
import torch
from pathlib import Path

from src.config import MLflowConfig, VAEConfig
from src.encode.encoder import EncodeResult, LatentEncoder
from src.train_vae.vae_trainer import DVAETrainer


class TestLatentEncoderPureProjection:
    """Boundary tests for pure projection encoding (no LSA)."""

    # --- fixtures ---

    @pytest.fixture(scope="module")
    def trained_encoder_path(self, tmp_path_factory):
        """Create a mock encoder checkpoint without training.
        
        DVAETrainer hasn't been updated for KL annealing yet, so we create
        a minimal mock checkpoint directly. The encoder just needs to have
        the right structure to load and encode.
        """
        from src.train_vae.vae_trainer import Encoder
        
        output_dir = tmp_path_factory.mktemp("models")
        encoder_path = output_dir / "vae_encoder.pth"
        
        # Create minimal encoder
        encoder = Encoder(
            input_dim=10,
            encoder_dims=[16, 8],
            latent_dim=4,
        )
        
        # Save checkpoint with required structure
        torch.save(
            {
                "state_dict": encoder.state_dict(),
                "input_dim": 10,
                "encoder_dims": [16, 8],
                "latent_dim": 4,
            },
            encoder_path,
        )
        
        return encoder_path

    @pytest.fixture
    def augmented_splits(self):
        """Synthetic X_train_augmented with CTGAN-generated Fatal rows already included.
        
        Simulates augment stage output:
        - X_train_augmented: 250 rows (200 original + 50 synthetic Fatal)
        - y_train_augmented: 250 labels (65 Fatal total: 15 original + 50 synthetic)
        - X_val: 50 rows (original, unchanged)
        - X_test: 50 rows (original, unchanged)
        """
        np.random.seed(42)
        
        # Original training split: 200 rows, 15 Fatal (7.5%)
        X_train_original = np.random.randn(200, 10).astype(np.float32)
        y_train_original = np.zeros(200, dtype=np.int64)
        y_train_original[:15] = 2   # 15 Fatal
        y_train_original[15:50] = 1  # 35 Injury
        
        # Synthetic Fatal rows (CTGAN output simulation): 50 rows
        X_train_synthetic = np.random.randn(50, 10).astype(np.float32) * 0.9
        y_train_synthetic = np.full(50, 2, dtype=np.int64)  # all Fatal
        
        # Augmented training split: 250 rows, 65 Fatal (26%)
        X_train_augmented = np.vstack([X_train_original, X_train_synthetic])
        y_train_augmented = np.hstack([y_train_original, y_train_synthetic])
        
        # Val/test unchanged (original splits)
        X_val = np.random.randn(50, 10).astype(np.float32)
        X_test = np.random.randn(50, 10).astype(np.float32)
        
        return X_train_augmented, y_train_augmented, X_val, X_test

    # --- tests ---

    def test_encoder_pure_projection_returns_correct_shapes(
        self, trained_encoder_path, augmented_splits
    ):
        """Encoder projects augmented X splits to Z latent vectors with correct shapes.

        No LSA — encoder does pure projection only. X_train_augmented already contains
        CTGAN-generated Fatal rows from upstream augment stage.
        """
        X_train_augmented, y_train_augmented, X_val, X_test = augmented_splits

        encoder = LatentEncoder(
            encoder_path=trained_encoder_path,
            latent_dim=4,
        )
        
        result = encoder.encode(X_train_augmented, y_train_augmented, X_val, X_test)
        
        # Assert correct shapes (pure projection, no row injection)
        assert result.Z_train_augmented.shape == (250, 4), \
            f"Expected Z_train_augmented shape (250, 4), got {result.Z_train_augmented.shape}"
        assert result.Z_val.shape == (50, 4), \
            f"Expected Z_val shape (50, 4), got {result.Z_val.shape}"
        assert result.Z_test.shape == (50, 4), \
            f"Expected Z_test shape (50, 4), got {result.Z_test.shape}"

    def test_encoder_does_not_inject_synthetic_rows(
        self, trained_encoder_path, augmented_splits
    ):
        """Encoder does NOT add synthetic rows — augmentation handled by augment stage.

        Output row counts must exactly match input row counts. No LSA injection.
        """
        X_train_augmented, y_train_augmented, X_val, X_test = augmented_splits

        encoder = LatentEncoder(
            encoder_path=trained_encoder_path,
            latent_dim=4,
        )
        
        result = encoder.encode(X_train_augmented, y_train_augmented, X_val, X_test)
        
        # Assert no row injection (output row count == input row count)
        assert len(result.Z_train_augmented) == len(X_train_augmented), \
            f"Encoder injected synthetic rows: expected {len(X_train_augmented)} rows, " \
            f"got {len(result.Z_train_augmented)} (LSA violation)"
        assert len(result.Z_val) == len(X_val), \
            f"Val split row count changed: expected {len(X_val)}, got {len(result.Z_val)}"
        assert len(result.Z_test) == len(X_test), \
            f"Test split row count changed: expected {len(X_test)}, got {len(result.Z_test)}"

    def test_encoder_uses_mean_path_deterministic_encoding(
        self, trained_encoder_path, augmented_splits
    ):
        """Encoder uses μ (mean) path, not sampled z — results are deterministic.

        Calling encode() twice with same inputs must produce identical Z vectors.
        """
        X_train_augmented, y_train_augmented, X_val, X_test = augmented_splits

        encoder = LatentEncoder(
            encoder_path=trained_encoder_path,
            latent_dim=4,
        )

        result1 = encoder.encode(X_train_augmented, y_train_augmented, X_val, X_test)
        result2 = encoder.encode(X_train_augmented, y_train_augmented, X_val, X_test)

        # Assert deterministic encoding (μ path, not stochastic sampling)
        assert np.allclose(result1.Z_train_augmented, result2.Z_train_augmented), \
            "Z_train_augmented encoding is stochastic (should use μ, not sampled z)"
        assert np.allclose(result1.Z_val, result2.Z_val), \
            "Z_val encoding is stochastic"
        assert np.allclose(result1.Z_test, result2.Z_test), \
            "Z_test encoding is stochastic"


class TestDriftReference:
    """Boundary tests for drift reference artifact saved by encode stage (T096)."""

    @pytest.fixture(scope="module")
    def encoder_path(self, tmp_path_factory):
        from src.train_vae.vae_trainer import Encoder

        out = tmp_path_factory.mktemp("models")
        path = out / "vae_encoder.pth"
        enc = Encoder(input_dim=10, encoder_dims=[16, 8], latent_dim=4)
        torch.save(
            {"state_dict": enc.state_dict(), "input_dim": 10,
             "encoder_dims": [16, 8], "latent_dim": 4},
            path,
        )
        return path

    @pytest.fixture
    def X_train(self):
        np.random.seed(0)
        return np.random.randn(500, 10).astype(np.float32)

    @pytest.fixture
    def splits(self):
        np.random.seed(1)
        return (
            np.random.randn(250, 10).astype(np.float32),  # X_train_augmented
            np.zeros(250, dtype=np.int64),                 # y_train_augmented
            np.random.randn(50, 10).astype(np.float32),   # X_val
            np.random.randn(50, 10).astype(np.float32),   # X_test
        )

    def test_drift_reference_file_created_with_correct_keys(
        self, encoder_path, X_train, splits, tmp_path
    ):
        """encode() saves drift_reference.npz with required keys when DriftConfig provided."""
        from src.config import DriftConfig
        from src.metrics import compute_reference_sample_size

        ref_path = str(tmp_path / "drift_reference.npz")
        drift = DriftConfig(reference_path=ref_path, random_state=42, min_ratio=3.0)

        X_train_augmented, y_train_augmented, X_val, X_test = splits
        enc = LatentEncoder(encoder_path=encoder_path, latent_dim=4, drift=drift)
        result = enc.encode(X_train_augmented, y_train_augmented, X_val, X_test, X_train=X_train)

        assert result.drift_reference_path == ref_path
        ref = np.load(ref_path)
        assert set(ref.files) == {"Z_ref", "Z_mean", "Z_std", "n_samples", "latent_dim"}

        latent_dim = int(ref["latent_dim"])
        n_samples = int(ref["n_samples"])
        assert latent_dim == 4
        assert ref["Z_ref"].shape == (n_samples, latent_dim)
        assert ref["Z_mean"].shape == (latent_dim,)
        assert ref["Z_std"].shape == (latent_dim,)

        n_required = compute_reference_sample_size(len(X_train), X_train.shape[1], min_ratio=3.0)
        assert n_samples >= n_required

    def test_drift_reference_skipped_when_drift_config_absent(
        self, encoder_path, splits
    ):
        """encode() skips drift artifact when drift=None (no config provided)."""
        X_train_augmented, y_train_augmented, X_val, X_test = splits
        enc = LatentEncoder(encoder_path=encoder_path, latent_dim=4, drift=None)
        result = enc.encode(X_train_augmented, y_train_augmented, X_val, X_test)
        assert result.drift_reference_path is None

    def test_drift_reference_drawn_from_X_train_not_augmented(
        self, encoder_path, X_train, splits, tmp_path
    ):
        """Z_ref row count <= len(X_train) — reference drawn from real X_train only."""
        from src.config import DriftConfig

        ref_path = str(tmp_path / "drift_reference.npz")
        drift = DriftConfig(reference_path=ref_path, random_state=42, min_ratio=3.0)

        X_train_augmented, y_train_augmented, X_val, X_test = splits
        enc = LatentEncoder(encoder_path=encoder_path, latent_dim=4, drift=drift)
        enc.encode(X_train_augmented, y_train_augmented, X_val, X_test, X_train=X_train)

        ref = np.load(ref_path)
        assert int(ref["n_samples"]) <= len(X_train)
