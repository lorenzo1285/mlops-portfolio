"""Tests for drift detection - MMD-based latent space drift detector."""
import numpy as np
import pytest
from pathlib import Path

from src.drift.detector import DriftDetector, DriftResult


class TestDriftDetector:
    """Boundary tests for MMD-based drift detection."""

    # --- fixtures ---

    @pytest.fixture(scope="module")
    def drift_reference_path(self, tmp_path_factory):
        """Create a mock drift_reference.npz file.
        
        Uses a simple 2D latent space with reference samples drawn from
        N(0, 1) to make in-distribution vs OOD tests easy to construct.
        """
        output_dir = tmp_path_factory.mktemp("drift_data")
        ref_path = output_dir / "drift_reference.npz"
        
        # Create reference samples: 100 samples, 2D latent space
        rng = np.random.default_rng(42)
        Z_ref = rng.standard_normal((100, 2))
        
        # Save with the structure created by LatentEncoder._save_drift_reference
        np.savez(
            ref_path,
            Z_ref=Z_ref,
            Z_mean=Z_ref.mean(axis=0),
            Z_std=Z_ref.std(axis=0),
            n_samples=np.array(100),
            latent_dim=np.array(2),
        )
        
        return str(ref_path)

    @pytest.fixture
    def detector(self, drift_reference_path):
        """Instantiate DriftDetector with mock reference."""
        return DriftDetector(reference_path=drift_reference_path)

    # --- tests ---

    def test_loads_reference_at_construction(self, drift_reference_path):
        """DriftDetector loads drift_reference.npz at construction."""
        detector = DriftDetector(reference_path=drift_reference_path)
        
        # Should have loaded the reference data
        assert hasattr(detector, '_Z_ref')
        assert detector._Z_ref.shape == (100, 2)

    def test_detect_returns_drift_result(self, detector):
        """detect() returns DriftResult with all required fields."""
        rng = np.random.default_rng(43)
        Z_new = rng.standard_normal((50, 2))
        
        result = detector.detect(Z_new)
        
        # DriftResult must have these fields
        assert isinstance(result, DriftResult)
        assert hasattr(result, 'mmd')
        assert hasattr(result, 'bandwidth')
        assert hasattr(result, 'is_drifted')
        assert hasattr(result, 'threshold')
        
        # All fields should be numeric or bool
        assert isinstance(result.mmd, (float, np.floating))
        assert isinstance(result.bandwidth, (float, np.floating))
        assert isinstance(result.is_drifted, (bool, np.bool_))
        assert isinstance(result.threshold, (float, np.floating))

    def test_no_drift_when_same_distribution(self, detector):
        """is_drifted=False when Z_new sampled from same distribution as Z_ref.
        
        Reference is N(0, 1). New samples also from N(0, 1).
        MMD should be small → is_drifted=False.
        """
        rng = np.random.default_rng(44)
        Z_new = rng.standard_normal((50, 2))  # Same distribution as Z_ref
        
        result = detector.detect(Z_new)
        
        assert result.is_drifted is False, (
            f"Expected no drift for same distribution, but is_drifted={result.is_drifted}. "
            f"MMD={result.mmd:.6f}, threshold={result.threshold:.6f}"
        )

    def test_drift_when_out_of_distribution(self, detector):
        """is_drifted=True when Z_new is clearly out-of-distribution.
        
        Reference is N(0, 1). New samples from N(10, 1) — mean shifted by 10 std devs.
        MMD should be large → is_drifted=True.
        """
        rng = np.random.default_rng(45)
        Z_new = rng.standard_normal((50, 2)) + 10.0  # Mean shifted to 10
        
        result = detector.detect(Z_new)
        
        assert result.is_drifted is True, (
            f"Expected drift for OOD samples, but is_drifted={result.is_drifted}. "
            f"MMD={result.mmd:.6f}, threshold={result.threshold:.6f}"
        )

    def test_bandwidth_computed_via_median_heuristic(self, detector):
        """Bandwidth estimated via median heuristic on Z_ref.
        
        Median heuristic: bandwidth = median(pairwise_distances(Z_ref)) / sqrt(2)
        Should be > 0 and roughly O(1) for standard normal data.
        """
        rng = np.random.default_rng(46)
        Z_new = rng.standard_normal((30, 2))
        
        result = detector.detect(Z_new)
        
        # Bandwidth should be positive and reasonable for unit variance data
        assert result.bandwidth > 0, "Bandwidth must be positive"
        assert 0.5 < result.bandwidth < 5.0, (
            f"Bandwidth {result.bandwidth:.3f} outside expected range for N(0,1) data"
        )

    def test_mmd_is_nonnegative(self, detector):
        """MMD is a divergence metric — must be >= 0."""
        rng = np.random.default_rng(47)
        Z_new = rng.standard_normal((40, 2))
        
        result = detector.detect(Z_new)
        
        assert result.mmd >= 0, f"MMD must be non-negative, got {result.mmd}"

    def test_threshold_is_positive(self, detector):
        """Threshold must be > 0."""
        rng = np.random.default_rng(48)
        Z_new = rng.standard_normal((40, 2))
        
        result = detector.detect(Z_new)
        
        assert result.threshold > 0, f"Threshold must be positive, got {result.threshold}"
