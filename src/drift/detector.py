"""MMD-based latent space drift detector."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DriftResult:
    """Result of drift detection.
    
    Attributes:
        mmd: Maximum Mean Discrepancy between reference and new samples.
        bandwidth: RBF kernel bandwidth (estimated via median heuristic).
        is_drifted: True if MMD exceeds threshold (drift detected).
        threshold: Drift detection threshold.
    """
    mmd: float
    bandwidth: float
    is_drifted: bool
    threshold: float


class DriftDetector:
    """MMD-based drift detector for latent space monitoring.
    
    Loads a reference distribution from drift_reference.npz and computes
    Maximum Mean Discrepancy (MMD) with RBF kernel to detect distribution shift.
    
    Bandwidth is estimated via median heuristic on reference samples.
    Threshold is set to a fixed multiple of bandwidth for portfolio demonstration
    (production would use permutation testing for 95th percentile null).
    """

    def __init__(self, reference_path: str):
        """Load drift reference distribution.
        
        Args:
            reference_path: Path to drift_reference.npz created by LatentEncoder.
                Must contain keys: Z_ref, Z_mean, Z_std, n_samples, latent_dim.
        """
        ref_data = np.load(reference_path)
        self._Z_ref = ref_data['Z_ref']
        self._Z_mean = ref_data['Z_mean']
        self._Z_std = ref_data['Z_std']
        self._n_samples = int(ref_data['n_samples'])
        self._latent_dim = int(ref_data['latent_dim'])
        
        # Precompute bandwidth via median heuristic
        self._bandwidth = self._compute_bandwidth_median_heuristic(self._Z_ref)
        
        # Fixed threshold heuristic: 0.5 * bandwidth
        # (production would use permutation null 95th percentile)
        # More conservative than 3.0x to ensure OOD detection sensitivity
        self._threshold = 0.5 * self._bandwidth

    def detect(self, Z_new: np.ndarray) -> DriftResult:
        """Detect drift via MMD with RBF kernel.
        
        Args:
            Z_new: New latent samples, shape (n_new, latent_dim).
        
        Returns:
            DriftResult with mmd, bandwidth, is_drifted, threshold.
        """
        # Subsample large datasets for computational efficiency
        # MMD is O(n^2), so cap at 5000 samples for new data
        if len(Z_new) > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(Z_new), size=5000, replace=False)
            Z_new_sample = Z_new[idx]
        else:
            Z_new_sample = Z_new
        
        mmd = self._compute_mmd(self._Z_ref, Z_new_sample, self._bandwidth)
        is_drifted = bool(mmd > self._threshold)
        
        return DriftResult(
            mmd=float(mmd),
            bandwidth=float(self._bandwidth),
            is_drifted=is_drifted,
            threshold=float(self._threshold),
        )

    def _compute_bandwidth_median_heuristic(self, Z: np.ndarray) -> float:
        """Estimate RBF bandwidth via median heuristic.
        
        bandwidth = median(pairwise_distances) / sqrt(2)
        
        Args:
            Z: Reference samples, shape (n, d).
        
        Returns:
            Bandwidth scalar > 0.
        """
        # Compute pairwise squared distances efficiently
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        sq_norms = np.sum(Z**2, axis=1, keepdims=True)  # (n, 1)
        pairwise_sq_dists = sq_norms + sq_norms.T - 2 * Z @ Z.T  # (n, n)
        
        # Avoid numerical issues with negative values near zero
        pairwise_sq_dists = np.maximum(pairwise_sq_dists, 0)
        
        # Extract upper triangle (exclude diagonal = self-distances)
        triu_idx = np.triu_indices_from(pairwise_sq_dists, k=1)
        pairwise_dists = np.sqrt(pairwise_sq_dists[triu_idx])
        
        # Median heuristic
        median_dist = np.median(pairwise_dists)
        bandwidth = median_dist / np.sqrt(2)
        
        # Ensure positive bandwidth (fallback if all samples identical)
        return max(bandwidth, 1e-6)

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
        """Compute Maximum Mean Discrepancy with RBF kernel.
        
        MMD^2 = E[k(X, X')] + E[k(Y, Y')] - 2E[k(X, Y)]
        
        where k(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
        
        Args:
            X: Reference samples, shape (n_x, d).
            Y: New samples, shape (n_y, d).
            bandwidth: RBF kernel bandwidth > 0.
        
        Returns:
            MMD statistic >= 0.
        """
        n_x = len(X)
        n_y = len(Y)
        
        # Compute kernel matrices
        K_XX = self._rbf_kernel(X, X, bandwidth)
        K_YY = self._rbf_kernel(Y, Y, bandwidth)
        K_XY = self._rbf_kernel(X, Y, bandwidth)
        
        # MMD^2 = E[K_XX] + E[K_YY] - 2E[K_XY]
        # Use unbiased estimator: exclude diagonal for K_XX and K_YY
        E_K_XX = (K_XX.sum() - np.trace(K_XX)) / (n_x * (n_x - 1))
        E_K_YY = (K_YY.sum() - np.trace(K_YY)) / (n_y * (n_y - 1))
        E_K_XY = K_XY.mean()
        
        mmd_sq = E_K_XX + E_K_YY - 2 * E_K_XY
        
        # MMD should be >= 0 (divergence), but numerical issues can cause small negatives
        return max(float(np.sqrt(max(mmd_sq, 0))), 0.0)

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
        """RBF (Gaussian) kernel matrix.
        
        K[i, j] = exp(-||X[i] - Y[j]||^2 / (2 * bandwidth^2))
        
        Args:
            X: Samples, shape (n_x, d).
            Y: Samples, shape (n_y, d).
            bandwidth: Kernel bandwidth > 0.
        
        Returns:
            Kernel matrix, shape (n_x, n_y).
        """
        # Pairwise squared distances: ||X[i] - Y[j]||^2
        X_sq = np.sum(X**2, axis=1, keepdims=True)  # (n_x, 1)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True)  # (n_y, 1)
        sq_dists = X_sq + Y_sq.T - 2 * X @ Y.T  # (n_x, n_y)
        
        # Avoid negative values from numerical errors
        sq_dists = np.maximum(sq_dists, 0)
        
        # RBF kernel
        gamma = 1.0 / (2 * bandwidth**2)
        return np.exp(-gamma * sq_dists)
