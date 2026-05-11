from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_recall_fscore_support


def make_eval_dataset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str] | None = None,
) -> mlflow.data.pandas_dataset.PandasDataset:
    df = pd.DataFrame({
        "label": y_true.astype(int),
        "prediction": y_pred.astype(int),
    })
    return mlflow.data.from_pandas(df, targets="label", predictions="prediction")


def per_class_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    return {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }


def cross_entropy_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean cross-entropy loss from integer class labels and predicted probabilities.

    Args:
        y_true:  Integer class labels, shape (N,).
        y_proba: Predicted class probabilities, shape (N, n_classes).

    Returns:
        Scalar mean cross-entropy (nats).
    """
    return float(log_loss(y_true, y_proba))


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    n = len(y)
    weights = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        count = (y == c).sum()
        weights[c] = n / (n_classes * count) if count > 0 else 0.0
    return weights


def compute_reference_sample_size(
    n_total: int, n_features: int, min_ratio: float = 3.0
) -> int:
    """Minimum reference set size for MMD to have adequate statistical power.

    Returns min(n_total, max(n_features * min_ratio * 100, 1000)).
    For latent_dim=16, min_ratio=3.0 → 4800 samples (floor 1000, cap n_total).
    """
    n_required = int(n_features * min_ratio * 100)
    return min(n_total, max(n_required, 1000))


def focal_loss_grad_hess(
    gamma: float = 2.0,
    alpha: np.ndarray | None = None,
) -> callable:
    """Multi-class focal loss for XGBoost (Softmax).

    FL(p) = -α (1-p)ᵞ log(p)

    Args:
        gamma: Focusing parameter. Higher gamma = more focus on hard samples.
        alpha: Class weights (N_classes,). Defaults to 1.0 for all classes.

    Returns:
        A callable compatible with XGBoost's objective parameter.
    """

    def focal_loss_obj(y_true: np.ndarray, y_pred: np.ndarray, sample_weight=None):
        # XGBoost may pass y_pred as flat (N*K,) or 2D (N, K) depending on version
        n_samples = len(y_true)
        if y_pred.ndim == 1:
            n_classes = len(y_pred) // n_samples
            y_pred = y_pred.reshape(n_samples, n_classes)
        else:
            n_classes = y_pred.shape[1]

        # Softmax to get probabilities
        exp_preds = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

        # One-hot encode y_true
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(n_samples), y_true.astype(int)] = 1.0

        # Focal Loss derivatives
        # Gradient: ∂FL/∂z = (1-p)ᵞ [ γ p log(p) + p - 1 ] ... simplified for softmax
        # We use the standard softmax gradient (p - y) and scale it by the focal term
        # (1-p)ᵞ is the modulating factor
        
        # Predicted probability for the true class
        pt = np.sum(y_onehot * probs, axis=1, keepdims=True)
        
        # Modulating factor
        mod = (1.0 - pt) ** gamma
        
        # Gradient
        grad = mod * (probs - y_onehot)
        
        # Add gamma term for harder samples
        # ∂FL/∂z = mod * (probs - y_onehot) * (1 + gamma * (1-pt) * log(pt)) is complex
        # A common robust approximation used in practice for multi-class XGBoost:
        if gamma > 0:
            # Scale gradient by focusing term
            grad = mod * (probs - y_onehot) * (gamma * (1 - pt) + 1)

        # Hessian (approximation)
        # Standard softmax hessian is p(1-p), scaled by modulation
        hess = mod * probs * (1.0 - probs) * (gamma * (1 - pt) + 1)
        
        if alpha is not None:
            # Apply class weights
            weights = alpha[y_true.astype(int)].reshape(-1, 1)
            grad *= weights
            hess *= weights

        # XGBoost 2.1.0+ requires (n_samples, n_classes) shape
        return grad, hess

    return focal_loss_obj
