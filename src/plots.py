from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    artifact_name: str = "confusion_matrix.png",
) -> None:
    """Log a confusion matrix PNG to the active MLflow run."""
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix — Test Set")
    mlflow.log_figure(fig, artifact_name)
    plt.close(fig)


def log_roc_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    artifact_name: str = "roc_curve.png",
) -> None:
    """Log a one-vs-rest ROC curve PNG and per-class AUC metrics to the active MLflow run.

    Skips any class with fewer than 2 unique labels in y_true (not enough support
    for a meaningful curve — logs AUC=0.0 for that class instead of crashing).
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, name in enumerate(class_names):
        col = y_bin[:, i]
        if len(np.unique(col)) < 2:
            # Not enough positive samples to compute a curve
            mlflow.log_metric(f"roc_auc_{name.lower()}", 0.0)
            continue
        fpr, tpr, _ = roc_curve(col, probs[:, i])
        roc_auc = auc(fpr, tpr)
        mlflow.log_metric(f"roc_auc_{name.lower()}", roc_auc)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — One-vs-Rest, Test Set")
    ax.legend(loc="lower right")
    mlflow.log_figure(fig, artifact_name)
    plt.close(fig)
