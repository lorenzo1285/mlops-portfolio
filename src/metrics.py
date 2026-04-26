from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


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


def compute_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    n = len(y)
    weights = np.zeros(n_classes, dtype=float)
    for c in range(n_classes):
        count = (y == c).sum()
        weights[c] = n / (n_classes * count) if count > 0 else 0.0
    return weights
