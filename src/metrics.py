"""MLflow evaluation dataset helpers for mlflow.evaluate()."""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd


def make_eval_dataset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str] | None = None,
) -> mlflow.data.Dataset:
    """Create an MLflow evaluation dataset from predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray
        Model predictions (0 or 1 after thresholding).
    feature_names : list[str], optional
        Feature column names. If None, uses generic names.

    Returns
    -------
    mlflow.data.Dataset
        Dataset compatible with mlflow.evaluate().

    Notes
    -----
    This is a pure data preparation function — no MLflow logging occurs here.
    All logging is the caller's responsibility.
    """
    # Create DataFrame for eval dataset
    data = {
        "predictions": y_pred,
        "targets": y_true,
    }
    
    df = pd.DataFrame(data)
    
    # Convert to MLflow dataset
    return mlflow.data.from_pandas(df, targets="targets")
