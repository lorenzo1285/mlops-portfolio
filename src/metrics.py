from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd


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
