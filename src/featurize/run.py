import os
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.config import load_config
from src.featurize.featurizer import Featurizer


def main() -> None:
    config = load_config()
    input_path = os.getenv("INPUT_PATH", config.data.processed_dir + "raw.csv")
    output_dir = os.getenv("OUTPUT_DIR", config.data.processed_dir)
    pipeline_path = os.getenv("PIPELINE_PATH", "models/preprocessing_pipeline.joblib")

    df = pd.read_csv(input_path, low_memory=False)
    initial_rows = len(df)

    result = Featurizer(
        feature_cols=config.features.columns,
        numeric_cols=config.features.numeric_columns,
        target_col=config.features.target_column,
        train_size=config.data.train_size,
        val_size=config.data.val_size,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        sentinel_value=config.data.sentinel_value,
        sentinel_cols=config.features.sentinel_columns,
        ordinal_cols=config.features.ordinal_columns,
    ).fit_transform(df)

    total_out = len(result.y_train) + len(result.y_val) + len(result.y_test)
    drop_pct = (initial_rows - total_out) / initial_rows
    if drop_pct > 0.05:
        print(f"ERROR: Dropped {drop_pct:.1%} of rows (threshold 5%)", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    for name, arr in [
        ("X_train", result.X_train), ("X_val", result.X_val), ("X_test", result.X_test),
        ("y_train", result.y_train), ("y_val", result.y_val), ("y_test", result.y_test),
    ]:
        np.save(os.path.join(output_dir, f"{name}.npy"), arr)

    os.makedirs(os.path.dirname(pipeline_path) or ".", exist_ok=True)
    joblib.dump(
        {"pipeline": result.preprocessor, "feature_cols": result.feature_cols},
        pipeline_path,
    )

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    with mlflow.start_run(run_name="featurize"):
        mlflow.log_params({
            "n_samples": total_out,
            "n_features": result.X_train.shape[1],
            "mlp_n_params": result.n_params,
            "samples_per_param_ratio": round(result.samples_per_param_ratio, 4),
        })

    print(
        f"Featurize: {len(result.y_train)} train / {len(result.y_val)} val / "
        f"{len(result.y_test)} test | {result.X_train.shape[1]} features | "
        f"ratio={result.samples_per_param_ratio:.2f}"
    )

    if result.samples_per_param_ratio < 3.0:
        print(
            f"ERROR: samples_per_param_ratio={result.samples_per_param_ratio:.2f} < 3.0 — "
            "MLP architecture must be reviewed before training proceeds",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
