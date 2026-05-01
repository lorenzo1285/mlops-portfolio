"""Train ML stage - XGBoost on Z-space."""
import os
import shutil
import sys

import mlflow
import numpy as np

from src.config import load_config
from src.train_ml.trainer import MLTrainer


def main() -> None:
    """Train XGBoost on latent vectors across N seeds; save best model."""
    try:
        # Load config
        config = load_config()
        
        # Read environment variables with defaults
        z_train_path = os.getenv(
            "Z_TRAIN_PATH",
            os.path.join(config.data.processed_dir, "Z_train_augmented.npy"),
        )
        y_train_path = os.getenv(
            "Y_TRAIN_PATH",
            os.path.join(config.data.processed_dir, "y_train_augmented.npy"),
        )
        z_val_path = os.getenv(
            "Z_VAL_PATH",
            os.path.join(config.data.processed_dir, "Z_val.npy"),
        )
        y_val_path = os.getenv(
            "Y_VAL_PATH",
            os.path.join(config.data.processed_dir, "y_val.npy"),
        )
        z_test_path = os.getenv(
            "Z_TEST_PATH",
            os.path.join(config.data.processed_dir, "Z_test.npy"),
        )
        y_test_path = os.getenv(
            "Y_TEST_PATH",
            os.path.join(config.data.processed_dir, "y_test.npy"),
        )
        model_output_path = os.getenv("MODEL_OUTPUT_PATH", "models/best_ml_model.pkl")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)

        # Load arrays
        Z_train = np.load(z_train_path)
        y_train = np.load(y_train_path)
        Z_val = np.load(z_val_path)
        y_val = np.load(y_val_path)
        Z_test = np.load(z_test_path)
        y_test = np.load(y_test_path)
        
        print(f"Train ML: {len(Z_train)} train / {len(Z_val)} val / {len(Z_test)} test")
        print(f"  Latent dim: {Z_train.shape[1]}")
        print(f"  Seeds: {config.ab_test.seeds}")
        print(f"  Model: XGBoost (multi:softprob, {config.model.n_classes} classes)")
        
        # Set MLflow tracking
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name_ml)
        
        # Create trainer and train
        trainer = MLTrainer(
            mlflow_config=config.mlflow,
            model_config=config.model,
            seeds=config.ab_test.seeds,
        )
        
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Save best model to output path
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        shutil.copy(result.model_path, model_output_path)
        
        # Report results
        print(f"\nML training complete:")
        print(f"  Best seed: {result.best_seed}")
        print(f"  Best macro F1: {result.eout_macro_f1:.4f}")
        print(f"  Model saved: {model_output_path}")
        print(f"  MLflow run: {result.best_run_id}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: ML training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
