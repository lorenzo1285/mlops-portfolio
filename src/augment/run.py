"""Augment stage - CTGAN synthetic Fatal class generation."""
import os
import sys

import mlflow
import numpy as np

from src.augment.augmenter import CTGANAugmenter
from src.config import load_config


def main() -> None:
    """Generate CTGAN synthetic Fatal samples to reach target class balance."""
    try:
        # Load config
        config = load_config()
        
        # Read environment variables with defaults
        x_train_path = os.getenv(
            "X_TRAIN_PATH",
            os.path.join(config.data.processed_dir, "X_train.npy"),
        )
        y_train_path = os.getenv(
            "Y_TRAIN_PATH",
            os.path.join(config.data.processed_dir, "y_train.npy"),
        )
        x_aug_output = os.getenv(
            "X_AUG_OUTPUT",
            os.path.join(config.data.processed_dir, "X_train_augmented.npy"),
        )
        y_aug_output = os.getenv(
            "Y_AUG_OUTPUT",
            os.path.join(config.data.processed_dir, "y_train_augmented.npy"),
        )
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Load training data
        X_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        
        fatal_fraction_before = (y_train == 2).mean()
        print(f"Augment: {len(X_train)} train rows; Fatal fraction before: {fatal_fraction_before:.3f}")
        print(f"  Target Fatal ratio: {config.augment.target_fatal_ratio}")
        print(f"  TVAE epochs: {config.augment.tvae_epochs}")
        
        # Instantiate augmenter and generate synthetic Fatal samples
        augmenter = CTGANAugmenter(config.augment)
        result = augmenter.augment(X_train, y_train)
        
        # Save augmented arrays
        output_dir = os.path.dirname(x_aug_output)
        os.makedirs(output_dir, exist_ok=True)
        np.save(x_aug_output, result.X_augmented)
        np.save(y_aug_output, result.y_augmented)
        
        fatal_fraction_after = (result.y_augmented == 2).mean()
        
        print(
            f"Augment complete: {len(result.X_augmented)} rows "
            f"({result.n_synthetic} synthetic Fatal); "
            f"Fatal fraction after: {fatal_fraction_after:.3f}"
        )
        
        # Log metadata to MLflow (no experiment set — logged to default)
        with mlflow.start_run():
            mlflow.log_params({
                "tvae_epochs": config.augment.tvae_epochs,
                "target_fatal_ratio": config.augment.target_fatal_ratio,
                "random_state": config.augment.random_state,
            })
            mlflow.log_metrics({
                "n_real_fatal": result.n_real_fatal,
                "n_synthetic": result.n_synthetic,
                "fatal_fraction_before": fatal_fraction_before,
                "fatal_fraction_after": fatal_fraction_after,
                "n_train_before": len(X_train),
                "n_train_after": len(result.X_augmented),
            })
        
        sys.exit(0)
        
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Augment stage failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
