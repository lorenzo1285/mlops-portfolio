"""Train GMM stage - per-class Gaussian Mixture classifier on Z-space."""
import os
import sys

import numpy as np

from src.config import load_config
from src.train_gmm.trainer import GMMTrainer


def main() -> None:
    """Train per-class GMM on latent vectors across N seeds; save best model."""
    try:
        config = load_config()

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

        Z_train = np.load(z_train_path)
        y_train = np.load(y_train_path)
        Z_val = np.load(z_val_path)
        y_val = np.load(y_val_path)
        Z_test = np.load(z_test_path)
        y_test = np.load(y_test_path)

        print(f"Train GMM: {len(Z_train)} train / {len(Z_val)} val / {len(Z_test)} test")
        print(f"  Latent dim: {Z_train.shape[1]}")
        print(f"  Seeds: {config.ab_test.seeds}")
        print(f"  n_components: {dict(config.gmm.n_components)}")
        print(f"  covariance_type: {config.gmm.covariance_type}")
        print(f"  fatal_prior_boost: {config.gmm.fatal_prior_boost}")

        trainer = GMMTrainer(
            gmm_config=config.gmm,
            model_config=config.model,
            mlflow_config=config.mlflow,
            ab_test_config=config.ab_test,
        )

        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)

        print(f"\nGMM training complete:")
        print(f"  Best seed: {result.best_seed}")
        print(f"  eout_macro_f1: {result.eout_macro_f1:.4f}")
        print(f"  eout_fatal_recall: {result.eout_fatal_recall:.4f}")
        print(f"  Model saved: {result.model_path}")
        print(f"  MLflow run: {result.run_id}")

        sys.exit(0)

    except Exception as e:
        print(f"ERROR: GMM training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
