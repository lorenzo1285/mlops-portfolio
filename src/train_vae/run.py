"""Train VAE stage - Denoising β-VAE training on full X (unsupervised)."""
import os
import sys

import mlflow
import numpy as np

from src.config import load_config
from src.train_vae.vae_trainer import DVAETrainer
from src.utils import get_run_name


def main() -> None:
    """Train VAE on concatenated X_train + X_val + X_test (no Y)."""
    try:
        # Load config
        config = load_config()
        
        # Read environment variables with defaults
        train_x_path = os.getenv(
            "TRAIN_X_PATH",
            os.path.join(config.data.processed_dir, "X_train.npy"),
        )
        val_x_path = os.getenv(
            "VAL_X_PATH",
            os.path.join(config.data.processed_dir, "X_val.npy"),
        )
        test_x_path = os.getenv(
            "TEST_X_PATH",
            os.path.join(config.data.processed_dir, "X_test.npy"),
        )
        encoder_output_path = os.getenv(
            "ENCODER_OUTPUT_PATH",
            "models/vae_encoder.pth",
        )
        decoder_output_path = os.getenv(
            "DECODER_OUTPUT_PATH",
            "models/vae_decoder.pth",
        )
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)
        
        # Load X arrays
        X_train = np.load(train_x_path)
        X_val = np.load(val_x_path)
        X_test = np.load(test_x_path)
        
        # Concatenate all X (unsupervised - no Y needed)
        X_all = np.vstack([X_train, X_val, X_test])
        
        print(f"Train VAE: {len(X_all)} samples, {X_all.shape[1]} features")
        print(f"  X_train: {len(X_train)}, X_val: {len(X_val)}, X_test: {len(X_test)}")
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create trainer and train
        trainer = DVAETrainer(config.vae, config.mlflow, run_name=get_run_name("train-vae"))
        result = trainer.train(X_all)
        
        # Report results
        print(f"VAE training complete:")
        print(f"  Best epoch: {result.best_epoch}")
        print(f"  Final ELBO: {result.final_elbo:.4f}")
        print(f"  Encoder saved: {result.encoder_path}")
        print(f"  Decoder saved: {result.decoder_path}")
        print(f"  MLflow run: {result.run_id}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: VAE training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
