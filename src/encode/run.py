"""Encode stage - Latent space encoding with LSA augmentation."""
import os
import sys

import numpy as np

from src.config import load_config
from src.encode.encoder import LatentEncoder


def main() -> None:
    """Encode train/val/test splits using frozen VAE encoder; apply LSA to Z_train only."""
    try:
        # Load config
        config = load_config()
        
        # Read environment variables with defaults
        encoder_path = os.getenv("ENCODER_PATH", "models/vae_encoder.pth")
        train_x_path = os.getenv(
            "TRAIN_X_PATH",
            os.path.join(config.data.processed_dir, "X_train.npy"),
        )
        train_y_path = os.getenv(
            "TRAIN_Y_PATH",
            os.path.join(config.data.processed_dir, "y_train.npy"),
        )
        val_x_path = os.getenv(
            "VAL_X_PATH",
            os.path.join(config.data.processed_dir, "X_val.npy"),
        )
        test_x_path = os.getenv(
            "TEST_X_PATH",
            os.path.join(config.data.processed_dir, "X_test.npy"),
        )
        output_dir = os.getenv("OUTPUT_DIR", config.data.processed_dir)
        
        # Load arrays
        X_train = np.load(train_x_path)
        y_train = np.load(train_y_path)
        X_val = np.load(val_x_path)
        X_test = np.load(test_x_path)
        
        print(f"Encode: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
        print(f"  Encoder: {encoder_path}")
        print(f"  Latent dim: {config.vae.latent_dim}")
        
        # Instantiate encoder and encode splits
        encoder = LatentEncoder(
            encoder_path=encoder_path,
            encode_config=config.encode,
            latent_dim=config.vae.latent_dim,
        )
        
        result = encoder.encode(X_train, y_train, X_val, X_test)
        
        # Calculate fatal fraction
        fatal_fraction = (result.y_train_augmented == 2).sum() / len(result.y_train_augmented)
        
        # Save output arrays
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "Z_train_augmented.npy"), result.Z_train_augmented)
        np.save(os.path.join(output_dir, "Z_val.npy"), result.Z_val)
        np.save(os.path.join(output_dir, "Z_test.npy"), result.Z_test)
        np.save(os.path.join(output_dir, "y_train_augmented.npy"), result.y_train_augmented)
        
        print(
            f"Encode complete: Z_train {result.Z_train_augmented.shape} | "
            f"Z_val {result.Z_val.shape} | Z_test {result.Z_test.shape}"
        )
        print(
            f"  Fatal class: {result.n_real_fatal} real + {result.n_synthetic} synthetic = "
            f"{fatal_fraction:.1%}"
        )
        
        sys.exit(0)
        
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Encode stage failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
