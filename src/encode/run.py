"""Encode stage - Pure projection to latent space (no augmentation)."""
import json
import os
import sys

import mlflow
import numpy as np

from src.config import load_config
from src.encode.encoder import LatentEncoder


def main() -> None:
    """Project augmented X splits to latent Z via frozen VAE encoder (pure projection)."""
    try:
        # Load config
        config = load_config()
        
        # Read environment variables with defaults
        encoder_path = os.getenv("ENCODER_PATH", "models/vae_encoder.pth")
        x_train_aug_path = os.getenv(
            "X_TRAIN_AUG_PATH",
            os.path.join(config.data.processed_dir, "X_train_augmented.npy"),
        )
        y_train_aug_path = os.getenv(
            "Y_TRAIN_AUG_PATH",
            os.path.join(config.data.processed_dir, "y_train_augmented.npy"),
        )
        x_val_path = os.getenv(
            "X_VAL_PATH",
            os.path.join(config.data.processed_dir, "X_val.npy"),
        )
        x_test_path = os.getenv(
            "X_TEST_PATH",
            os.path.join(config.data.processed_dir, "X_test.npy"),
        )
        output_dir = os.getenv("OUTPUT_DIR", config.data.processed_dir)
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)

        mlflow.set_tracking_uri(mlflow_tracking_uri)

        x_train_path = os.getenv(
            "X_TRAIN_PATH",
            os.path.join(config.data.processed_dir, "X_train.npy"),
        )

        # Load arrays
        X_train_augmented = np.load(x_train_aug_path)
        y_train_augmented = np.load(y_train_aug_path)
        X_val = np.load(x_val_path)
        X_test = np.load(x_test_path)
        X_train = np.load(x_train_path)

        print(f"Encode: {len(X_train_augmented)} train (augmented) / {len(X_val)} val / {len(X_test)} test")
        print(f"  Encoder: {encoder_path}")
        print(f"  Latent dim: {config.vae.latent_dim}")
        if config.drift is not None:
            print(f"  Drift reference: {config.drift.reference_path}")

        # Instantiate encoder and encode splits
        os.makedirs("models", exist_ok=True)
        encoder = LatentEncoder(
            encoder_path=encoder_path,
            latent_dim=config.vae.latent_dim,
            drift=config.drift,
        )

        result = encoder.encode(X_train_augmented, y_train_augmented, X_val, X_test, X_train=X_train)
        
        # Save output arrays
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "Z_train_augmented.npy"), result.Z_train_augmented)
        np.save(os.path.join(output_dir, "Z_val.npy"), result.Z_val)
        np.save(os.path.join(output_dir, "Z_test.npy"), result.Z_test)
        np.save(os.path.join(output_dir, "y_train_augmented.npy"), y_train_augmented)  # pass-through
        
        print(
            f"Encode complete: Z_train_augmented {result.Z_train_augmented.shape} | "
            f"Z_val {result.Z_val.shape} | Z_test {result.Z_test.shape}"
        )
        
        # Drift detection (advisory signal — never halts pipeline)
        if config.drift is not None:
            from src.drift.detector import DriftDetector

            print("Running drift detection on Z_train_augmented (in-distribution self-check)...")
            detector = DriftDetector(reference_path=config.drift.reference_path)
            drift_result = detector.detect(result.Z_train_augmented)

            # Log to MLflow
            with mlflow.start_run(run_name="encode_drift_check"):
                mlflow.log_metric("drift_mmd", drift_result.mmd)
                mlflow.log_metric("drift_bandwidth", drift_result.bandwidth)
                mlflow.log_metric("drift_threshold", drift_result.threshold)
                mlflow.log_metric("drift_detected", int(drift_result.is_drifted))

            # Write drift report
            os.makedirs("docs", exist_ok=True)
            drift_report = {
                "mmd": float(drift_result.mmd),
                "bandwidth": float(drift_result.bandwidth),
                "threshold": float(drift_result.threshold),
                "drift_detected": int(drift_result.is_drifted),
                "is_drifted": bool(drift_result.is_drifted),
            }
            with open("docs/drift_report.json", "w") as f:
                json.dump(drift_report, f, indent=2)

            print(
                f"Drift detection: MMD={drift_result.mmd:.6f}, "
                f"threshold={drift_result.threshold:.6f}, "
                f"is_drifted={drift_result.is_drifted}"
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
