"""Tune stage - Katib HPO for VAE hyperparameters."""
import json
import os
import sys

import numpy as np
import yaml

from src.config import load_config
from src.tune.tuner import HyperparamTuner


def main() -> None:
    """Submit Katib Experiment; poll until complete; write best params to params.yaml."""
    try:
        # Load config
        config = load_config()
        
        # Read environment variables
        evaluation_report_path = os.getenv(
            "EVALUATION_REPORT_PATH",
            "docs/evaluation_report.json",
        )
        params_path = os.getenv("PARAMS_PATH", "params.yaml")
        
        # Load evaluation report to get winner
        with open(evaluation_report_path) as f:
            report = json.load(f)
        
        winner = report["winner"]
        gates_passed = report["gates_passed"]
        
        if gates_passed:
            print("Gates PASSED — skipping HPO (model is already good enough)")
            sys.exit(0)
        
        print(f"Gates FAILED — running Katib HPO to improve VAE representation")
        print(f"  Winner: {winner}")
        print(f"  Current val metrics: F1={report.get('ml_mean_f1' if winner == 'ml' else 'dl_mean_f1', 0):.4f}, "
              f"Fatal recall={report.get('ml_mean_fatal_recall' if winner == 'ml' else 'dl_mean_fatal_recall', 0):.4f}")
        
        # Dummy arrays (not used - trials read from PVC directly)
        X_train = np.zeros((1, 1))
        y_train = np.zeros(1)
        X_val = np.zeros((1, 1))
        y_val = np.zeros(1)
        
        # Create tuner and run
        tuner = HyperparamTuner(
            mlflow_config=config.mlflow,
            tune_config=config.tune,
            winner=winner,
        )
        
        result = tuner.tune(X_train, y_train, X_val, y_val)
        
        # Write best params to params.yaml
        with open(params_path) as f:
            params = yaml.safe_load(f)
        
        # Update vae section with best hyperparameters
        if "vae" not in params:
            params["vae"] = {}
        
        params["vae"]["beta_max"] = result.best_params["beta_max"]
        params["vae"]["latent_dim"] = result.best_params["latent_dim"]
        
        with open(params_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)
        
        print(f"\nTune complete:")
        print(f"  Best beta_max: {result.best_params['beta_max']}")
        print(f"  Best latent_dim: {result.best_params['latent_dim']}")
        print(f"  Best val_fitness: {result.best_value:.4f}")
        print(f"  Trials: {result.n_trials}")
        print(f"  Updated: {params_path}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR: Tune stage failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
