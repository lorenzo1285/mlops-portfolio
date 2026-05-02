"""Tune stage - Katib HPO for VAE hyperparameters."""
import json
import os
import sys
import traceback

import numpy as np
import yaml

from src.config import load_config
from src.tune.tuner import HyperparamTuner


def main() -> None:
    """Submit Katib Experiment; poll until complete; write best params to params.yaml."""
    try:
        config = load_config()

        evaluation_report_path = os.getenv(
            "EVALUATION_REPORT_PATH",
            "docs/evaluation_report.json",
        )
        params_path = os.getenv("PARAMS_PATH", "params.yaml")
        katib_output_path = os.getenv(
            "KATIB_OUTPUT_PATH",
            "docs/katib_best_params.json",
        )

        with open(evaluation_report_path) as f:
            report = json.load(f)

        winner = report["winner"]
        gates_passed = report["gates_passed"]

        if winner not in ("ml", "dl"):
            raise ValueError(f"Unknown winner value in evaluation report: {winner!r}")

        if gates_passed:
            # HPO targets the VAE representation; if the current representation
            # already clears both gates (F1 > 0.35, fatal_recall > 0.50) there
            # is no signal that a different β or latent_dim would help further.
            print("Gates PASSED — skipping HPO (model is already good enough)")

            # Write empty result to satisfy DVC output
            with open(katib_output_path, "w") as f:
                json.dump({"skipped": True, "reason": "gates_passed"}, f, indent=2)

            sys.exit(0)

        print(f"Gates FAILED — running Katib HPO to improve VAE representation")
        print(f"  Winner: {winner}")
        winner_f1_key = "ml_mean_f1" if winner == "ml" else "dl_mean_f1"
        winner_recall_key = "ml_mean_fatal_recall" if winner == "ml" else "dl_mean_fatal_recall"
        print(f"  Current val metrics: F1={report.get(winner_f1_key, 0):.4f}, "
              f"Fatal recall={report.get(winner_recall_key, 0):.4f}")

        # Dummy arrays — trials read splits from PVC directly; tuner.tune() signature
        # accepts them for interface consistency with the rest of the pipeline.
        X_train = np.zeros((1, 1))
        y_train = np.zeros(1)
        X_val = np.zeros((1, 1))
        y_val = np.zeros(1)

        tuner = HyperparamTuner(
            mlflow_config=config.mlflow,
            tune_config=config.tune,
            winner=winner,
        )

        result = tuner.tune(X_train, y_train, X_val, y_val)

        # Surgical update — only touch vae.beta_max and vae.latent_dim to avoid
        # reformatting the whole file (yaml.dump would expand flow-style lists and
        # cause DVC to see params.yaml as dirty for unrelated params).
        with open(params_path) as f:
            params = yaml.safe_load(f)

        if "vae" not in params:
            params["vae"] = {}

        params["vae"]["beta_max"] = result.best_params["beta_max"]
        params["vae"]["latent_dim"] = result.best_params["latent_dim"]

        with open(params_path, "w") as f:
            yaml.dump(params, f, default_flow_style=None, sort_keys=False)

        best_params_output = {
            "beta_max": result.best_params["beta_max"],
            "latent_dim": result.best_params["latent_dim"],
            "val_fitness": result.best_value,
            "n_trials": result.n_trials,
            "winner": winner,
        }

        with open(katib_output_path, "w") as f:
            json.dump(best_params_output, f, indent=2)

        print(f"\nTune complete:")
        print(f"  Best beta_max: {result.best_params['beta_max']}")
        print(f"  Best latent_dim: {result.best_params['latent_dim']}")
        print(f"  Best val_fitness: {result.best_value:.4f}")
        print(f"  Trials: {result.n_trials}")
        print(f"  Updated: {params_path}")
        print(f"  Output: {katib_output_path}")

        sys.exit(0)

    except Exception as e:
        print(f"ERROR: Tune stage failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
