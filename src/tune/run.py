"""Tune stage — Optuna HPO for VAE hyperparameters."""
import json
import os
import sys
import traceback

import yaml

from src.config import load_config
from src.tune.optuna_tuner import OptunaTuner


def main() -> None:
    """Run Optuna HPO; write best params to params.yaml."""
    try:
        config = load_config()

        evaluation_report_path = os.getenv(
            "EVALUATION_REPORT_PATH",
            "docs/evaluation_report.json",
        )
        params_path = os.getenv("PARAMS_PATH", "params.yaml")
        optuna_output_path = os.getenv(
            "OPTUNA_OUTPUT_PATH",
            "docs/optuna_best_params.json",
        )
        data_dir = os.getenv("DATA_DIR", "data")

        with open(evaluation_report_path) as f:
            report = json.load(f)

        gates_passed = report["gates_passed"]

        if gates_passed:
            print("Gates PASSED — skipping HPO (model is already good enough)")
            with open(optuna_output_path, "w") as f:
                json.dump({"skipped": True, "reason": "gates_passed"}, f, indent=2)
            sys.exit(0)

        winner = report["winner"]
        winner_f1_key = "ml_mean_f1" if winner == "ml" else "dl_mean_f1"
        winner_recall_key = "ml_mean_fatal_recall" if winner == "ml" else "dl_mean_fatal_recall"
        print(f"Gates FAILED — running Optuna HPO to improve VAE representation")
        print(f"  Winner: {winner}")
        print(
            f"  Current val metrics: F1={report.get(winner_f1_key, 0):.4f}, "
            f"Fatal recall={report.get(winner_recall_key, 0):.4f}"
        )

        tuner = OptunaTuner(
            tune_config=config.tune,
            vae_config=config.vae,
            mlflow_config=config.mlflow,
            data_dir=data_dir,
            model_config=config.model,
        )

        result = tuner.tune()

        # Surgical update — only touch the 5 tuned params + dl.input_dim
        with open(params_path) as f:
            params = yaml.safe_load(f)

        params.setdefault("vae", {})
        params["vae"]["beta_max"] = result.best_params["beta_max"]
        params["vae"]["latent_dim"] = result.best_params["latent_dim"]
        params["vae"]["warmup_epochs"] = result.best_params["warmup_epochs"]
        params["vae"]["lr"] = result.best_params["lr"]
        params["vae"]["dropout_p"] = result.best_params["dropout_p"]

        params.setdefault("dl", {})
        params["dl"]["input_dim"] = result.best_params["latent_dim"]

        with open(params_path, "w") as f:
            yaml.dump(params, f, default_flow_style=None, sort_keys=False)

        best_params_output = {
            **result.best_params,
            "val_fitness": result.best_value,
            "n_trials": result.n_trials,
            "winner": winner,
        }

        with open(optuna_output_path, "w") as f:
            json.dump(best_params_output, f, indent=2)

        print(f"\nTune complete:")
        for k, v in result.best_params.items():
            print(f"  Best {k}: {v}")
        print(f"  Best val_fitness: {result.best_value:.4f}")
        print(f"  Trials: {result.n_trials}")
        print(f"  Updated: {params_path}")
        print(f"  Output: {optuna_output_path}")

        sys.exit(0)

    except Exception as e:
        print(f"ERROR: Tune stage failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
