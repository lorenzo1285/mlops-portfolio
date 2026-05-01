"""Register stage — promote champion model to MLflow Model Registry."""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import mlflow

from src.config import load_config
from src.register.registrar import ModelRegistrar


def main() -> None:
    config = load_config()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config.mlflow.tracking_uri)
    report_path = os.getenv("REPORT_PATH", "docs/evaluation_report.json")
    receipt_path = os.getenv("RECEIPT_PATH", "models/registry_receipt.json")
    encoder_path = os.getenv("ENCODER_PATH", "models/vae_encoder.pth")

    try:
        # Fast-fail: check gates before any MLflow query
        with open(report_path) as f:
            report_data = json.load(f)
        if not report_data.get("gates_passed", False):
            raise ValueError(
                "Constitutional gates FAILED - model does not meet quality thresholds"
            )
        winner = report_data["winner"]

        default_classifier = (
            "models/best_ml_model.pkl" if winner == "ml" else "models/mlp_model.pth"
        )
        classifier_path = os.getenv("CLASSIFIER_PATH", default_classifier)

        # Propagate env-var tracking URI into config so ModelRegistrar uses it
        mlflow_config = dataclasses.replace(config.mlflow, tracking_uri=tracking_uri)
        mlflow.set_tracking_uri(tracking_uri)

        exp_name = (
            mlflow_config.experiment_name_ml
            if winner == "ml"
            else mlflow_config.experiment_name_dl
        )
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            raise ValueError(f"Experiment '{exp_name}' not found")

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["metrics.eout_macro_f1 DESC"],
            max_results=1,
            output_format="pandas",
        )
        if runs.empty:
            raise ValueError(f"No finished runs in experiment '{exp_name}'")
        run_id = runs.iloc[0]["run_id"]

        model_metadata = {
            "winner": winner,
            "latent_dim": config.vae.latent_dim,
            "input_dim": config.dl.input_dim,
            "hidden_dim": config.dl.hidden_dim,
            "n_classes": config.model.n_classes,
            "dropout_p": config.dl.dropout_p,
        }

        print(f"Register: promoting {winner.upper()} champion to registry")
        print(f"  Run ID: {run_id}")
        print(f"  Model name: {mlflow_config.model_name}")

        registrar = ModelRegistrar(mlflow_config=mlflow_config)
        receipt = registrar.register(
            winner=winner,
            run_id=run_id,
            report_path=report_path,
            receipt_path=receipt_path,
            encoder_path=encoder_path,
            classifier_path=classifier_path,
            model_metadata=model_metadata,
        )

        print(f"\nSUCCESS: {receipt.model_name}@{receipt.alias} v{receipt.version}")
        print(f"  Receipt: {receipt_path}")

    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
