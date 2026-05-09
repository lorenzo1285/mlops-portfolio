from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import torch


@dataclass
class RegistryReceipt:
    model_name: str
    version: str
    alias: str
    run_id: str
    winner: str


class CrashSeverityPyfunc(mlflow.pyfunc.PythonModel):
    """Inference bundle: frozen VAE encoder + champion classifier.

    predict(X) path: featurized X → LatentEncoder (μ) → classifier → class labels
    """

    def __init__(self, model_metadata: dict[str, Any] | None = None) -> None:
        self._meta = model_metadata or {}

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from src.train_vae.vae_trainer import Encoder

        ckpt = torch.load(context.artifacts["encoder"], weights_only=True)
        encoder = Encoder(
            input_dim=ckpt["input_dim"],
            encoder_dims=ckpt["encoder_dims"],
            latent_dim=ckpt["latent_dim"],
        )
        encoder.load_state_dict(ckpt["state_dict"])
        encoder.eval()
        self._encoder = encoder

        winner = self._meta.get("winner", "ml")
        clf_path = context.artifacts["classifier"]

        if winner == "ml":
            with open(clf_path, "rb") as f:
                self._classifier = pickle.load(f)
        elif winner == "gmm":
            with open(clf_path, "rb") as f:
                self._classifier = pickle.load(f)
        else:
            from src.train_dl.trainer import ShallowMLP

            mlp = ShallowMLP(
                input_dim=self._meta["input_dim"],
                hidden_dim=self._meta["hidden_dim"],
                n_classes=self._meta["n_classes"],
                dropout_p=self._meta["dropout_p"],
            )
            ckpt = torch.load(clf_path, weights_only=True)
            mlp.load_state_dict(ckpt["model_state_dict"])
            mlp.eval()
            self._classifier = mlp

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input) -> np.ndarray:
        X = np.array(model_input, dtype=np.float32)
        with torch.no_grad():
            mu, _ = self._encoder(torch.tensor(X))
            Z = mu.numpy()

        winner = self._meta.get("winner", "ml")
        if winner in ("ml", "gmm"):
            return self._classifier.predict(Z)
        # winner == "dl" (PyTorch MLP)
        with torch.no_grad():
            return self._classifier(torch.tensor(Z)).argmax(dim=1).numpy()


class ModelRegistrar:
    """Promote the winning model's best-seed artifact to the MLflow Model Registry.

    Logs a CrashSeverityPyfunc bundle (encoder + classifier) as a new MLflow
    artifact, registers it as `model_name@champion`, and writes registry_receipt.json.

    Public interface
    ----------------
    register(winner, run_id, report_path, receipt_path,
             encoder_path, classifier_path, model_metadata) → RegistryReceipt
    """

    def __init__(self, mlflow_config) -> None:
        self._mlflow_config = mlflow_config

    def register(
        self,
        winner: str,
        run_id: str,
        report_path: str,
        receipt_path: str,
        encoder_path: str,
        classifier_path: str,
        model_metadata: dict[str, Any],
    ) -> RegistryReceipt:
        """Register winning model to MLflow registry with @champion alias.

        Raises:
            ValueError: If constitutional gates failed in evaluation report.
        """
        with open(report_path) as f:
            report = json.load(f)

        if not report.get("gates_passed", False):
            raise ValueError(
                "Constitutional gates FAILED - refusing to register model that "
                "does not meet quality thresholds"
            )

        mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)

        if winner == "ml":
            experiment_name = self._mlflow_config.experiment_name_ml
        elif winner == "dl":
            experiment_name = self._mlflow_config.experiment_name_dl
        elif winner == "gmm":
            experiment_name = self._mlflow_config.experiment_name_gmm
        else:
            raise ValueError(f"Unknown winner: {winner}")

        exp = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = exp.experiment_id if exp is not None else None

        code_paths = ["src/"] if Path("src/").exists() else None

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="register_champion",
        ) as reg_run:
            mlflow.log_params({"winner": winner, "champion_run_id": run_id})
            mlflow.pyfunc.log_model(
                artifact_path="crash_severity_model",
                python_model=CrashSeverityPyfunc(model_metadata=model_metadata),
                artifacts={
                    "encoder": str(encoder_path),
                    "classifier": str(classifier_path),
                },
                code_paths=code_paths,
            )
            pyfunc_run_id = reg_run.info.run_id

        model_version = mlflow.register_model(
            model_uri=f"runs:/{pyfunc_run_id}/crash_severity_model",
            name=self._mlflow_config.model_name,
        )

        client = mlflow.MlflowClient(
            tracking_uri=self._mlflow_config.tracking_uri
        )
        client.set_registered_model_alias(
            name=self._mlflow_config.model_name,
            alias="champion",
            version=model_version.version,
        )

        receipt = RegistryReceipt(
            model_name=self._mlflow_config.model_name,
            version=model_version.version,
            alias="champion",
            run_id=run_id,
            winner=winner,
        )

        Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
        with open(receipt_path, "w") as f:
            json.dump(asdict(receipt), f, indent=2)

        return receipt
