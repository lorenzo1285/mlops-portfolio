from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from kubernetes import client, config


_DEFAULT_YAML_PATH = Path(__file__).parents[2] / "k8s" / "katib" / "vae_experiment.yaml"
_POLL_INTERVAL_SECONDS = 10
_POLL_TIMEOUT_SECONDS = 14400  # 4 hours


@dataclass
class TuneResult:
    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    best_run_id: str


class HyperparamTuner:
    """Katib Bayesian HPO on the winning model family with MLflow per-trial logging.

    Submits a Katib Experiment CRD to Kubernetes; each trial runs as a pod
    executing src/tune/trial.py. Search space is defined in k8s/katib/*.yaml.
    For DL models, the MLP architecture is fixed by the prior EvoTorch NAS run.

    Public interface
    ----------------
    tune(X_train, y_train, X_val, y_val) → TuneResult
        Submits Katib Experiment; polls until Succeeded; returns best params.
    """

    def __init__(
        self,
        mlflow_config,
        tune_config,
        winner: str,
        yaml_path: Path | str | None = None,
    ) -> None:
        self._mlflow_config = mlflow_config
        self._tune_config = tune_config
        self._winner = winner
        self._yaml_path = Path(yaml_path) if yaml_path is not None else _DEFAULT_YAML_PATH

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TuneResult:
        """Submit Katib Experiment and poll until complete; return best params."""
        with open(self._yaml_path) as f:
            experiment_spec = yaml.safe_load(f)

        # Render {{winner}} placeholder before CRD submission — Katib does not
        # know about winner; HyperparamTuner resolves it at construction time.
        # Path matches the batch/v1 Job wrapper in vae_experiment.yaml:
        #   spec.trialTemplate.trialSpec.spec.template.spec.containers
        containers = (
            experiment_spec
            .get("spec", {})
            .get("trialTemplate", {})
            .get("trialSpec", {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        if not containers:
            raise ValueError(
                f"No containers found under spec.trialTemplate.trialSpec.spec.template.spec "
                f"in {self._yaml_path}"
            )
        for container in containers:
            if container.get("name") == "trial":
                container["command"] = [
                    cmd.replace("{{winner}}", self._winner)
                    for cmd in container.get("command", [])
                ]

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        custom_api = client.CustomObjectsApi()

        experiment_name = experiment_spec["metadata"]["name"]
        namespace = experiment_spec["metadata"]["namespace"]

        try:
            custom_api.create_namespaced_custom_object(
                group="kubeflow.org",
                version="v1beta1",
                namespace=namespace,
                plural="experiments",
                body=experiment_spec,
            )
            print(f"Submitted Katib experiment: {experiment_name}")
        except client.exceptions.ApiException as e:
            if e.status == 409:
                # Experiment already exists (e.g. prior run timed out locally);
                # skip creation and resume polling the in-progress experiment.
                print(f"Experiment {experiment_name} already exists — resuming poll")
            else:
                raise

        deadline = time.time() + _POLL_TIMEOUT_SECONDS
        while True:
            if time.time() > deadline:
                raise RuntimeError(
                    f"Katib experiment timed out after {_POLL_TIMEOUT_SECONDS}s"
                )

            experiment = custom_api.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v1beta1",
                namespace=namespace,
                plural="experiments",
                name=experiment_name,
            )

            status = experiment.get("status", {})
            conditions = status.get("conditions", [])

            for condition in conditions:
                if condition.get("type") == "Succeeded" and condition.get("status") == "True":
                    optimal_trial = status.get("currentOptimalTrial", {})
                    param_assignments = optimal_trial.get("parameterAssignments", [])

                    best_params: dict[str, Any] = {}
                    for assignment in param_assignments:
                        name = assignment["name"]
                        value = assignment["value"]
                        if name == "beta_max":
                            best_params[name] = float(value)
                        elif name == "latent_dim":
                            best_params[name] = int(value)
                        else:
                            best_params[name] = value

                    best_value = 0.0
                    for metric in optimal_trial.get("observation", {}).get("metrics", []):
                        if metric.get("name") == "val_fitness":
                            best_value = float(metric.get("latest", 0.0))

                    n_trials = status.get("trials", 0)

                    print(f"Experiment succeeded: {n_trials} trials completed")
                    print(f"  Best params: {best_params}")
                    print(f"  Best fitness: {best_value:.4f}")

                    return TuneResult(
                        best_params=best_params,
                        best_value=best_value,
                        n_trials=n_trials,
                        best_run_id="",  # Trial pods log to MLflow directly
                    )

                elif condition.get("type") == "Failed" and condition.get("status") == "True":
                    reason = condition.get("reason", "Unknown")
                    raise RuntimeError(f"Katib experiment failed: {reason}")

            time.sleep(_POLL_INTERVAL_SECONDS)
