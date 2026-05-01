from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from kubernetes import client


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

    def __init__(self, mlflow_config, tune_config, winner: str) -> None:
        self._mlflow_config = mlflow_config
        self._tune_config = tune_config
        self._winner = winner

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TuneResult:
        """Submit Katib Experiment and poll until complete; return best params."""
        # Load experiment YAML template
        experiment_yaml_path = Path("k8s/katib/vae_experiment.yaml")
        with open(experiment_yaml_path) as f:
            experiment_spec = yaml.safe_load(f)
        
        # Inject winner into trial template command
        trial_spec = experiment_spec["spec"]["trialTemplate"]["trialSpec"]
        for container in trial_spec["containers"]:
            if container["name"] == "trial":
                # Replace {{winner}} placeholder in command
                container["command"] = [
                    cmd.replace("{{winner}}", self._winner)
                    for cmd in container["command"]
                ]
        
        # Create Kubernetes API client
        custom_api = client.CustomObjectsApi()
        
        # Submit experiment
        experiment_name = experiment_spec["metadata"]["name"]
        namespace = experiment_spec["metadata"]["namespace"]
        
        custom_api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1beta1",
            namespace=namespace,
            plural="experiments",
            body=experiment_spec,
        )
        
        print(f"Submitted Katib experiment: {experiment_name}")
        
        # Poll until experiment completes
        while True:
            experiment = custom_api.get_namespaced_custom_object(
                group="kubeflow.org",
                version="v1beta1",
                namespace=namespace,
                plural="experiments",
                name=experiment_name,
            )
            
            status = experiment.get("status", {})
            conditions = status.get("conditions", [])
            
            # Check for completion
            for condition in conditions:
                if condition.get("type") == "Succeeded" and condition.get("status") == "True":
                    # Extract optimal trial
                    optimal_trial = status.get("currentOptimalTrial", {})
                    param_assignments = optimal_trial.get("parameterAssignments", [])
                    
                    # Parse best params
                    best_params = {}
                    for assignment in param_assignments:
                        name = assignment["name"]
                        value = assignment["value"]
                        
                        # Convert to appropriate type
                        if name == "beta_max":
                            best_params[name] = float(value)
                        elif name == "latent_dim":
                            best_params[name] = int(value)
                        else:
                            best_params[name] = value
                    
                    # Extract best fitness value
                    observation = optimal_trial.get("observation", {})
                    metrics = observation.get("metrics", [])
                    best_value = 0.0
                    for metric in metrics:
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
            
            # Wait before next poll
            time.sleep(10)

