"""Tests for tune stage — T057/T058.

Validates that the tune stage:
1. Submits Katib Experiment CRD to Kubernetes with {{winner}} rendered
2. Polls until experiment completes (Succeeded status)
3. Reads optimal trial parameterAssignments (beta_max, latent_dim)
4. Returns TuneResult with correct typed best_params and best_value
   (params.yaml is updated by run.py, not by HyperparamTuner)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import yaml

from src.config import MLflowConfig, TuneConfig
from src.tune.tuner import HyperparamTuner, TuneResult


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mlflow_cfg():
    return MLflowConfig(
        tracking_uri="mlruns/",
        experiment_name_ml="crash-severity-ml",
        experiment_name_dl="crash-severity-dl",
        experiment_name_vae="crash-severity-vae",
        experiment_name_tune="crash-severity-tune",
        model_name="crash-severity",
    )


@pytest.fixture
def tune_cfg():
    return TuneConfig(
        experiment_name="vae-hyperparameter-tuning",
        max_trials=15,
        namespace="default",
    )


@pytest.fixture
def sample_data():
    """Generate small sample arrays for testing."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    X_val = np.random.randn(20, 10)
    y_val = np.random.randint(0, 3, 20)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def katib_experiment_yaml(tmp_path):
    """Create a mock Katib experiment YAML template."""
    yaml_content = {
        "apiVersion": "kubeflow.org/v1beta1",
        "kind": "Experiment",
        "metadata": {
            "name": "vae-hyperparameter-tuning",
            "namespace": "default",
        },
        "spec": {
            "objective": {
                "type": "maximize",
                "objectiveMetricName": "val_fitness",
            },
            "algorithm": {
                "algorithmName": "bayesianoptimization",
            },
            "parallelTrialCount": 1,
            "maxTrialCount": 15,
            "parameters": [
                {
                    "name": "beta_max",
                    "parameterType": "categorical",
                    "feasibleSpace": {
                        "list": ["0.05", "0.1", "0.2", "0.5", "1.0"],
                    },
                },
                {
                    "name": "latent_dim",
                    "parameterType": "categorical",
                    "feasibleSpace": {
                        "list": ["8", "16", "32"],
                    },
                },
            ],
            "trialTemplate": {
                "primaryContainerName": "trial",
                "trialParameters": [
                    {"name": "beta_max", "reference": "beta_max"},
                    {"name": "latent_dim", "reference": "latent_dim"},
                ],
                "trialSpec": {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "spec": {
                        "template": {
                            "spec": {
                                "containers": [
                                    {
                                        "name": "trial",
                                        "image": "mlops-portfolio:latest",
                                        "command": [
                                            "python",
                                            "-m",
                                            "src.tune.trial",
                                            "--beta_max=${trialParameters.beta_max}",
                                            "--latent_dim=${trialParameters.latent_dim}",
                                            "--winner={{winner}}",
                                        ],
                                    }
                                ],
                                "restartPolicy": "Never",
                            }
                        }
                    },
                },
            },
        },
    }
    yaml_path = tmp_path / "vae_experiment.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    return yaml_path


# ── Tests ────────────────────────────────────────────────────────────────────


def test_tune_submits_katib_experiment(
    mlflow_cfg, tune_cfg, sample_data, katib_experiment_yaml
):
    """T057: HyperparamTuner submits Katib Experiment CRD with winner rendered."""
    X_train, y_train, X_val, y_val = sample_data

    mock_k8s_client = MagicMock()
    mock_custom_api = MagicMock()
    mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api

    mock_experiment_status = {
        "status": {
            "conditions": [
                {
                    "type": "Succeeded",
                    "status": "True",
                }
            ],
            "currentOptimalTrial": {
                "parameterAssignments": [
                    {"name": "beta_max", "value": "0.2"},
                    {"name": "latent_dim", "value": "16"},
                ],
                "observation": {
                    "metrics": [
                        {"name": "val_fitness", "latest": "0.42"},
                    ],
                },
            },
            "trials": 15,
        }
    }

    mock_custom_api.get_namespaced_custom_object.return_value = mock_experiment_status

    with (
        patch("src.tune.tuner.client", mock_k8s_client),
        patch("src.tune.tuner.config"),
        patch("src.tune.tuner.time.sleep"),
    ):
        tuner = HyperparamTuner(
            mlflow_config=mlflow_cfg,
            tune_config=tune_cfg,
            winner="ml",
            yaml_path=katib_experiment_yaml,
        )

        result = tuner.tune(X_train, y_train, X_val, y_val)

        # Assert Experiment was submitted
        mock_custom_api.create_namespaced_custom_object.assert_called_once()
        call_args = mock_custom_api.create_namespaced_custom_object.call_args

        # Verify API call parameters
        assert call_args[1]["group"] == "kubeflow.org"
        assert call_args[1]["version"] == "v1beta1"
        assert call_args[1]["namespace"] == "default"
        assert call_args[1]["plural"] == "experiments"

        # Verify {{winner}} was rendered before submission
        body = call_args[1]["body"]
        submitted_cmd = (
            body["spec"]["trialTemplate"]["trialSpec"]
            ["spec"]["template"]["spec"]["containers"][0]["command"]
        )
        assert "--winner=ml" in submitted_cmd
        assert "{{winner}}" not in " ".join(submitted_cmd)

        # Verify result contains best params
        assert result.best_params["beta_max"] == 0.2
        assert result.best_params["latent_dim"] == 16
        assert result.best_value == 0.42
        assert result.n_trials == 15


def test_tune_polls_until_succeeded(
    mlflow_cfg, tune_cfg, sample_data, katib_experiment_yaml
):
    """T057: HyperparamTuner polls experiment until Succeeded status."""
    X_train, y_train, X_val, y_val = sample_data

    mock_k8s_client = MagicMock()
    mock_custom_api = MagicMock()
    mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api

    # Simulate multiple polls: Running → Running → Succeeded
    mock_experiment_statuses = [
        {
            "status": {
                "conditions": [{"type": "Running", "status": "True"}],
            }
        },
        {
            "status": {
                "conditions": [{"type": "Running", "status": "True"}],
            }
        },
        {
            "status": {
                "conditions": [{"type": "Succeeded", "status": "True"}],
                "currentOptimalTrial": {
                    "parameterAssignments": [
                        {"name": "beta_max", "value": "0.5"},
                        {"name": "latent_dim", "value": "32"},
                    ],
                    "observation": {
                        "metrics": [{"name": "val_fitness", "latest": "0.55"}],
                    },
                },
                "trials": 15,
            }
        },
    ]

    mock_custom_api.get_namespaced_custom_object.side_effect = mock_experiment_statuses

    with (
        patch("src.tune.tuner.client", mock_k8s_client),
        patch("src.tune.tuner.config"),
        patch("src.tune.tuner.time.sleep"),
    ):
        tuner = HyperparamTuner(
            mlflow_config=mlflow_cfg,
            tune_config=tune_cfg,
            winner="dl",
            yaml_path=katib_experiment_yaml,
        )

        result = tuner.tune(X_train, y_train, X_val, y_val)

        # Assert it polled 3 times
        assert mock_custom_api.get_namespaced_custom_object.call_count == 3

        # Assert final result
        assert result.best_params["beta_max"] == 0.5
        assert result.best_params["latent_dim"] == 32


def test_tune_returns_typed_best_params(
    mlflow_cfg, tune_cfg, sample_data, katib_experiment_yaml
):
    """T057: best_params values are typed (float for beta_max, int for latent_dim)."""
    X_train, y_train, X_val, y_val = sample_data

    mock_k8s_client = MagicMock()
    mock_custom_api = MagicMock()
    mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api

    mock_experiment_status = {
        "status": {
            "conditions": [{"type": "Succeeded", "status": "True"}],
            "currentOptimalTrial": {
                "parameterAssignments": [
                    {"name": "beta_max", "value": "0.1"},
                    {"name": "latent_dim", "value": "16"},
                ],
                "observation": {
                    "metrics": [{"name": "val_fitness", "latest": "0.48"}],
                },
            },
            "trials": 10,
        }
    }
    mock_custom_api.get_namespaced_custom_object.return_value = mock_experiment_status

    with (
        patch("src.tune.tuner.client", mock_k8s_client),
        patch("src.tune.tuner.config"),
        patch("src.tune.tuner.time.sleep"),
    ):
        tuner = HyperparamTuner(
            mlflow_config=mlflow_cfg,
            tune_config=tune_cfg,
            winner="ml",
            yaml_path=katib_experiment_yaml,
        )

        result = tuner.tune(X_train, y_train, X_val, y_val)

        assert isinstance(result.best_params["beta_max"], float)
        assert isinstance(result.best_params["latent_dim"], int)
        assert result.best_params["beta_max"] == 0.1
        assert result.best_params["latent_dim"] == 16


def test_tune_handles_failed_experiment(
    mlflow_cfg, tune_cfg, sample_data, katib_experiment_yaml
):
    """T057: HyperparamTuner raises RuntimeError when Katib experiment fails."""
    X_train, y_train, X_val, y_val = sample_data

    mock_k8s_client = MagicMock()
    mock_custom_api = MagicMock()
    mock_k8s_client.CustomObjectsApi.return_value = mock_custom_api

    mock_experiment_status = {
        "status": {
            "conditions": [{"type": "Failed", "status": "True", "reason": "TrialFailed"}],
        }
    }
    mock_custom_api.get_namespaced_custom_object.return_value = mock_experiment_status

    with (
        patch("src.tune.tuner.client", mock_k8s_client),
        patch("src.tune.tuner.config"),
        patch("src.tune.tuner.time.sleep"),
    ):
        tuner = HyperparamTuner(
            mlflow_config=mlflow_cfg,
            tune_config=tune_cfg,
            winner="ml",
            yaml_path=katib_experiment_yaml,
        )

        with pytest.raises(RuntimeError, match="Katib experiment failed"):
            tuner.tune(X_train, y_train, X_val, y_val)
