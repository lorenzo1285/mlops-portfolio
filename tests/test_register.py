"""Tests for register stage — T049/T050.

Validates that the register stage:
1. Refuses to register when gates_passed=false (exit 1, no registry mutation)
2. Registers model + sets @champion alias when gates_passed=true (exit 0)
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import numpy as np
import pytest
import torch
import yaml

from src.config import MLflowConfig
from src.register.registrar import ModelRegistrar, RegistryReceipt


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
def evaluation_report_pass(tmp_path) -> Path:
    report_path = tmp_path / "evaluation_report_pass.json"
    report_data = {
        "winner": "ml",
        "p_value": 0.001,
        "cohens_d": 0.8,
        "ml_mean_f1": 0.50,
        "dl_mean_f1": 0.45,
        "ml_ci_low": 0.48,
        "ml_ci_high": 0.52,
        "dl_ci_low": 0.43,
        "dl_ci_high": 0.47,
        "ml_mean_fatal_recall": 0.60,
        "dl_mean_fatal_recall": 0.55,
        "gates_passed": True,
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    return report_path


@pytest.fixture
def evaluation_report_fail(tmp_path) -> Path:
    report_path = tmp_path / "evaluation_report_fail.json"
    report_data = {
        "winner": "ml",
        "p_value": 0.5,
        "cohens_d": 0.1,
        "ml_mean_f1": 0.30,
        "dl_mean_f1": 0.28,
        "ml_ci_low": 0.28,
        "ml_ci_high": 0.32,
        "dl_ci_low": 0.26,
        "dl_ci_high": 0.30,
        "ml_mean_fatal_recall": 0.20,
        "dl_mean_fatal_recall": 0.18,
        "gates_passed": False,
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    return report_path


@pytest.fixture
def base_params() -> dict:
    """Minimal params.yaml content satisfying load_config()."""
    return {
        "features": {
            "columns": ["HOUR", "MONTH"],
            "numeric_columns": [],
            "target_column": "CRASHSEVER",
        },
        "data": {
            "raw_path": "data/raw/CGR_Crash_Data.csv",
            "processed_dir": "data/processed",
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            "random_state": 42,
            "sentinel_value": 99,
        },
        "model": {
            "n_classes": 3,
            "n_select": 3,
            "macro_f1_threshold": 0.35,
            "fatal_recall_threshold": 0.50,
        },
        "dl": {
            "input_dim": 8,
            "hidden_dim": 64,
            "dropout_p": 0.1,
            "epochs": 100,
            "patience": 10,
            "batch_size": 256,
            "lr": 0.001,
            "experiment_name": "crash-severity-dl",
        },
        "vae": {
            "encoder_dims": [256, 128, 64],
            "latent_dim": 8,
            "beta_start": 0.0,
            "beta_max": 0.5,
            "warmup_epochs": 15,
            "dropout_p": 0.15,
            "epochs": 200,
            "patience": 20,
            "batch_size": 512,
            "lr": 0.0005,
            "experiment_name": "crash-severity-vae",
        },
        "augment": {
            "tvae_epochs": 500,
            "target_fatal_ratio": 0.05,
            "random_state": 42,
        },
        "ab_test": {"seeds": [0, 1, 2], "alpha": 0.05, "tiebreak": "ml"},
        "mlflow": {
            "tracking_uri": "mlruns/",
            "experiment_name_ml": "crash-severity-ml",
            "experiment_name_dl": "crash-severity-dl",
            "experiment_name_vae": "crash-severity-vae",
            "experiment_name_tune": "crash-severity-tune",
            "model_name": "crash-severity",
        },
        "great_expectations": {
            "suite_name": "crash_data_suite",
            "datasource_name": "crash_data",
        },
    }


@pytest.fixture
def model_metadata() -> dict:
    return {
        "winner": "ml",
        "latent_dim": 8,
        "input_dim": 8,
        "hidden_dim": 64,
        "n_classes": 3,
        "dropout_p": 0.1,
    }


# ── Unit tests (ModelRegistrar class) ────────────────────────────────────────


class TestModelRegistrar:

    def test_register_refuses_when_gates_failed(
        self, mlflow_cfg, evaluation_report_fail, tmp_path, model_metadata
    ):
        """Given gates_passed=false, register() raises ValueError; no receipt written."""
        registrar = ModelRegistrar(mlflow_cfg)
        receipt_path = tmp_path / "registry_receipt.json"

        with pytest.raises(ValueError, match="Constitutional gates FAILED"):
            registrar.register(
                winner="ml",
                run_id="test_run_id",
                report_path=str(evaluation_report_fail),
                receipt_path=str(receipt_path),
                encoder_path="models/vae_encoder.pth",
                classifier_path="models/best_ml_model.pkl",
                model_metadata=model_metadata,
            )

        assert not receipt_path.exists()

    @patch("mlflow.MlflowClient")
    @patch("mlflow.register_model")
    @patch("mlflow.pyfunc.log_model")
    @patch("mlflow.log_params")
    @patch("mlflow.start_run")
    @patch("mlflow.get_experiment_by_name")
    def test_register_succeeds_when_gates_passed(
        self,
        mock_get_exp,
        mock_start_run,
        mock_log_params,
        mock_log_model,
        mock_register_model,
        mock_client_cls,
        mlflow_cfg,
        evaluation_report_pass,
        tmp_path,
        model_metadata,
    ):
        """Given gates_passed=true, register() logs pyfunc bundle, registers model,
        sets @champion alias, and writes receipt pointing to the champion training run."""
        # Arrange
        mock_get_exp.return_value = MagicMock(experiment_id="test_exp_id")

        pyfunc_run = MagicMock()
        pyfunc_run.info.run_id = "pyfunc_run_id_123"
        mock_start_run.return_value.__enter__.return_value = pyfunc_run

        mock_model_version = MagicMock()
        mock_model_version.version = "1"
        mock_register_model.return_value = mock_model_version

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        registrar = ModelRegistrar(mlflow_cfg)
        receipt_path = tmp_path / "registry_receipt.json"
        champion_run_id = "champion_training_run_123"

        # Act
        result = registrar.register(
            winner="ml",
            run_id=champion_run_id,
            report_path=str(evaluation_report_pass),
            receipt_path=str(receipt_path),
            encoder_path="models/vae_encoder.pth",
            classifier_path="models/best_ml_model.pkl",
            model_metadata=model_metadata,
        )

        # Assert: pyfunc bundle logged with correct artifacts
        mock_log_model.assert_called_once()
        log_model_kwargs = mock_log_model.call_args.kwargs
        assert log_model_kwargs["artifact_path"] == "crash_severity_model"
        assert log_model_kwargs["artifacts"]["encoder"] == "models/vae_encoder.pth"
        assert log_model_kwargs["artifacts"]["classifier"] == "models/best_ml_model.pkl"

        # Assert: model registered from pyfunc run (not the champion training run)
        mock_register_model.assert_called_once()
        reg_kwargs = mock_register_model.call_args.kwargs
        assert "pyfunc_run_id_123" in reg_kwargs["model_uri"]
        assert "crash_severity_model" in reg_kwargs["model_uri"]
        assert reg_kwargs["name"] == mlflow_cfg.model_name

        # Assert: @champion alias set
        mock_client.set_registered_model_alias.assert_called_once_with(
            name=mlflow_cfg.model_name,
            alias="champion",
            version="1",
        )

        # Assert: receipt written; run_id points back to champion training run
        assert receipt_path.exists()
        with open(receipt_path) as f:
            receipt_data = json.load(f)
        assert receipt_data["run_id"] == champion_run_id
        assert receipt_data["model_name"] == mlflow_cfg.model_name
        assert receipt_data["version"] == "1"
        assert receipt_data["alias"] == "champion"
        assert receipt_data["winner"] == "ml"

        # Assert: RegistryReceipt returned
        assert isinstance(result, RegistryReceipt)
        assert result.run_id == champion_run_id


# ── Integration tests (run.py entry point) ───────────────────────────────────


class TestRegisterStage:

    def _write_params(self, path: Path, params: dict) -> None:
        with open(path, "w") as f:
            yaml.dump(params, f)

    def test_register_exits_1_when_gates_failed(
        self, evaluation_report_fail, tmp_path, base_params
    ):
        """Gates failed → exit 1 before MLflow query; no receipt written."""
        receipt_path = tmp_path / "registry_receipt.json"
        params_path = tmp_path / "params.yaml"
        self._write_params(params_path, base_params)

        env = os.environ.copy()
        env["REPORT_PATH"] = str(evaluation_report_fail)
        env["RECEIPT_PATH"] = str(receipt_path)
        env["PARAMS_PATH"] = str(params_path)

        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.register.run"],
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, (
            f"Expected exit 1 when gates fail, got {result.returncode}. "
            f"stdout: {result.stdout}, stderr: {result.stderr}"
        )
        assert not receipt_path.exists()
        assert "Constitutional gates FAILED" in result.stderr

    def test_register_exits_0_when_gates_passed_with_real_mlflow(
        self, evaluation_report_pass, tmp_path, base_params
    ):
        """Gates passed → exit 0; receipt written; @champion alias set in registry."""
        from src.train_vae.vae_trainer import Encoder
        from xgboost import XGBClassifier

        mlflow_dir = tmp_path / "mlruns"
        mlflow_dir.mkdir()
        receipt_path = tmp_path / "registry_receipt.json"
        params_path = tmp_path / "params.yaml"
        encoder_path = tmp_path / "vae_encoder.pth"
        classifier_path = tmp_path / "best_ml_model.pkl"

        mlflow_uri = f"file:///{str(mlflow_dir).replace(os.sep, '/')}"

        params = {**base_params, "mlflow": {**base_params["mlflow"], "tracking_uri": mlflow_uri}}
        self._write_params(params_path, params)

        # Minimal dummy encoder checkpoint
        enc = Encoder(input_dim=2, encoder_dims=[4], latent_dim=2)
        torch.save(
            {"input_dim": 2, "encoder_dims": [4], "latent_dim": 2, "state_dict": enc.state_dict()},
            str(encoder_path),
        )

        # Minimal dummy XGBoost classifier
        clf = XGBClassifier(n_estimators=1, max_depth=1, random_state=42)
        clf.fit(np.array([[0, 1], [1, 0], [0.5, 0.5]]), np.array([0, 1, 2]))
        with open(str(classifier_path), "wb") as f:
            pickle.dump(clf, f)

        # MLflow experiment with one FINISHED run (needed for run query in run.py)
        mlflow.set_tracking_uri(mlflow_uri)
        exp_id = mlflow.create_experiment("crash-severity-ml")
        with mlflow.start_run(experiment_id=exp_id) as run:
            mlflow.log_metric("eout_macro_f1", 0.45)
            test_run_id = run.info.run_id

        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = mlflow_uri
        env["REPORT_PATH"] = str(evaluation_report_pass)
        env["RECEIPT_PATH"] = str(receipt_path)
        env["PARAMS_PATH"] = str(params_path)
        env["ENCODER_PATH"] = str(encoder_path)
        env["CLASSIFIER_PATH"] = str(classifier_path)

        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.register.run"],
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"Expected exit 0 when gates pass, got {result.returncode}. "
            f"stdout: {result.stdout}, stderr: {result.stderr}"
        )

        assert receipt_path.exists()
        with open(receipt_path) as f:
            receipt_data = json.load(f)
        assert receipt_data["model_name"] == "crash-severity"
        assert receipt_data["alias"] == "champion"
        assert receipt_data["run_id"] == test_run_id
        assert receipt_data["winner"] == "ml"

        # Verify @champion alias set in MLflow registry
        client = mlflow.MlflowClient(tracking_uri=mlflow_uri)
        model_version = client.get_model_version_by_alias("crash-severity", "champion")
        assert model_version is not None
