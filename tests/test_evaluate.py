"""RED tests for evaluate stage — T046.

Gate logic tests call ABEvaluator.evaluate() directly so that
mock.patch works (same process). One subprocess test uses a real
temp MLflow store — no cross-process patching required.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import ABTestConfig, MLflowConfig, ModelConfig
from src.evaluate.evaluator import ABEvaluator, EvaluationResult


# ── experiment-id constants that mirror what the mock dispatcher uses ────────

_ML_EXP_ID = "1"
_DL_EXP_ID = "2"


# ── DataFrame helpers ────────────────────────────────────────────────────────

def _ml_df(f1_values: list[float], recall_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "run_id": f"ml_{i}",
            "status": "FINISHED",
            "metrics.eout_macro_f1": f1,
            "metrics.eout_fatal_recall": recall,
            "experiment_id": _ML_EXP_ID,
        }
        for i, (f1, recall) in enumerate(zip(f1_values, recall_values))
    ])


def _dl_df(f1_values: list[float], recall_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "run_id": f"dl_{i}",
            "status": "FINISHED",
            "metrics.eout_macro_f1": f1,
            "metrics.eout_fatal_recall": recall,
            "experiment_id": _DL_EXP_ID,
        }
        for i, (f1, recall) in enumerate(zip(f1_values, recall_values))
    ])


def _search_runs_dispatch(ml_df: pd.DataFrame, dl_df: pd.DataFrame):
    """Return a mock for mlflow.search_runs that dispatches by experiment_ids."""
    def _search(experiment_ids, filter_string=None, **kwargs):
        if _ML_EXP_ID in experiment_ids:
            return ml_df
        if _DL_EXP_ID in experiment_ids:
            return dl_df
        return pd.DataFrame()
    return _search


def _mock_get_experiment_by_name(name: str):
    exp = MagicMock()
    exp.experiment_id = _ML_EXP_ID if "ml" in name else _DL_EXP_ID
    return exp


# ── shared fixtures ──────────────────────────────────────────────────────────

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
def ab_cfg():
    return ABTestConfig(seeds=[0, 1, 2], alpha=0.05, tiebreak="ml")


@pytest.fixture
def model_cfg():
    return ModelConfig(
        n_classes=3,
        n_select=3,
        macro_f1_threshold=0.35,
        fatal_recall_threshold=0.50,
    )


@pytest.fixture
def evaluator(mlflow_cfg, ab_cfg, model_cfg):
    return ABEvaluator(mlflow_cfg, ab_cfg, model_cfg)


# ── T046 RED tests — gate logic via direct class interface ───────────────────

class TestABEvaluatorGates:
    """Boundary tests for ABEvaluator.evaluate() gate enforcement.

    All patches are applied in the same process so mock.patch works.
    Tests are RED: evaluate() raises NotImplementedError until T047.
    """

    def test_evaluate_returns_evaluation_result(self, evaluator):
        """ABEvaluator.evaluate() returns EvaluationResult (not None, not dict)."""
        ml = _ml_df([0.52, 0.52, 0.52], [0.55, 0.55, 0.55])
        dl = _dl_df([0.46, 0.46, 0.46], [0.60, 0.60, 0.60])

        with patch("mlflow.search_runs", _search_runs_dispatch(ml, dl)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        assert isinstance(result, EvaluationResult)

    def test_result_has_all_required_fields(self, evaluator):
        """EvaluationResult exposes: winner, p_value, cohens_d, ml_mean_f1,
        dl_mean_f1, gates_passed."""
        ml = _ml_df([0.52, 0.52, 0.52], [0.55, 0.55, 0.55])
        dl = _dl_df([0.46, 0.46, 0.46], [0.60, 0.60, 0.60])

        with patch("mlflow.search_runs", _search_runs_dispatch(ml, dl)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        required = ("winner", "p_value", "cohens_d", "ml_mean_f1", "dl_mean_f1", "gates_passed")
        for field in required:
            assert hasattr(result, field), f"EvaluationResult missing field: '{field}'"

    def test_gates_fail_when_winner_fatal_recall_below_threshold(self, evaluator):
        """gates_passed is False when winner mean fatal_recall < 0.50.

        ML mean_f1 ≈ 0.41 (wins; > 0.35 passes F1 gate) but
        ML mean_fatal_recall ≈ 0.46 (< 0.50 fails recall gate).
        """
        ml = _ml_df([0.40, 0.41, 0.42], [0.45, 0.46, 0.47])  # mean recall ≈ 0.46
        dl = _dl_df([0.37, 0.38, 0.39], [0.55, 0.56, 0.57])

        with patch("mlflow.search_runs", _search_runs_dispatch(ml, dl)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        assert result.winner == "ml", f"Expected winner='ml', got '{result.winner}'"
        assert result.gates_passed is False, (
            f"Expected gates_passed=False (ML fatal_recall ≈ 0.46 < 0.50 threshold), "
            f"got gates_passed={result.gates_passed}"
        )

    def test_gates_fail_when_winner_f1_below_threshold(self, mlflow_cfg, ab_cfg, model_cfg):
        """gates_passed is False when winner mean macro_f1 <= 0.35.

        ML mean_f1 ≈ 0.31 (< 0.35 gate); recall passes but F1 gate fails.
        """
        ml = _ml_df([0.30, 0.31, 0.32], [0.60, 0.61, 0.62])  # mean f1 ≈ 0.31
        dl = _dl_df([0.28, 0.29, 0.30], [0.60, 0.61, 0.62])

        ev = ABEvaluator(mlflow_cfg, ab_cfg, model_cfg)

        with patch("mlflow.search_runs", _search_runs_dispatch(ml, dl)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = ev.evaluate()

        assert result.gates_passed is False, (
            f"Expected gates_passed=False (ML macro_f1 ≈ 0.31 < 0.35 threshold), "
            f"got gates_passed={result.gates_passed}"
        )

    def test_gates_pass_when_winner_clears_both_thresholds(self, evaluator):
        """gates_passed is True when winner macro_f1 > 0.35 AND fatal_recall > 0.50.

        ML mean_f1 ≈ 0.52 (> 0.35) and mean_fatal_recall ≈ 0.56 (> 0.50).
        """
        ml = _ml_df([0.52, 0.52, 0.52], [0.55, 0.56, 0.57])
        dl = _dl_df([0.46, 0.46, 0.46], [0.60, 0.61, 0.62])

        with patch("mlflow.search_runs", _search_runs_dispatch(ml, dl)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        assert result.winner == "ml"
        assert result.gates_passed is True, (
            f"Expected gates_passed=True when winner clears both thresholds, "
            f"got gates_passed={result.gates_passed}"
        )

    def test_tiebreak_defaults_to_ml_when_not_significant(self, evaluator):
        """Winner defaults to tiebreak='ml' when p_value >= alpha.

        Identical ML/DL F1 distributions → p_value = 1.0 → tiebreak applies.
        """
        ml = _ml_df([0.52, 0.52, 0.52], [0.55, 0.55, 0.55])
        dl = _dl_df([0.52, 0.52, 0.52], [0.55, 0.55, 0.55])  # identical to ML

        with patch("mlflow.search_runs", _search_runs_dispatch(ml, dl)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        assert result.winner == "ml", (
            f"Expected tiebreak winner='ml' when p >= alpha, got '{result.winner}'"
        )


# ── Subprocess boundary test — uses real temp MLflow store ───────────────────

class TestEvaluateRunScript:
    """Subprocess test: run.py writes reports and exits 1 when gates fail.

    Uses a real temp MLflow tracking directory — no cross-process patches.
    RED: will fail with ModuleNotFoundError until src/evaluate/run.py exists (T047).
    """

    def _log_runs(
        self,
        tracking_uri: str,
        exp_name: str,
        f1_values: list[float],
        recall_values: list[float],
    ) -> None:
        import mlflow as mf
        mf.set_tracking_uri(tracking_uri)
        mf.set_experiment(exp_name)
        for i, (f1, recall) in enumerate(zip(f1_values, recall_values)):
            with mf.start_run(run_name=f"seed_{i}"):
                mf.log_metrics({
                    "eout_macro_f1": f1,
                    "eout_fatal_recall": recall,
                    "ein_macro_f1": f1 + 0.05,
                    "generalisation_gap": 0.05,
                })

    def test_run_exits_1_and_writes_report_when_gates_fail(self, tmp_path):
        """run.py exits 1 and writes evaluation_report.json when gates fail.

        ML wins on F1 (0.41 > DL's 0.38) but fatal_recall 0.46 < 0.50 gate.
        """
        import subprocess
        import yaml

        mlruns = tmp_path / "mlruns"
        mlruns.mkdir()
        tracking_uri = mlruns.as_uri()  # file:///C:/... — required on Windows

        self._log_runs(tracking_uri, "crash-severity-ml",
                       [0.41, 0.41, 0.41], [0.46, 0.46, 0.46])
        self._log_runs(tracking_uri, "crash-severity-dl",
                       [0.38, 0.38, 0.38], [0.60, 0.60, 0.60])

        params = {
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
                "tracking_uri": tracking_uri,
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
        params_path = tmp_path / "params.yaml"
        with open(params_path, "w") as f:
            yaml.dump(params, f)

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        workspace_root = Path(__file__).parent.parent

        result = subprocess.run(
            [sys.executable, "-m", "src.evaluate.run"],
            cwd=str(workspace_root),
            env={
                **os.environ,
                "PARAMS_PATH": str(params_path),
                "MLFLOW_TRACKING_URI": tracking_uri,
                "REPORT_PATH": str(docs_dir / "evaluation_report.json"),
                "AB_REPORT_PATH": str(docs_dir / "ab_test_comparison.json"),
            },
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1, (
            f"Expected exit 1 when gates fail, got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        report_path = docs_dir / "evaluation_report.json"
        assert report_path.exists(), (
            f"evaluation_report.json not written.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        report = json.loads(report_path.read_text())
        for field in ("winner", "p_value", "cohens_d", "ml_mean_f1", "dl_mean_f1", "gates_passed"):
            assert field in report, f"evaluation_report.json missing field: '{field}'"

        assert report["gates_passed"] is False