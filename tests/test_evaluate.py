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
_GMM_EXP_ID = "3"


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


def _gmm_df(f1_values: list[float], recall_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "run_id": f"gmm_{i}",
            "status": "FINISHED",
            "metrics.eout_macro_f1": f1,
            "metrics.eout_fatal_recall": recall,
            "experiment_id": _GMM_EXP_ID,
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


def _search_runs_dispatch_3way(ml_df: pd.DataFrame, dl_df: pd.DataFrame, gmm_df: pd.DataFrame):
    """Return a mock for mlflow.search_runs that dispatches by experiment_ids (3-way)."""
    def _search(experiment_ids, filter_string=None, **kwargs):
        if _ML_EXP_ID in experiment_ids:
            return ml_df
        if _DL_EXP_ID in experiment_ids:
            return dl_df
        if _GMM_EXP_ID in experiment_ids:
            return gmm_df
        return pd.DataFrame()
    return _search


def _mock_get_experiment_by_name(name: str):
    exp = MagicMock()
    if "ml" in name:
        exp.experiment_id = _ML_EXP_ID
    elif "dl" in name:
        exp.experiment_id = _DL_EXP_ID
    elif "gmm" in name:
        exp.experiment_id = _GMM_EXP_ID
    else:
        exp.experiment_id = _ML_EXP_ID  # fallback
    return exp


# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mlflow_cfg():
    return MLflowConfig(
        tracking_uri="mlruns/",
        experiment_name_ml="crash-severity-ml",
        experiment_name_dl="crash-severity-dl",
        experiment_name_gmm="crash-severity-gmm",
        experiment_name_vae="crash-severity-vae",
        experiment_name_tune="crash-severity-tune",
        model_name="crash-severity",
    )


@pytest.fixture
def ab_cfg():
    return ABTestConfig(seeds=[0, 1, 2], alpha=0.05, tiebreak=["ml", "dl", "gmm"])


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
        gmm = _gmm_df([0.20, 0.20, 0.20], [0.30, 0.30, 0.30])  # neutral/weak

        with patch("mlflow.search_runs", _search_runs_dispatch_3way(ml, dl, gmm)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        assert isinstance(result, EvaluationResult)

    def test_result_has_all_required_fields(self, evaluator):
        """EvaluationResult exposes: winner, p_value, cohens_d, ml_mean_f1,
        dl_mean_f1, gates_passed."""
        ml = _ml_df([0.52, 0.52, 0.52], [0.55, 0.55, 0.55])
        dl = _dl_df([0.46, 0.46, 0.46], [0.60, 0.60, 0.60])
        gmm = _gmm_df([0.20, 0.20, 0.20], [0.30, 0.30, 0.30])  # neutral/weak

        with patch("mlflow.search_runs", _search_runs_dispatch_3way(ml, dl, gmm)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        required = ("winner", "p_value_ml_dl", "cohens_d_ml_dl", "ml_mean_f1", "dl_mean_f1", "gates_passed")
        for field in required:
            assert hasattr(result, field), f"EvaluationResult missing field: '{field}'"

    def test_gates_fail_when_winner_fatal_recall_below_threshold(self, evaluator):
        """gates_passed is False when winner mean fatal_recall < 0.50.

        ML mean_f1 ≈ 0.41 (wins; > 0.35 passes F1 gate) but
        ML mean_fatal_recall ≈ 0.46 (< 0.50 fails recall gate).
        """
        ml = _ml_df([0.40, 0.41, 0.42], [0.45, 0.46, 0.47])  # mean recall ≈ 0.46
        dl = _dl_df([0.37, 0.38, 0.39], [0.55, 0.56, 0.57])
        gmm = _gmm_df([0.20, 0.20, 0.20], [0.30, 0.30, 0.30])  # neutral/weak

        with patch("mlflow.search_runs", _search_runs_dispatch_3way(ml, dl, gmm)):
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
        gmm = _gmm_df([0.20, 0.20, 0.20], [0.30, 0.30, 0.30])  # neutral/weak

        ev = ABEvaluator(mlflow_cfg, ab_cfg, model_cfg)

        with patch("mlflow.search_runs", _search_runs_dispatch_3way(ml, dl, gmm)):
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
        gmm = _gmm_df([0.20, 0.20, 0.20], [0.30, 0.30, 0.30])  # neutral/weak

        with patch("mlflow.search_runs", _search_runs_dispatch_3way(ml, dl, gmm)):
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
        gmm = _gmm_df([0.20, 0.20, 0.20], [0.30, 0.30, 0.30])  # neutral/weak

        with patch("mlflow.search_runs", _search_runs_dispatch_3way(ml, dl, gmm)):
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

    def test_run_exits_0_and_writes_report_when_gates_fail(self, tmp_path):
        """run.py exits 0 and writes evaluation_report.json with gates_passed=False.

        run.py always exits 0 so DVC continues to tune; tune reads gates_passed
        from evaluation_report.json to decide whether to run HPO.
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
        self._log_runs(tracking_uri, "crash-severity-gmm",
                       [0.20, 0.20, 0.20], [0.30, 0.30, 0.30])  # neutral/weak

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
                "focal_loss_enabled": False,
                "focal_loss_gamma": 1.0,
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
            "ab_test": {"seeds": [0, 1, 2], "alpha": 0.05, "tiebreak": ["ml", "dl", "gmm"]},
            "mlflow": {
                "tracking_uri": tracking_uri,
                "experiment_name_ml": "crash-severity-ml",
                "experiment_name_dl": "crash-severity-dl",
                "experiment_name_gmm": "crash-severity-gmm",
                "experiment_name_vae": "crash-severity-vae",
                "experiment_name_tune": "crash-severity-tune",
                "model_name": "crash-severity",
            },
            "great_expectations": {
                "suite_name": "crash_data_suite",
                "datasource_name": "crash_data",
            },
            "gmm": {
                "n_components": {0: 1, 1: 1, 2: 2},
                "covariance_type": "full",
                "reg_covar": 1.0e-6,
                "max_iter": 100,
                "n_init": 5,
                "fatal_prior_boost": 1.0,
                "experiment_name": "crash-severity-gmm",
            },
            "tune": {
                "experiment_name": "vae-hyperparameter-tuning",
                "max_trials": 15,
                "namespace": "default",
                "max_dl_trial_epochs": 50,
                "optuna": {
                    "n_trials": 30,
                    "study_name": "vae-optuna-hpo",
                    "direction": "maximize",
                    "pruner": {"n_startup_trials": 5, "n_warmup_steps": 15},
                    "search_space": {
                        "beta_max_low": 0.01, "beta_max_high": 1.0,
                        "latent_dim_choices": [8, 16, 32, 64],
                        "warmup_epochs_low": 5, "warmup_epochs_high": 30,
                        "lr_low": 0.0001, "lr_high": 0.001,
                        "dropout_p_low": 0.05, "dropout_p_high": 0.3,
                        "target_fatal_ratio_choices": [0.05, 0.1, 0.15, 0.2],
                        "fatal_threshold_low": 0.02, "fatal_threshold_high": 0.2,
                        "focal_loss_gamma_low": 0.5, "focal_loss_gamma_high": 5.0,
                        "fatal_prior_boost_low": 1.0, "fatal_prior_boost_high": 5.0,
                    },
                },
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

        assert result.returncode == 0, (
            f"Expected exit 0 (tune stage reads gates_passed), got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        report_path = docs_dir / "evaluation_report.json"
        assert report_path.exists(), (
            f"evaluation_report.json not written.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        report = json.loads(report_path.read_text())
        for field in ("winner", "p_value_ml_dl", "cohens_d_ml_dl", "ml_mean_f1", "dl_mean_f1", "gmm_mean_f1", "gates_passed"):
            assert field in report, f"evaluation_report.json missing field: '{field}'"

        assert report["gates_passed"] is False

        ab_report_path = docs_dir / "ab_test_comparison.json"
        assert ab_report_path.exists(), (
            f"ab_test_comparison.json not written.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        ab_report = json.loads(ab_report_path.read_text())
        for field in ("winner", "p_value_ml_dl", "cohens_d_ml_dl", "ml_mean_f1", "dl_mean_f1", "gmm_mean_f1", "significant_ml_dl"):
            assert field in ab_report, f"ab_test_comparison.json missing field: '{field}'"
        assert "significant" not in ab_report, (
            "ab_test_comparison.json has old 'significant' key — use 'significant_ml_dl'"
        )


# ── T014 RED test — 3-way A/B/C evaluation ───────────────────────────────────

class TestThreeWayEvaluation:
    """Boundary tests for 3-way A/B/C evaluation (ml vs dl vs gmm).

    Tests verify EvaluationResult has fields for all three classifiers and
    pairwise comparisons. Tests MUST fail until T015 extends EvaluationResult.
    """

    def test_result_has_gmm_fields(self, evaluator):
        """EvaluationResult includes gmm_mean_f1, p_value_ml_gmm, p_value_dl_gmm, cohens_d_ml_gmm.

        This test MUST fail because these fields do not exist yet.
        """
        ml = _ml_df([0.52, 0.52, 0.52], [0.55, 0.55, 0.55])
        dl = _dl_df([0.46, 0.46, 0.46], [0.60, 0.60, 0.60])
        gmm = _gmm_df([0.58, 0.59, 0.60], [0.62, 0.63, 0.64])  # GMM wins on F1

        with patch("mlflow.search_runs", _search_runs_dispatch_3way(ml, dl, gmm)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        # Assert new GMM fields exist
        assert hasattr(result, "gmm_mean_f1"), "EvaluationResult missing field: 'gmm_mean_f1'"
        assert hasattr(result, "gmm_ci_low"), "EvaluationResult missing field: 'gmm_ci_low'"
        assert hasattr(result, "gmm_ci_high"), "EvaluationResult missing field: 'gmm_ci_high'"
        assert hasattr(result, "gmm_mean_fatal_recall"), "EvaluationResult missing field: 'gmm_mean_fatal_recall'"

        # Assert pairwise p-values exist
        assert hasattr(result, "p_value_ml_dl"), "EvaluationResult missing field: 'p_value_ml_dl' (renamed from p_value)"
        assert hasattr(result, "p_value_ml_gmm"), "EvaluationResult missing field: 'p_value_ml_gmm'"
        assert hasattr(result, "p_value_dl_gmm"), "EvaluationResult missing field: 'p_value_dl_gmm'"

        # Assert pairwise Cohen's d exist
        assert hasattr(result, "cohens_d_ml_dl"), "EvaluationResult missing field: 'cohens_d_ml_dl' (renamed from cohens_d)"
        assert hasattr(result, "cohens_d_ml_gmm"), "EvaluationResult missing field: 'cohens_d_ml_gmm'"
        assert hasattr(result, "cohens_d_dl_gmm"), "EvaluationResult missing field: 'cohens_d_dl_gmm'"

    def test_winner_can_be_gmm(self, evaluator):
        """Winner field accepts 'gmm' when GMM has highest mean F1 and is significantly better.

        GMM mean F1 ≈ 0.59 >> ML ≈ 0.42 and DL ≈ 0.38. With 3 seeds, distributions
        are well-separated → GMM should be significantly better than both → winner='gmm'.

        This test MUST fail because evaluate() does not yet query GMM experiment or
        run 3-way comparison logic.
        """
        ml = _ml_df([0.41, 0.42, 0.43], [0.55, 0.56, 0.57])
        dl = _dl_df([0.37, 0.38, 0.39], [0.60, 0.61, 0.62])
        gmm = _gmm_df([0.58, 0.59, 0.60], [0.62, 0.63, 0.64])  # clearly best

        with patch("mlflow.search_runs", _search_runs_dispatch_3way(ml, dl, gmm)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        assert result.winner in {"ml", "dl", "gmm"}, (
            f"Winner must be one of {{'ml', 'dl', 'gmm'}}, got '{result.winner}'"
        )
        assert result.winner == "gmm", (
            f"Expected winner='gmm' when GMM F1 ≈ 0.59 >> ML ≈ 0.42 and DL ≈ 0.38, "
            f"got winner='{result.winner}'"
        )

    def test_bonferroni_correction_applied(self, evaluator):
        """3-way comparison uses Bonferroni-corrected alpha (α/3 ≈ 0.017).

        When all three classifiers have similar F1 distributions, no pairwise
        p-value should be < α/3 = 0.05/3 ≈ 0.017 → tiebreak applies.

        This test MUST fail because evaluate() does not yet implement 3-way logic.
        """
        # All three have identical F1 distributions → p-values all ≈ 1.0
        ml = _ml_df([0.52, 0.52, 0.52], [0.55, 0.55, 0.55])
        dl = _dl_df([0.52, 0.52, 0.52], [0.55, 0.55, 0.55])
        gmm = _gmm_df([0.52, 0.52, 0.52], [0.55, 0.55, 0.55])

        with patch("mlflow.search_runs", _search_runs_dispatch_3way(ml, dl, gmm)):
            with patch("mlflow.get_experiment_by_name", _mock_get_experiment_by_name):
                result = evaluator.evaluate()

        # All pairwise p-values should be ≈ 1.0 (no difference)
        alpha_bonf = 0.05 / 3.0
        assert result.p_value_ml_dl >= alpha_bonf, (
            f"Expected p_value_ml_dl >= {alpha_bonf} when distributions identical, "
            f"got {result.p_value_ml_dl}"
        )
        assert result.p_value_ml_gmm >= alpha_bonf, (
            f"Expected p_value_ml_gmm >= {alpha_bonf} when distributions identical, "
            f"got {result.p_value_ml_gmm}"
        )
        assert result.p_value_dl_gmm >= alpha_bonf, (
            f"Expected p_value_dl_gmm >= {alpha_bonf} when distributions identical, "
            f"got {result.p_value_dl_gmm}"
        )