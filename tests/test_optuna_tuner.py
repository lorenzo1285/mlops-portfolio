"""Tests for OptunaTuner — Optuna-based VAE hyperparameter optimization (T128a).

Validates that OptunaTuner:
1. Tunes 5 VAE hyperparameters (beta_max, latent_dim, warmup_epochs, lr, dropout_p)
2. Returns TuneResult with all 5 param keys
3. latent_dim is from valid choices [8, 16, 32, 64]
4. n_trials matches config
5. TrialPruned exceptions are handled as pruned trials (not crashes)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest

from src.config import (
    MLflowConfig,
    OptunaConfig,
    OptunaPrunerConfig,
    OptunaSearchSpace,
    TuneConfig,
    VAEConfig,
)
from src.encode.encoder import EncodeResult
from src.train_ml.trainer import MLTrainResult
from src.train_vae.vae_trainer import VAETrainResult
from src.tune.tuner import TuneResult


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
def optuna_cfg():
    """Minimal Optuna configuration for fast testing."""
    return OptunaConfig(
        n_trials=3,
        study_name="test-vae-optuna",
        direction="maximize",
        pruner=OptunaPrunerConfig(
            n_startup_trials=1,
            n_warmup_steps=2,
        ),
        search_space=OptunaSearchSpace(
            beta_max_low=0.01,
            beta_max_high=1.0,
            latent_dim_choices=[8, 16, 32, 64],
            warmup_epochs_low=5,
            warmup_epochs_high=30,
            lr_low=0.0001,
            lr_high=0.001,
            dropout_p_low=0.05,
            dropout_p_high=0.30,
        ),
    )


@pytest.fixture
def tune_cfg(optuna_cfg):
    """TuneConfig with Optuna settings."""
    return TuneConfig(
        experiment_name="test-optuna-tune",
        max_trials=15,  # Not used by Optuna (uses optuna.n_trials)
        namespace="default",
        max_dl_trial_epochs=10,
        optuna=optuna_cfg,
    )


@pytest.fixture
def vae_cfg():
    """Minimal VAE config for testing."""
    return VAEConfig(
        encoder_dims=[16, 8],
        latent_dim=8,
        beta_start=0.0,
        beta_max=0.5,
        warmup_epochs=5,
        dropout_p=0.1,
        epochs=5,
        patience=3,
        batch_size=32,
        lr=0.001,
        experiment_name="test-vae",
    )


@pytest.fixture
def sample_data(tmp_path):
    """Load small real data slices from data/processed/ and save to tmp_path."""
    real_dir = Path("data/processed")
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    n = 50  # small slice — enough to exercise np.vstack/hstack

    for fname, arr_fname in [
        ("X_train.npy", "X_train.npy"),
        ("X_train_augmented.npy", "X_train_augmented.npy"),
        ("y_train_augmented.npy", "y_train_augmented.npy"),
        ("X_val.npy", "X_val.npy"),
        ("y_val.npy", "y_val.npy"),
        ("X_test.npy", "X_test.npy"),
        ("y_test.npy", "y_test.npy"),
        ("y_train.npy", "y_train.npy"),
    ]:
        arr = np.load(real_dir / fname)[:n]
        np.save(data_dir / arr_fname, arr)

    return {"data_dir": data_dir}


@pytest.fixture
def mock_vae_result(tmp_path):
    """Mock VAETrainResult."""
    encoder_path = tmp_path / "encoder.pth"
    decoder_path = tmp_path / "decoder.pth"
    encoder_path.touch()
    decoder_path.touch()
    
    return VAETrainResult(
        best_epoch=10,
        final_elbo=-250.5,
        encoder_path=str(encoder_path),
        decoder_path=str(decoder_path),
        run_id="vae-run-123",
    )


@pytest.fixture
def mock_encode_result():
    """Mock EncodeResult using real Z-space slices."""
    real_dir = Path("data/processed")
    return EncodeResult(
        Z_train_augmented=np.load(real_dir / "Z_train_augmented.npy")[:50],
        Z_val=np.load(real_dir / "Z_val.npy")[:50],
        Z_test=np.load(real_dir / "Z_test.npy")[:50],
    )


@pytest.fixture
def mock_ml_result():
    """Mock MLTrainResult with good fitness — fatal recall above gate."""
    return MLTrainResult(
        best_run_id="ml-run-456",
        model_path="models/xgb_seed0.pkl",
        best_seed=0,
        eout_macro_f1=0.68,
        eval_macro_f1=0.70,
        eval_fatal_recall=0.60,  # above 0.50 gate → no penalty
    )


# ── Test Class ───────────────────────────────────────────────────────────────


class TestOptunaTuner:
    """TDD tests for OptunaTuner class."""

    def test_optuna_tuner_tune_returns_tune_result(
        self,
        tune_cfg,
        vae_cfg,
        mlflow_cfg,
        sample_data,
        mock_vae_result,
        mock_encode_result,
        mock_ml_result,
    ):
        """OptunaTuner.tune() returns TuneResult with best_params, best_value, n_trials, best_run_id."""
        # Arrange: Import OptunaTuner (will fail - RED phase)
        from src.tune.optuna_tuner import OptunaTuner
        
        # Mock the training components to avoid real training
        with patch("src.tune.optuna_tuner.DVAETrainer") as MockVAETrainer, \
             patch("src.tune.optuna_tuner.LatentEncoder") as MockEncoder, \
             patch("src.tune.optuna_tuner.MLTrainer") as MockMLTrainer:
            
            # Configure mocks
            mock_vae_trainer = MagicMock()
            mock_vae_trainer.train.return_value = mock_vae_result
            MockVAETrainer.return_value = mock_vae_trainer
            
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = mock_encode_result
            MockEncoder.return_value = mock_encoder
            
            mock_ml_trainer = MagicMock()
            mock_ml_trainer.train.return_value = mock_ml_result
            MockMLTrainer.return_value = mock_ml_trainer
            
            # Create tuner
            tuner = OptunaTuner(
                tune_config=tune_cfg,
                vae_config=vae_cfg,
                mlflow_config=mlflow_cfg,
                data_dir=sample_data["data_dir"].parent,
            )
            
            # Act: Run tuning
            result = tuner.tune()
            
            # Assert: Returns TuneResult
            assert isinstance(result, TuneResult), \
                f"Expected TuneResult, got {type(result)}"
            assert hasattr(result, "best_params"), \
                "TuneResult should have best_params attribute"
            assert hasattr(result, "best_value"), \
                "TuneResult should have best_value attribute"
            assert hasattr(result, "n_trials"), \
                "TuneResult should have n_trials attribute"
            assert hasattr(result, "best_run_id"), \
                "TuneResult should have best_run_id attribute"

    def test_optuna_tuner_best_params_has_all_five_keys(
        self,
        tune_cfg,
        vae_cfg,
        mlflow_cfg,
        sample_data,
        mock_vae_result,
        mock_encode_result,
        mock_ml_result,
    ):
        """best_params dict contains all 5 hyperparameters: beta_max, latent_dim, warmup_epochs, lr, dropout_p."""
        # Arrange
        from src.tune.optuna_tuner import OptunaTuner
        
        with patch("src.tune.optuna_tuner.DVAETrainer") as MockVAETrainer, \
             patch("src.tune.optuna_tuner.LatentEncoder") as MockEncoder, \
             patch("src.tune.optuna_tuner.MLTrainer") as MockMLTrainer:
            
            mock_vae_trainer = MagicMock()
            mock_vae_trainer.train.return_value = mock_vae_result
            MockVAETrainer.return_value = mock_vae_trainer
            
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = mock_encode_result
            MockEncoder.return_value = mock_encoder
            
            mock_ml_trainer = MagicMock()
            mock_ml_trainer.train.return_value = mock_ml_result
            MockMLTrainer.return_value = mock_ml_trainer
            
            tuner = OptunaTuner(
                tune_config=tune_cfg,
                vae_config=vae_cfg,
                mlflow_config=mlflow_cfg,
                data_dir=sample_data["data_dir"].parent,
            )
            
            # Act
            result = tuner.tune()
            
            # Assert: All 5 param keys present
            expected_keys = {"beta_max", "latent_dim", "warmup_epochs", "lr", "dropout_p"}
            actual_keys = set(result.best_params.keys())
            assert actual_keys == expected_keys, \
                f"Expected param keys {expected_keys}, got {actual_keys}"

    def test_optuna_tuner_latent_dim_in_valid_choices(
        self,
        tune_cfg,
        vae_cfg,
        mlflow_cfg,
        sample_data,
        mock_vae_result,
        mock_encode_result,
        mock_ml_result,
    ):
        """best_params['latent_dim'] is one of [8, 16, 32, 64]."""
        # Arrange
        from src.tune.optuna_tuner import OptunaTuner
        
        with patch("src.tune.optuna_tuner.DVAETrainer") as MockVAETrainer, \
             patch("src.tune.optuna_tuner.LatentEncoder") as MockEncoder, \
             patch("src.tune.optuna_tuner.MLTrainer") as MockMLTrainer:
            
            mock_vae_trainer = MagicMock()
            mock_vae_trainer.train.return_value = mock_vae_result
            MockVAETrainer.return_value = mock_vae_trainer
            
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = mock_encode_result
            MockEncoder.return_value = mock_encoder
            
            mock_ml_trainer = MagicMock()
            mock_ml_trainer.train.return_value = mock_ml_result
            MockMLTrainer.return_value = mock_ml_trainer
            
            tuner = OptunaTuner(
                tune_config=tune_cfg,
                vae_config=vae_cfg,
                mlflow_config=mlflow_cfg,
                data_dir=sample_data["data_dir"].parent,
            )
            
            # Act
            result = tuner.tune()
            
            # Assert: latent_dim in valid choices
            valid_choices = tune_cfg.optuna.search_space.latent_dim_choices
            assert result.best_params["latent_dim"] in valid_choices, \
                f"latent_dim={result.best_params['latent_dim']} not in {valid_choices}"

    def test_optuna_tuner_n_trials_matches_config(
        self,
        tune_cfg,
        vae_cfg,
        mlflow_cfg,
        sample_data,
        mock_vae_result,
        mock_encode_result,
        mock_ml_result,
    ):
        """TuneResult.n_trials matches optuna.n_trials from config."""
        # Arrange
        from src.tune.optuna_tuner import OptunaTuner
        
        with patch("src.tune.optuna_tuner.DVAETrainer") as MockVAETrainer, \
             patch("src.tune.optuna_tuner.LatentEncoder") as MockEncoder, \
             patch("src.tune.optuna_tuner.MLTrainer") as MockMLTrainer:
            
            mock_vae_trainer = MagicMock()
            mock_vae_trainer.train.return_value = mock_vae_result
            MockVAETrainer.return_value = mock_vae_trainer
            
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = mock_encode_result
            MockEncoder.return_value = mock_encoder
            
            mock_ml_trainer = MagicMock()
            mock_ml_trainer.train.return_value = mock_ml_result
            MockMLTrainer.return_value = mock_ml_trainer
            
            tuner = OptunaTuner(
                tune_config=tune_cfg,
                vae_config=vae_cfg,
                mlflow_config=mlflow_cfg,
                data_dir=sample_data["data_dir"].parent,
            )
            
            # Act
            result = tuner.tune()
            
            # Assert: n_trials matches config
            assert result.n_trials == tune_cfg.optuna.n_trials, \
                f"Expected {tune_cfg.optuna.n_trials} trials, got {result.n_trials}"

    def test_optuna_tuner_fitness_penalised_when_fatal_recall_below_gate(
        self,
        tune_cfg,
        vae_cfg,
        mlflow_cfg,
        sample_data,
        mock_vae_result,
        mock_encode_result,
    ):
        """When eval_fatal_recall < 0.50, best_value is 0.5 × eval_macro_f1 (not full F1)."""
        from src.tune.optuna_tuner import OptunaTuner

        low_recall_result = MLTrainResult(
            best_run_id="ml-run-low",
            model_path="models/xgb_seed0.pkl",
            best_seed=0,
            eout_macro_f1=0.60,
            eval_macro_f1=0.70,
            eval_fatal_recall=0.30,  # below 0.50 gate → 50% penalty
        )

        with patch("src.tune.optuna_tuner.DVAETrainer") as MockVAETrainer, \
             patch("src.tune.optuna_tuner.LatentEncoder") as MockEncoder, \
             patch("src.tune.optuna_tuner.MLTrainer") as MockMLTrainer:

            mock_vae_trainer = MagicMock()
            mock_vae_trainer.train.return_value = mock_vae_result
            MockVAETrainer.return_value = mock_vae_trainer

            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = mock_encode_result
            MockEncoder.return_value = mock_encoder

            mock_ml_trainer = MagicMock()
            mock_ml_trainer.train.return_value = low_recall_result
            MockMLTrainer.return_value = mock_ml_trainer

            tuner = OptunaTuner(
                tune_config=tune_cfg,
                vae_config=vae_cfg,
                mlflow_config=mlflow_cfg,
                data_dir=sample_data["data_dir"].parent,
            )

            result = tuner.tune()

            expected_fitness = 0.70 * 0.5
            assert abs(result.best_value - expected_fitness) < 1e-6, (
                f"Expected penalised fitness={expected_fitness:.4f} "
                f"(eval_macro_f1=0.70 × 0.5), got {result.best_value:.4f}"
            )

    def test_optuna_tuner_handles_trial_pruned_gracefully(
        self,
        tune_cfg,
        vae_cfg,
        mlflow_cfg,
        sample_data,
        mock_vae_result,
        mock_encode_result,
        mock_ml_result,
    ):
        """When DVAETrainer raises TrialPruned, OptunaTuner handles it as a pruned trial (not crash)."""
        # Arrange
        from src.tune.optuna_tuner import OptunaTuner
        
        with patch("src.tune.optuna_tuner.DVAETrainer") as MockVAETrainer, \
             patch("src.tune.optuna_tuner.LatentEncoder") as MockEncoder, \
             patch("src.tune.optuna_tuner.MLTrainer") as MockMLTrainer:
            
            # First trial: VAE raises TrialPruned
            # Subsequent trials: return normal results
            call_count = [0]
            def train_side_effect(*args, **kwargs):
                if call_count[0] == 0:
                    call_count[0] += 1
                    raise optuna.TrialPruned("Pruned by median pruner")
                call_count[0] += 1
                return mock_vae_result
            
            mock_vae_trainer = MagicMock()
            mock_vae_trainer.train.side_effect = train_side_effect
            MockVAETrainer.return_value = mock_vae_trainer
            
            mock_encoder = MagicMock()
            mock_encoder.encode.return_value = mock_encode_result
            MockEncoder.return_value = mock_encoder
            
            mock_ml_trainer = MagicMock()
            mock_ml_trainer.train.return_value = mock_ml_result
            MockMLTrainer.return_value = mock_ml_trainer
            
            tuner = OptunaTuner(
                tune_config=tune_cfg,
                vae_config=vae_cfg,
                mlflow_config=mlflow_cfg,
                data_dir=sample_data["data_dir"].parent,
            )
            
            # Act: Should not crash even though first trial is pruned
            result = tuner.tune()
            
            # Assert: Study completed successfully
            assert isinstance(result, TuneResult), \
                "OptunaTuner should return TuneResult even with pruned trials"
            # At least one trial should have completed (n_trials=3, first pruned, 2 completed)
            # Optuna still counts pruned trials in n_trials
            assert result.n_trials == tune_cfg.optuna.n_trials, \
                f"Expected {tune_cfg.optuna.n_trials} trials total (including pruned)"
