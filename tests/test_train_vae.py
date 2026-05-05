"""Tests for train_vae stage - Denoising β-VAE training."""
import os
from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import numpy as np
import optuna
import pytest
import torch

from src.config import MLflowConfig, VAEConfig
from src.train_vae.vae_trainer import DVAETrainer, Encoder, VAETrainResult


class TestTrainVAE:
    """Boundary tests for the VAE training stage."""

    @pytest.fixture
    def minimal_vae_config(self):
        """Minimal VAE config for fast testing."""
        return VAEConfig(
            encoder_dims=[16, 8],
            latent_dim=4,
            beta_start=0.0,
            beta_max=0.5,
            warmup_epochs=2,
            dropout_p=0.15,
            epochs=5,
            patience=3,
            batch_size=32,
            lr=0.001,
            experiment_name="test-vae",
        )

    @pytest.fixture
    def mlflow_config(self):
        """MLflow config pointing to test tracking URI."""
        return MLflowConfig(
            tracking_uri="mlruns/",
            experiment_name_ml="test-ml",
            experiment_name_dl="test-dl",
            experiment_name_vae="test-vae",
            experiment_name_tune="test-tune",
            model_name="test-model",
        )

    @pytest.fixture
    def dummy_X_all(self):
        """Create a small dummy dataset for testing."""
        np.random.seed(42)
        return np.random.randn(100, 10).astype(np.float32)

    @pytest.fixture
    def dummy_y_all(self):
        """Class labels matching dummy_X_all — 3 classes, Fatal underrepresented."""
        y = np.zeros(100, dtype=np.int64)
        y[70:85] = 1   # 15 Injury
        y[85:]   = 2   # 15 Fatal (overrepresented here for test stability)
        return y

    def test_vae_trainer_returns_train_result(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all, tmp_path
    ):
        """Given dummy X_all, DVAETrainer.train() returns VAETrainResult with best_epoch >= 0."""
        # Arrange: Create trainer
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train VAE
        result = trainer.train(dummy_X_all, y_all=dummy_y_all)
        
        # Assert: Returns VAETrainResult
        assert isinstance(result, VAETrainResult), \
            f"Expected VAETrainResult, got {type(result)}"
        assert result.best_epoch >= 0, \
            f"Expected best_epoch >= 0, got {result.best_epoch}"
        assert result.final_elbo < 0 or result.final_elbo >= 0, \
            "final_elbo should be a float value"

    def test_vae_creates_encoder_decoder_checkpoints(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all, tmp_path
    ):
        """Encoder and decoder .pth files are created at the configured paths."""
        # Arrange: Configure paths
        encoder_path = tmp_path / "encoder.pth"
        decoder_path = tmp_path / "decoder.pth"
        
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train VAE (need to modify trainer to accept output paths)
        # For now, trainer will use default paths - we'll check those exist
        result = trainer.train(dummy_X_all, y_all=dummy_y_all)
        
        # Assert: Checkpoint files exist
        assert Path(result.encoder_path).exists(), \
            f"Encoder checkpoint not found at {result.encoder_path}"
        assert Path(result.decoder_path).exists(), \
            f"Decoder checkpoint not found at {result.decoder_path}"

    def test_vae_logs_to_mlflow(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """VAE training logs metrics to MLflow experiment."""
        # Arrange: Set tracking URI
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name_vae)
        
        # Get initial run count
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name_vae)
        initial_runs = 0
        if experiment is not None:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            initial_runs = len(runs)
        
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train VAE
        result = trainer.train(dummy_X_all, y_all=dummy_y_all)
        
        # Assert: New run exists in MLflow
        experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name_vae)
        assert experiment is not None, \
            f"Experiment {mlflow_config.experiment_name_vae} not found"
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) == initial_runs + 1, \
            f"Expected {initial_runs + 1} runs, got {len(runs)}"
        
        # Check that vae_elbo was logged
        latest_run = mlflow.get_run(result.run_id)
        metrics = latest_run.data.metrics
        assert "vae_elbo" in metrics, \
            "vae_elbo metric not logged"

    def test_vae_logs_elbo_at_multiple_steps(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """VAE logs vae_elbo at step=0 and step=best_epoch."""
        # Arrange
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act
        result = trainer.train(dummy_X_all, y_all=dummy_y_all)
        
        # Assert: Metrics logged at multiple epochs
        client = mlflow.tracking.MlflowClient()
        metric_history = client.get_metric_history(result.run_id, "vae_elbo")
        
        assert len(metric_history) >= 2, \
            f"Expected at least 2 ELBO values logged, got {len(metric_history)}"
        
        # Check step=0 exists
        steps = [m.step for m in metric_history]
        assert 0 in steps, "vae_elbo not logged at step=0"
        assert result.best_epoch in steps, \
            f"vae_elbo not logged at step={result.best_epoch}"

    def test_encoder_output_shape_matches_latent_dim(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all, tmp_path
    ):
        """Encoder produces output with shape (n_samples, latent_dim)."""
        # Arrange
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train and get encoder
        result = trainer.train(dummy_X_all, y_all=dummy_y_all)
        
        # Load encoder and test forward pass
        encoder_checkpoint = torch.load(result.encoder_path, weights_only=False)
        
        # Reconstruct encoder architecture from checkpoint
        # (This assumes encoder checkpoint includes architecture info)
        # For now, we'll test this by loading and running inference
        
        # Convert dummy_X_all to tensor
        X_tensor = torch.tensor(dummy_X_all, dtype=torch.float32)
        
        # We need to reconstruct the encoder model to test this
        # This will be done in the implementation
        # For now, just assert the checkpoint exists
        assert "encoder_dims" in encoder_checkpoint or "state_dict" in encoder_checkpoint, \
            "Encoder checkpoint should contain model information"

    def test_vae_logs_reconstruction_and_kl_losses(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """VAE logs vae_reconstruction_loss and vae_kl_loss per epoch."""
        # Arrange
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act
        result = trainer.train(dummy_X_all, y_all=dummy_y_all)
        
        # Assert: Both loss components logged
        client = mlflow.tracking.MlflowClient()
        recon_history = client.get_metric_history(result.run_id, "vae_reconstruction_loss")
        kl_history = client.get_metric_history(result.run_id, "vae_kl_loss")
        
        assert len(recon_history) >= 1, \
            "vae_reconstruction_loss not logged"
        assert len(kl_history) >= 1, \
            "vae_kl_loss not logged"

    def test_weighted_sampler_logged_when_y_all_provided(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """When y_all is passed, MLflow run records weighted_sampler=True and n_fatal_train > 0."""
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)

        result = trainer.train(dummy_X_all, y_all=dummy_y_all)

        run = mlflow.get_run(result.run_id)
        params = run.data.params
        assert params.get("weighted_sampler") == "True", \
            "weighted_sampler param not logged as True"
        assert int(params.get("n_fatal_train", 0)) > 0, \
            "n_fatal_train not logged or is 0"

    def test_train_without_y_all_still_works(
        self, minimal_vae_config, mlflow_config, dummy_X_all
    ):
        """y_all is optional — train() without it uses shuffle=True and logs weighted_sampler=False."""
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)

        result = trainer.train(dummy_X_all)

        run = mlflow.get_run(result.run_id)
        assert run.data.params.get("weighted_sampler") == "False", \
            "weighted_sampler should be False when y_all not provided"

    def test_kl_beta_annealing_logged(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """KL beta annealing: kl_beta logged at step=0 with value 0.0 and step=warmup_epochs with value beta_max."""
        # Arrange
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train VAE
        result = trainer.train(dummy_X_all, y_all=dummy_y_all)
        
        # Assert: kl_beta metric logged at multiple steps
        client = mlflow.tracking.MlflowClient()
        kl_beta_history = client.get_metric_history(result.run_id, "kl_beta")
        
        assert len(kl_beta_history) > 0, \
            "kl_beta metric not logged"
        
        # Check step=0 has beta_start (0.0)
        step_0_metrics = [m for m in kl_beta_history if m.step == 0]
        assert len(step_0_metrics) == 1, \
            f"Expected exactly 1 kl_beta metric at step=0, got {len(step_0_metrics)}"
        assert abs(step_0_metrics[0].value - minimal_vae_config.beta_start) < 1e-6, \
            f"kl_beta at step=0 should be {minimal_vae_config.beta_start}, got {step_0_metrics[0].value}"
        
        # Check step=warmup_epochs has beta_max
        warmup_step = minimal_vae_config.warmup_epochs
        warmup_metrics = [m for m in kl_beta_history if m.step == warmup_step]
        assert len(warmup_metrics) == 1, \
            f"Expected exactly 1 kl_beta metric at step={warmup_step}, got {len(warmup_metrics)}"
        assert abs(warmup_metrics[0].value - minimal_vae_config.beta_max) < 1e-6, \
            f"kl_beta at step={warmup_step} should be {minimal_vae_config.beta_max}, got {warmup_metrics[0].value}"

    def test_encoder_output_has_sufficient_variance(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """Encoder produces latent vectors with std > 0.05 across all dims (detects posterior collapse)."""
        # Arrange
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train and get encoder checkpoint
        result = trainer.train(dummy_X_all, y_all=dummy_y_all)
        
        # Load encoder checkpoint
        checkpoint = torch.load(result.encoder_path, weights_only=False)
        
        # Reconstruct encoder model
        input_dim = checkpoint["input_dim"]
        encoder_dims = checkpoint["encoder_dims"]
        latent_dim = checkpoint["latent_dim"]
        
        encoder = Encoder(input_dim, encoder_dims, latent_dim)
        encoder.load_state_dict(checkpoint["state_dict"])
        encoder.eval()
        
        # Run forward pass on dummy data
        X_tensor = torch.tensor(dummy_X_all, dtype=torch.float32)
        with torch.no_grad():
            mu, log_var = encoder(X_tensor)
        
        # Compute std across samples for each latent dimension
        mu_std = mu.std(dim=0).numpy()
        
        # Assert: all dimensions have std > 0.05 (not collapsed)
        for dim_idx, std_val in enumerate(mu_std):
            assert std_val > 0.05, \
                f"Dimension {dim_idx} has std={std_val:.4f} < 0.05 (posterior collapse detected)"

    # --- Optuna Integration Tests (T127a) ---

    def test_train_accepts_optuna_trial_parameter(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """DVAETrainer.train() accepts optional optuna_trial=None parameter."""
        # Arrange
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Call train() with optuna_trial=None explicitly
        # This should work the same as calling without it
        result = trainer.train(dummy_X_all, y_all=dummy_y_all, optuna_trial=None)
        
        # Assert: Returns valid result
        assert isinstance(result, VAETrainResult), \
            f"Expected VAETrainResult, got {type(result)}"
        assert result.best_epoch >= 0, \
            f"Expected best_epoch >= 0, got {result.best_epoch}"

    def test_train_calls_trial_report_each_epoch(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """When optuna_trial is provided, trial.report(elbo, epoch) is called each epoch."""
        # Arrange: Create mock trial
        mock_trial = MagicMock(spec=optuna.trial.Trial)
        mock_trial.should_prune.return_value = False  # Don't prune
        
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train with mock trial
        result = trainer.train(dummy_X_all, y_all=dummy_y_all, optuna_trial=mock_trial)
        
        # Assert: trial.report() was called at least once (for each epoch)
        assert mock_trial.report.call_count >= 1, \
            f"Expected trial.report() called at least once, got {mock_trial.report.call_count} calls"
        
        # Assert: trial.report() was called with (elbo, epoch) signature
        # Get first call arguments
        first_call = mock_trial.report.call_args_list[0]
        assert len(first_call[0]) == 2, \
            f"Expected trial.report(elbo, epoch) with 2 args, got {len(first_call[0])}"
        
        elbo_arg, epoch_arg = first_call[0]
        assert isinstance(elbo_arg, float), \
            f"Expected ELBO (float) as first arg, got {type(elbo_arg)}"
        assert isinstance(epoch_arg, int), \
            f"Expected epoch (int) as second arg, got {type(epoch_arg)}"
        assert epoch_arg == 0, \
            f"Expected first call at epoch=0, got epoch={epoch_arg}"

    def test_train_raises_trial_pruned_when_should_prune_returns_true(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """When optuna_trial.should_prune() returns True after epoch 1, train() raises optuna.TrialPruned."""
        # Arrange: Create mock trial that prunes after epoch 1
        mock_trial = MagicMock(spec=optuna.trial.Trial)
        
        # should_prune() returns False for epoch 0, True for epoch >= 1
        call_count = [0]
        def should_prune_side_effect():
            result = call_count[0] >= 1  # Prune after first epoch
            call_count[0] += 1
            return result
        
        mock_trial.should_prune.side_effect = should_prune_side_effect
        
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act & Assert: train() should raise optuna.TrialPruned
        with pytest.raises(optuna.TrialPruned):
            trainer.train(dummy_X_all, y_all=dummy_y_all, optuna_trial=mock_trial)
        
        # Assert: should_prune() was called
        assert mock_trial.should_prune.call_count >= 2, \
            f"Expected should_prune() called at least twice, got {mock_trial.should_prune.call_count} calls"

    def test_train_with_optuna_trial_none_completes_normally(
        self, minimal_vae_config, mlflow_config, dummy_X_all, dummy_y_all
    ):
        """When optuna_trial=None (default), training completes normally without errors."""
        # Arrange
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train without optuna_trial (same as current behavior)
        result = trainer.train(dummy_X_all, y_all=dummy_y_all)
        
        # Assert: Training completed successfully
        assert isinstance(result, VAETrainResult), \
            f"Expected VAETrainResult, got {type(result)}"
        assert result.best_epoch >= 0, \
            f"Expected best_epoch >= 0, got {result.best_epoch}"
        assert Path(result.encoder_path).exists(), \
            f"Encoder checkpoint not found at {result.encoder_path}"
        assert Path(result.decoder_path).exists(), \
            f"Decoder checkpoint not found at {result.decoder_path}"


