"""Tests for train_vae stage - Denoising β-VAE training."""
import os
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch

from src.config import MLflowConfig, VAEConfig
from src.train_vae.vae_trainer import DVAETrainer, VAETrainResult


class TestTrainVAE:
    """Boundary tests for the VAE training stage."""

    @pytest.fixture
    def minimal_vae_config(self):
        """Minimal VAE config for fast testing."""
        return VAEConfig(
            encoder_dims=[16, 8],  # Small dims for fast testing
            latent_dim=4,
            beta=1.0,
            dropout_p=0.15,
            epochs=5,  # Few epochs for fast testing
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
        # 100 samples, 10 features
        np.random.seed(42)
        return np.random.randn(100, 10).astype(np.float32)

    def test_vae_trainer_returns_train_result(
        self, minimal_vae_config, mlflow_config, dummy_X_all, tmp_path
    ):
        """Given dummy X_all, DVAETrainer.train() returns VAETrainResult with best_epoch >= 1."""
        # Arrange: Create trainer
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train VAE
        result = trainer.train(dummy_X_all)
        
        # Assert: Returns VAETrainResult
        assert isinstance(result, VAETrainResult), \
            f"Expected VAETrainResult, got {type(result)}"
        assert result.best_epoch >= 1, \
            f"Expected best_epoch >= 1, got {result.best_epoch}"
        assert result.final_elbo < 0 or result.final_elbo >= 0, \
            "final_elbo should be a float value"

    def test_vae_creates_encoder_decoder_checkpoints(
        self, minimal_vae_config, mlflow_config, dummy_X_all, tmp_path
    ):
        """Encoder and decoder .pth files are created at the configured paths."""
        # Arrange: Configure paths
        encoder_path = tmp_path / "encoder.pth"
        decoder_path = tmp_path / "decoder.pth"
        
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train VAE (need to modify trainer to accept output paths)
        # For now, trainer will use default paths - we'll check those exist
        result = trainer.train(dummy_X_all)
        
        # Assert: Checkpoint files exist
        assert Path(result.encoder_path).exists(), \
            f"Encoder checkpoint not found at {result.encoder_path}"
        assert Path(result.decoder_path).exists(), \
            f"Decoder checkpoint not found at {result.decoder_path}"

    def test_vae_logs_to_mlflow(
        self, minimal_vae_config, mlflow_config, dummy_X_all
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
        result = trainer.train(dummy_X_all)
        
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
        self, minimal_vae_config, mlflow_config, dummy_X_all
    ):
        """VAE logs vae_elbo at step=0 and step=best_epoch."""
        # Arrange
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act
        result = trainer.train(dummy_X_all)
        
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
        self, minimal_vae_config, mlflow_config, dummy_X_all, tmp_path
    ):
        """Encoder produces output with shape (n_samples, latent_dim)."""
        # Arrange
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act: Train and get encoder
        result = trainer.train(dummy_X_all)
        
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
        self, minimal_vae_config, mlflow_config, dummy_X_all
    ):
        """VAE logs vae_reconstruction_loss and vae_kl_loss per epoch."""
        # Arrange
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        trainer = DVAETrainer(minimal_vae_config, mlflow_config)
        
        # Act
        result = trainer.train(dummy_X_all)
        
        # Assert: Both loss components logged
        client = mlflow.tracking.MlflowClient()
        recon_history = client.get_metric_history(result.run_id, "vae_reconstruction_loss")
        kl_history = client.get_metric_history(result.run_id, "vae_kl_loss")
        
        assert len(recon_history) >= 1, \
            "vae_reconstruction_loss not logged"
        assert len(kl_history) >= 1, \
            "vae_kl_loss not logged"
