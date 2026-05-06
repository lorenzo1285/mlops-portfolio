"""Tests for train_dl stage - Shallow MLP on Z-space."""
import json
import os
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch

from src.config import DLConfig, MLflowConfig, ModelConfig
from src.train_dl.trainer import DLTrainer, DLTrainResult


class TestTrainDL:
    """Boundary tests for the DL training stage (shallow MLP on Z-space)."""

    @pytest.fixture
    def minimal_dl_config(self):
        """Minimal DL config for fast testing."""
        return DLConfig(
            input_dim=8,
            hidden_dim=16,
            dropout_p=0.1,
            epochs=3,
            patience=10,
            batch_size=32,
            lr=0.001,
            experiment_name="test-dl",
            focal_loss_enabled=False,
            focal_loss_gamma=2.0,
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
    def model_config(self):
        """Model config for 3-class classification."""
        return ModelConfig(
            n_classes=3,
            n_select=10,
            macro_f1_threshold=0.35,
            fatal_recall_threshold=0.50,
        )

    @pytest.fixture
    def dummy_Z_splits(self):
        """Create synthetic Z vectors (60 train, 20 val, 20 test) × 8 latent dims."""
        np.random.seed(42)
        Z_train = np.random.randn(60, 8).astype(np.float32)
        Z_val = np.random.randn(20, 8).astype(np.float32)
        Z_test = np.random.randn(20, 8).astype(np.float32)
        return Z_train, Z_val, Z_test

    @pytest.fixture
    def dummy_y_splits(self):
        """Create balanced 3-class labels (at least 5 per class in train)."""
        # Train: 20 PDO (0), 20 Injury (1), 20 Fatal (2)
        y_train = np.array([0]*20 + [1]*20 + [2]*20, dtype=np.int64)
        # Val: 7/7/6
        y_val = np.array([0]*7 + [1]*7 + [2]*6, dtype=np.int64)
        # Test: 7/7/6
        y_test = np.array([0]*7 + [1]*7 + [2]*6, dtype=np.int64)
        return y_train, y_val, y_test

    def test_dl_trainer_returns_train_result(
        self, minimal_dl_config, mlflow_config, model_config, 
        dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """Given dummy Z splits, DLTrainer.train() returns DLTrainResult with best_epoch >= 1."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        trainer = DLTrainer(
            dl_config=minimal_dl_config,
            mlflow_config=mlflow_config,
            seeds=[0],  # single seed for fast test
            model_config=model_config,
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: Returns DLTrainResult
        assert isinstance(result, DLTrainResult), \
            f"Expected DLTrainResult, got {type(result)}"
        assert result.best_epoch >= 1, \
            f"Expected best_epoch >= 1, got {result.best_epoch}"
        assert result.model_path is not None, \
            "model_path should be set"

    def test_dl_trainer_saves_loadable_checkpoint(
        self, minimal_dl_config, mlflow_config, model_config,
        dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """Trained MLP checkpoint should be saved and loadable."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        model_path = tmp_path / "mlp_model.pth"
        trainer = DLTrainer(
            dl_config=minimal_dl_config,
            mlflow_config=mlflow_config,
            seeds=[0],
            model_config=model_config,
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: Checkpoint file should exist and be loadable
        assert Path(result.model_path).exists(), \
            f"Model checkpoint not found at {result.model_path}"
        
        checkpoint = torch.load(result.model_path)
        assert "state_dict" in checkpoint or "model_state_dict" in checkpoint, \
            "Checkpoint should contain model state dict"

    def test_dl_trainer_logs_mandatory_metrics_to_mlflow(
        self, minimal_dl_config, mlflow_config, model_config,
        dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """DLTrainer should log mandatory metrics (eout_macro_f1, eout_fatal_recall, etc.) to MLflow."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(minimal_dl_config.experiment_name)
        
        trainer = DLTrainer(
            dl_config=minimal_dl_config,
            mlflow_config=mlflow_config,
            seeds=[0],
            model_config=model_config,
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: MLflow run should exist with mandatory metrics
        run = mlflow.get_run(result.run_id)
        metrics = run.data.metrics
        
        assert "eout_macro_f1" in metrics, \
            "Missing mandatory metric: eout_macro_f1"
        assert "eout_fatal_recall" in metrics, \
            "Missing mandatory metric: eout_fatal_recall"
        assert "ein_macro_f1" in metrics, \
            "Missing mandatory metric: ein_macro_f1"
        assert "generalisation_gap" in metrics, \
            "Missing mandatory metric: generalisation_gap"

    def test_dl_trainer_logs_per_class_matrix_artifact(
        self, minimal_dl_config, mlflow_config, model_config,
        dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """DLTrainer should log per_class_matrix.json as MLflow artifact."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(minimal_dl_config.experiment_name)
        
        trainer = DLTrainer(
            dl_config=minimal_dl_config,
            mlflow_config=mlflow_config,
            seeds=[0],
            model_config=model_config,
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: Artifact should exist
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(result.run_id)
        artifact_names = [a.path for a in artifacts]
        
        assert "per_class_matrix.json" in artifact_names, \
            f"Missing per_class_matrix.json artifact. Found: {artifact_names}"

    def test_dl_trainer_does_not_use_autolog(
        self, minimal_dl_config, mlflow_config, model_config,
        dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """DLTrainer should NOT use mlflow.sklearn.autolog() or mlflow.autolog()."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(minimal_dl_config.experiment_name)
        
        trainer = DLTrainer(
            dl_config=minimal_dl_config,
            mlflow_config=mlflow_config,
            seeds=[0],
            model_config=model_config,
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: No autolog parameters should be present
        run = mlflow.get_run(result.run_id)
        params = run.data.params
        
        # Autolog typically adds params like 'estimator_name', 'estimator_class'
        autolog_indicators = ['estimator_name', 'estimator_class', 'sklearn_version']
        for indicator in autolog_indicators:
            assert indicator not in params, \
                f"Autolog indicator '{indicator}' found in params — autolog should be disabled"

    def test_balanced_focal_loss_returns_scalar(self):
        """BalancedFocalLoss(gamma, weight) should return a scalar loss."""
        # Arrange
        from src.train_dl.losses import BalancedFocalLoss
        
        batch_size = 8
        n_classes = 3
        gamma = 2.0
        weight = torch.tensor([1.0, 1.5, 2.0])  # class weights
        
        loss_fn = BalancedFocalLoss(gamma=gamma, weight=weight)
        logits = torch.randn(batch_size, n_classes, requires_grad=True)
        targets = torch.randint(0, n_classes, (batch_size,))
        
        # Act
        loss = loss_fn(logits, targets)
        
        # Assert: returns scalar tensor
        assert isinstance(loss, torch.Tensor), \
            f"Expected torch.Tensor, got {type(loss)}"
        assert loss.dim() == 0, \
            f"Expected scalar (0-dim), got shape {loss.shape}"
        assert loss.requires_grad, \
            "Loss should be differentiable"

    def test_balanced_focal_loss_focal_property(self):
        """Focal property: confident correct predictions have lower loss than hard incorrect ones."""
        # Arrange
        from src.train_dl.losses import BalancedFocalLoss
        
        gamma = 2.0
        weight = torch.ones(3)  # equal weights to isolate focal effect
        loss_fn = BalancedFocalLoss(gamma=gamma, weight=weight)
        
        # Case 1: Confident correct prediction (high logit for true class)
        logits_confident = torch.tensor([[10.0, 0.0, 0.0]])  # strongly predicts class 0
        targets_confident = torch.tensor([0])  # true class is 0
        
        # Case 2: Hard incorrect prediction (low logit for true class)
        logits_hard = torch.tensor([[0.0, 10.0, 0.0]])  # strongly predicts class 1
        targets_hard = torch.tensor([0])  # true class is 0, but predicted 1
        
        # Act
        loss_confident = loss_fn(logits_confident, targets_confident)
        loss_hard = loss_fn(logits_hard, targets_hard)
        
        # Assert: focal property — confident correct loss < hard incorrect loss
        assert loss_confident.item() < loss_hard.item(), \
            f"Focal property violated: confident loss {loss_confident.item():.4f} >= hard loss {loss_hard.item():.4f}"

    def test_dl_trainer_uses_focal_loss_when_enabled(self, mlflow_config, model_config, dummy_Z_splits, dummy_y_splits):
        """DLTrainer logs loss_function=BalancedFocalLoss when enabled, CrossEntropyLoss when not."""
        from src.train_dl.losses import BalancedFocalLoss  # RED: ImportError until T133c

        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits

        mlflow.set_tracking_uri(mlflow_config.tracking_uri)

        config_enabled = DLConfig(
            input_dim=8, hidden_dim=16, dropout_p=0.1, epochs=2, patience=10,
            batch_size=32, lr=0.001, experiment_name="test-dl-focal-enabled",
            focal_loss_enabled=True, focal_loss_gamma=2.0,
        )
        config_disabled = DLConfig(
            input_dim=8, hidden_dim=16, dropout_p=0.1, epochs=2, patience=10,
            batch_size=32, lr=0.001, experiment_name="test-dl-focal-disabled",
            focal_loss_enabled=False, focal_loss_gamma=2.0,
        )

        result_enabled = DLTrainer(
            dl_config=config_enabled, mlflow_config=mlflow_config,
            seeds=[0], model_config=model_config,
        ).train(Z_train, y_train, Z_val, y_val, Z_test, y_test)

        result_disabled = DLTrainer(
            dl_config=config_disabled, mlflow_config=mlflow_config,
            seeds=[0], model_config=model_config,
        ).train(Z_train, y_train, Z_val, y_val, Z_test, y_test)

        assert mlflow.get_run(result_enabled.run_id).data.params.get("loss_function") == "BalancedFocalLoss", \
            "focal_loss_enabled=True must log loss_function=BalancedFocalLoss"
        assert mlflow.get_run(result_disabled.run_id).data.params.get("loss_function") == "CrossEntropyLoss", \
            "focal_loss_enabled=False must log loss_function=CrossEntropyLoss"
