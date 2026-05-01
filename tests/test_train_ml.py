"""Tests for train_ml stage - XGBoost on Z-space."""
import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pytest
from xgboost import XGBClassifier

from src.config import MLflowConfig, ModelConfig
from src.train_ml.trainer import MLTrainer, MLTrainResult


class TestTrainML:
    """Boundary tests for the ML training stage (XGBoost on Z-space)."""

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

    def test_ml_trainer_returns_train_result(
        self, mlflow_config, model_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """Given dummy Z splits, MLTrainer.train() returns MLTrainResult with best_run_id."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=model_config,
            seeds=[0],  # single seed for fast test
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: Returns MLTrainResult
        assert isinstance(result, MLTrainResult), \
            f"Expected MLTrainResult, got {type(result)}"
        assert result.best_run_id is not None, \
            "best_run_id should be set"
        assert result.model_path is not None, \
            "model_path should be set"

    def test_ml_trainer_logs_exactly_one_run_with_single_seed(
        self, mlflow_config, model_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """With seeds=[0], exactly 1 MLflow run should be logged to crash-severity-ml."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        experiment = mlflow.set_experiment(mlflow_config.experiment_name_ml)
        
        # Count runs before training
        runs_before = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        count_before = len(runs_before)
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=model_config,
            seeds=[0],
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: Exactly 1 new run
        runs_after = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        count_after = len(runs_after)
        
        assert count_after == count_before + 1, \
            f"Expected exactly 1 new run, got {count_after - count_before}"

    def test_ml_trainer_tags_run_with_seed_and_model_type(
        self, mlflow_config, model_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """MLflow run should be tagged with seed=0 and model_type=xgboost."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name_ml)
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=model_config,
            seeds=[0],
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: Run should have correct tags
        run = mlflow.get_run(result.best_run_id)
        tags = run.data.tags
        
        assert "seed" in tags, \
            "Run should be tagged with 'seed'"
        assert tags["seed"] == "0", \
            f"Expected seed tag = '0', got '{tags.get('seed')}'"
        assert "model_type" in tags, \
            "Run should be tagged with 'model_type'"
        assert tags["model_type"] == "xgboost", \
            f"Expected model_type tag = 'xgboost', got '{tags.get('model_type')}'"

    def test_ml_trainer_logs_mandatory_metrics(
        self, mlflow_config, model_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """MLTrainer should log eout_macro_f1, eout_fatal_recall, ein_macro_f1, generalisation_gap."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name_ml)
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=model_config,
            seeds=[0],
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: MLflow run should have mandatory metrics
        run = mlflow.get_run(result.best_run_id)
        metrics = run.data.metrics
        
        assert "eout_macro_f1" in metrics, \
            "Missing mandatory metric: eout_macro_f1"
        assert "eout_fatal_recall" in metrics, \
            "Missing mandatory metric: eout_fatal_recall"
        assert "ein_macro_f1" in metrics, \
            "Missing mandatory metric: ein_macro_f1"
        assert "generalisation_gap" in metrics, \
            "Missing mandatory metric: generalisation_gap"

    def test_ml_trainer_logs_per_class_matrix_artifact(
        self, mlflow_config, model_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """MLTrainer should log per_class_matrix.json as MLflow artifact."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name_ml)
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=model_config,
            seeds=[0],
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: Artifact should exist
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(result.best_run_id)
        artifact_names = [a.path for a in artifacts]
        
        assert "per_class_matrix.json" in artifact_names, \
            f"Missing per_class_matrix.json artifact. Found: {artifact_names}"

    def test_ml_trainer_saves_loadable_xgboost_model(
        self, mlflow_config, model_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """models/best_ml_model.pkl should exist and be loadable as XGBClassifier."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=model_config,
            seeds=[0],
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: Model file should exist and be loadable
        assert Path(result.model_path).exists(), \
            f"Model file not found at {result.model_path}"
        
        with open(result.model_path, "rb") as f:
            model = pickle.load(f)
        
        assert isinstance(model, XGBClassifier), \
            f"Expected XGBClassifier, got {type(model)}"

    def test_ml_trainer_does_not_use_autolog(
        self, mlflow_config, model_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """MLTrainer should NOT use mlflow.sklearn.autolog() or mlflow.autolog()."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name_ml)
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=model_config,
            seeds=[0],
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: No autolog parameters should be present
        run = mlflow.get_run(result.best_run_id)
        params = run.data.params
        
        # Autolog typically adds params like 'estimator_name', 'estimator_class'
        autolog_indicators = ['estimator_name', 'estimator_class', 'sklearn_version']
        for indicator in autolog_indicators:
            assert indicator not in params, \
                f"Autolog indicator '{indicator}' found in params — autolog should be disabled"
