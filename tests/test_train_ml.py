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

    def test_fatal_threshold_overrides_argmax(
        self, mlflow_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """T122b: fatal_threshold=0.15 predicts Fatal when prob=0.2; threshold=0.5 does not."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        # We need a row where Fatal prob is 0.2
        # We will mock the classifier and its predict_proba
        from unittest.mock import MagicMock, patch
        
        mock_clf = MagicMock()
        # Mock predict_proba to return 0.2 for Fatal (class 2) on the first row
        # Probs: [0.4, 0.4, 0.2] -> Argmax is 0 or 1, but Threshold 0.15 makes it 2
        mock_probs = np.array([[0.4, 0.4, 0.2]])
        mock_clf.predict_proba.return_value = mock_probs
        
        # Config with threshold 0.15
        config_low = ModelConfig(
            n_classes=3,
            n_select=10,
            macro_f1_threshold=0.35,
            fatal_recall_threshold=0.50,
            fatal_threshold=0.15,
        )
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=config_low,
            seeds=[0],
        )
        
        # Act
        preds = trainer._apply_threshold(mock_probs)
        
        # Assert: Should be Fatal (class 2)
        assert preds[0] == 2, \
            f"Expected Fatal (2) with threshold 0.15 and prob 0.2, got {preds[0]}"
            
        # Config with threshold 0.5
        config_high = ModelConfig(
            n_classes=3,
            n_select=10,
            macro_f1_threshold=0.35,
            fatal_recall_threshold=0.50,
            fatal_threshold=0.50,
        )
        trainer_high = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=config_high,
            seeds=[0],
        )
        preds_high = trainer_high._apply_threshold(mock_probs)
        
        # Assert: Should NOT be Fatal (argmax of [0.4, 0.4] is 0 or 1)
        assert preds_high[0] != 2, \
            f"Expected non-Fatal with threshold 0.5 and prob 0.2, got {preds_high[0]}"

    def test_seed_selection_uses_val_not_test(
        self, mlflow_config, model_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """T122b RED: seed selection uses eval_macro_f1 (val), not eout_macro_f1 (test)."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        mlflow.set_tracking_uri(mlflow_config.tracking_uri)
        mlflow.set_experiment(mlflow_config.experiment_name_ml)
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=model_config,
            seeds=[42, 43],  # two seeds to test selection
        )
        
        # Act
        result = trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)
        
        # Assert: the winning run should have logged eval_macro_f1 (val metric)
        run = mlflow.get_run(result.best_run_id)
        metrics = run.data.metrics
        
        # RED expectation: eval_macro_f1 should exist (validation metric)
        # Current implementation doesn't compute val predictions, so this will fail
        assert "eval_macro_f1" in metrics, \
            ("Seed selection should use eval_macro_f1 (val), but metric not found. "
             "Current implementation likely uses eout_macro_f1 (test) — violates constitution II.")
        
        # Additional check: eval_fatal_recall should also be logged
        assert "eval_fatal_recall" in metrics, \
            "eval_fatal_recall (val) should be logged for transparency."

    def test_ml_trainer_uses_custom_focal_loss_when_enabled(
        self, mlflow_config, dummy_Z_splits, dummy_y_splits, tmp_path
    ):
        """T125b RED: MLTrainer should pass objective='focal_loss_grad_hess' to XGBClassifier when enabled."""
        # Arrange
        Z_train, Z_val, Z_test = dummy_Z_splits
        y_train, y_val, y_test = dummy_y_splits
        
        config = ModelConfig(
            n_classes=3,
            n_select=10,
            macro_f1_threshold=0.35,
            fatal_recall_threshold=0.50,
            focal_loss_enabled=True,
            focal_loss_gamma=3.0,
        )
        
        trainer = MLTrainer(
            mlflow_config=mlflow_config,
            model_config=config,
            seeds=[0],
        )
        
        # Patch XGBClassifier and pickle.dump so the mock doesn't reach serialisation
        from unittest.mock import patch, MagicMock
        with patch("src.train_ml.trainer.XGBClassifier") as mock_xgb, \
             patch("src.train_ml.trainer.pickle.dump"):
            mock_instance = mock_xgb.return_value
            mock_instance.predict_proba.side_effect = [
                np.zeros((len(Z_train), 3)),
                np.zeros((len(Z_val), 3)),
                np.zeros((len(Z_test), 3)),
            ]
            mock_instance.predict.side_effect = [
                np.zeros(len(Z_train), dtype=int),
                np.zeros(len(Z_val), dtype=int),
                np.zeros(len(Z_test), dtype=int),
            ]

            # Act
            trainer.train(Z_train, y_train, Z_val, y_val, Z_test, y_test)

            # Assert: objective must be a callable (focal loss fn), not a string
            args, kwargs = mock_xgb.call_args
            assert "objective" in kwargs, "objective param missing from XGBClassifier constructor"
            assert callable(kwargs["objective"]), "objective should be a focal loss callable"

    def test_focal_loss_grad_hess_returns_correct_shapes(self):
        """T125b RED: focal_loss_grad_hess(gamma, alpha)(y_true, y_pred) -> (grad, hess) both shape (N*K,)."""
        from src.metrics import focal_loss_grad_hess

        N = 50
        K = 3
        rng = np.random.default_rng(0)
        y_pred = rng.standard_normal((N * K,)).astype(np.float32)  # Raw margins (logits)
        y_true = rng.integers(0, K, size=N).astype(np.int32)

        # Instantiate objective
        obj = focal_loss_grad_hess(gamma=2.0)
        grad, hess = obj(y_true, y_pred)

        assert grad.shape == (N, K), f"Expected grad shape {(N, K)}, got {grad.shape}"
        assert hess.shape == (N, K), f"Expected hess shape {(N, K)}, got {hess.shape}"

        assert np.isfinite(grad).all(), "grad contains non-finite values"
        assert np.isfinite(hess).all(), "hess contains non-finite values"
        assert (hess > 0).all(), "hess should be strictly positive"
