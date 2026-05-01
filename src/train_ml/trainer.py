from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.metrics import compute_class_weights, per_class_matrix
from src.plots import log_confusion_matrix, log_roc_curve

_CLASS_NAMES = ["PDO", "Injury", "Fatal"]


@dataclass
class MLTrainResult:
    best_run_id: str
    model_path: str
    best_seed: int
    eout_macro_f1: float


class MLTrainer:
    """XGBoost multi-seed training on Z-space with MLflow tracking.

    Trains one XGBoost classifier per seed, logs all metrics to the
    crash-severity-ml experiment, and saves the best-seed model.
    autolog is explicitly disabled; all metrics are logged manually.

    Public interface
    ----------------
    train(Z_train, y_train, Z_val, y_val, Z_test, y_test) → MLTrainResult
    """

    def __init__(self, mlflow_config, model_config, seeds: list[int]) -> None:
        self._mlflow_config = mlflow_config
        self._model_config = model_config
        self._seeds = seeds

    def train(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_val: np.ndarray,
        y_val: np.ndarray,
        Z_test: np.ndarray,
        y_test: np.ndarray,
    ) -> MLTrainResult:
        """Train XGBoost across N seeds; return best by eout_macro_f1."""
        mlflow.sklearn.autolog(disable=True)

        # Compute sample weights for class imbalance
        sample_weights = compute_class_weights(y_train, n_classes=self._model_config.n_classes)
        # Map class weights to per-sample weights
        sample_weight_array = np.array([sample_weights[y] for y in y_train])

        mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)
        mlflow.set_experiment(self._mlflow_config.experiment_name_ml)

        best_f1 = -1.0
        best_result = None

        for seed in self._seeds:
            result = self._train_single_seed(
                seed=seed,
                Z_train=Z_train,
                y_train=y_train,
                Z_val=Z_val,
                y_val=y_val,
                Z_test=Z_test,
                y_test=y_test,
                sample_weight_array=sample_weight_array,
            )
            
            # Track best seed by eout_macro_f1
            if result.eout_macro_f1 > best_f1:
                best_f1 = result.eout_macro_f1
                best_result = result

        return best_result

    def _train_single_seed(
        self,
        seed: int,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_val: np.ndarray,
        y_val: np.ndarray,
        Z_test: np.ndarray,
        y_test: np.ndarray,
        sample_weight_array: np.ndarray,
    ) -> MLTrainResult:
        """Train a single XGBoost model with given seed."""
        # Create XGBoost classifier
        clf = XGBClassifier(
            objective="multi:softprob",
            num_class=self._model_config.n_classes,
            random_state=seed,
            early_stopping_rounds=10,
            eval_metric="mlogloss",
            verbosity=0,
        )

        with mlflow.start_run(run_name=f"xgb_seed_{seed}") as run:
            # Tag the run
            mlflow.set_tags({
                "seed": str(seed),
                "model_type": "xgboost",
            })

            # Log parameters
            mlflow.log_params({
                "seed": seed,
                "objective": "multi:softprob",
                "num_class": self._model_config.n_classes,
                "early_stopping_rounds": 10,
            })

            # Fit with sample weights and validation set
            clf.fit(
                Z_train,
                y_train,
                sample_weight=sample_weight_array,
                eval_set=[(Z_val, y_val)],
                verbose=False,
            )

            # Predictions
            y_train_pred = clf.predict(Z_train)
            y_test_pred = clf.predict(Z_test)
            test_probs = clf.predict_proba(Z_test)

            # Mandatory metrics
            ein_macro_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
            eout_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
            
            # Fatal recall (class 2)
            fatal_mask = y_test == 2
            eout_fatal_recall = (
                float((y_test_pred[fatal_mask] == 2).sum() / fatal_mask.sum())
                if fatal_mask.sum() > 0 else 0.0
            )

            mlflow.log_metrics({
                "ein_macro_f1": ein_macro_f1,
                "eout_macro_f1": eout_macro_f1,
                "eout_fatal_recall": eout_fatal_recall,
                "generalisation_gap": ein_macro_f1 - eout_macro_f1,
            })

            # Visual diagnostics
            log_confusion_matrix(y_test, y_test_pred, _CLASS_NAMES)
            log_roc_curve(y_test, test_probs, _CLASS_NAMES)

            # Per-class matrix JSON artifact
            matrix_path = Path("per_class_matrix.json")
            matrix_path.write_text(
                json.dumps(per_class_matrix(y_test, y_test_pred, _CLASS_NAMES), indent=2)
            )
            mlflow.log_artifact(str(matrix_path))
            matrix_path.unlink()

            # Save model
            model_path = f"models/xgb_model_seed{seed}.pkl"
            Path("models").mkdir(exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(clf, f)

            return MLTrainResult(
                best_run_id=run.info.run_id,
                model_path=model_path,
                best_seed=seed,
                eout_macro_f1=eout_macro_f1,
            )
