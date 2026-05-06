from __future__ import annotations

import json
import pickle

import cloudpickle
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.metrics import (
    compute_class_weights,
    focal_loss_grad_hess,
    per_class_matrix,
)
from src.plots import log_confusion_matrix, log_roc_curve

_CLASS_NAMES = ["PDO", "Injury", "Fatal"]


@dataclass
class MLTrainResult:
    best_run_id: str
    model_path: str
    best_seed: int
    eout_macro_f1: float
    eval_macro_f1: float  # val F1 for seed selection
    eval_fatal_recall: float = 0.0  # val fatal recall for fitness gate


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

    def _apply_threshold(self, probs: np.ndarray) -> np.ndarray:
        """Apply fatal_threshold to probabilities; return class predictions.
        
        If Fatal prob >= threshold, predict Fatal (class 2).
        Otherwise, argmax between PDO (0) and Injury (1).
        """
        fatal_mask = probs[:, 2] >= self._model_config.fatal_threshold
        predictions = np.where(
            fatal_mask,
            2,  # Fatal
            np.argmax(probs[:, :2], axis=1)  # argmax(PDO, Injury)
        )
        return predictions

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
            
            # Track best seed by eval_macro_f1 (val) — constitution II compliance
            if result.eval_macro_f1 > best_f1:
                best_f1 = result.eval_macro_f1
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
        clf_params = {
            "random_state": seed,
            "early_stopping_rounds": 10,
            "num_class": self._model_config.n_classes,
            "verbosity": 0,
        }

        if self._model_config.focal_loss_enabled:
            alpha = compute_class_weights(y_train, self._model_config.n_classes)
            clf_params["objective"] = focal_loss_grad_hess(
                gamma=self._model_config.focal_loss_gamma,
                alpha=alpha
            )
            clf_params["eval_metric"] = "merror"
        else:
            clf_params["objective"] = "multi:softprob"
            clf_params["eval_metric"] = "mlogloss"

        clf = XGBClassifier(**clf_params)

        active_run = mlflow.active_run()
        with mlflow.start_run(run_name=f"xgb_seed_{seed}", nested=active_run is not None) as run:
            # Tag the run
            mlflow.set_tags({
                "seed": str(seed),
                "model_type": "xgboost",
                "loss_type": "focal" if self._model_config.focal_loss_enabled else "ce",
            })

            # Log parameters
            log_params = {
                "seed": seed,
                "objective": "focal_loss_grad_hess" if self._model_config.focal_loss_enabled else "multi:softprob",
                "early_stopping_rounds": 10,
            }
            if not self._model_config.focal_loss_enabled:
                log_params["num_class"] = self._model_config.n_classes
            else:
                log_params["focal_gamma"] = self._model_config.focal_loss_gamma
            
            mlflow.log_params(log_params)

            # Fit with sample weights and validation set
            clf.fit(
                Z_train,
                y_train,
                sample_weight=sample_weight_array,
                eval_set=[(Z_val, y_val)],
                verbose=False,
            )

            # Get probabilities for all splits
            train_probs = clf.predict_proba(Z_train)
            val_probs = clf.predict_proba(Z_val)
            test_probs = clf.predict_proba(Z_test)

            # Apply threshold-based prediction
            y_train_pred = self._apply_threshold(train_probs)
            y_val_pred = self._apply_threshold(val_probs)
            y_test_pred = self._apply_threshold(test_probs)

            # Train metrics (ein)
            ein_macro_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
            
            # Val metrics (eval) — for seed selection
            eval_macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
            val_fatal_mask = y_val == 2
            eval_fatal_recall = (
                float((y_val_pred[val_fatal_mask] == 2).sum() / val_fatal_mask.sum())
                if val_fatal_mask.sum() > 0 else 0.0
            )
            
            # Test metrics (eout) — for final reporting only
            eout_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
            test_fatal_mask = y_test == 2
            eout_fatal_recall = (
                float((y_test_pred[test_fatal_mask] == 2).sum() / test_fatal_mask.sum())
                if test_fatal_mask.sum() > 0 else 0.0
            )

            # Log all metrics
            mlflow.log_metrics({
                "ein_macro_f1": ein_macro_f1,
                "eval_macro_f1": eval_macro_f1,
                "eval_fatal_recall": eval_fatal_recall,
                "eout_macro_f1": eout_macro_f1,
                "eout_fatal_recall": eout_fatal_recall,
                "generalisation_gap": ein_macro_f1 - eout_macro_f1,
                "fatal_threshold": self._model_config.fatal_threshold,
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
                cloudpickle.dump(clf, f)

            return MLTrainResult(
                best_run_id=run.info.run_id,
                model_path=model_path,
                best_seed=seed,
                eout_macro_f1=eout_macro_f1,
                eval_macro_f1=eval_macro_f1,
                eval_fatal_recall=eval_fatal_recall,
            )
