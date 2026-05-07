"""GMM-based classifier on latent space."""
from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

from src.config import ABTestConfig, GMMConfig, MLflowConfig, ModelConfig
from src.metrics import per_class_matrix
from src.plots import log_confusion_matrix, log_roc_curve

_CLASS_NAMES = ["PDO", "Injury", "Fatal"]


@dataclass
class GMMTrainResult:
    run_id: str
    model_path: str
    best_seed: int
    eout_macro_f1: float
    eout_fatal_recall: float
    eval_macro_f1: float  # val — seed selection only, never exposed to evaluator


class GMMClassifier:
    """Per-class GMM MAP classifier: argmax_c [log p(Z|c) + log P(c)]."""

    def __init__(
        self,
        gmms: dict[int, GaussianMixture],
        log_priors: np.ndarray,
        fatal_prior_boost: float = 1.0,
    ):
        self._gmms = gmms
        self._log_priors = log_priors.copy()
        self._fatal_prior_boost = fatal_prior_boost
        self._classes = sorted(gmms.keys())

    def predict_log_posteriors(self, Z: np.ndarray) -> np.ndarray:
        """Return (N, n_classes) log-posterior matrix before argmax."""
        N = len(Z)
        log_posteriors = np.zeros((N, len(self._classes)))
        for col, class_label in enumerate(self._classes):
            log_prior = self._log_priors[class_label]
            # log(boost * P(2)) = log(boost) + log(P(2)) — boost in linear space
            if class_label == 2:
                log_prior = np.log(self._fatal_prior_boost) + log_prior
            log_posteriors[:, col] = self._gmms[class_label].score_samples(Z) + log_prior
        return log_posteriors

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """Return argmax class labels for latent vectors Z (N, latent_dim)."""
        return np.array(self._classes)[self.predict_log_posteriors(Z).argmax(axis=1)].astype(np.int64)


class GMMTrainer:
    """GMM multi-seed training on Z-space with MLflow tracking."""

    def __init__(
        self,
        gmm_config: GMMConfig,
        model_config: ModelConfig,
        mlflow_config: MLflowConfig,
        ab_test_config: ABTestConfig,
    ) -> None:
        self._gmm_config = gmm_config
        self._model_config = model_config
        self._mlflow_config = mlflow_config
        self._seeds = ab_test_config.seeds

    def train(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_val: np.ndarray,
        y_val: np.ndarray,
        Z_test: np.ndarray,
        y_test: np.ndarray,
    ) -> GMMTrainResult:
        """Train GMM across N seeds; return best by eval_macro_f1 (val)."""
        mlflow.autolog(disable=True)
        mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)
        mlflow.set_experiment(self._mlflow_config.experiment_name_gmm)

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
            )
            if result.eval_macro_f1 > best_f1:
                best_f1 = result.eval_macro_f1
                best_result = result

        canonical_path = "models/best_gmm_model.pkl"
        shutil.copy2(best_result.model_path, canonical_path)
        best_result.model_path = canonical_path

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
    ) -> GMMTrainResult:
        np.random.seed(seed)

        priors = np.array([
            (y_train == c).sum() / len(y_train)
            for c in range(self._model_config.n_classes)
        ])
        log_priors = np.log(priors)

        gmms = {}
        for class_label in range(self._model_config.n_classes):
            gmm = GaussianMixture(
                n_components=self._gmm_config.n_components[class_label],
                covariance_type=self._gmm_config.covariance_type,
                reg_covar=self._gmm_config.reg_covar,
                max_iter=self._gmm_config.max_iter,
                n_init=self._gmm_config.n_init,
                random_state=seed,
            )
            gmm.fit(Z_train[y_train == class_label])
            gmms[class_label] = gmm

        classifier = GMMClassifier(
            gmms=gmms,
            log_priors=log_priors,
            fatal_prior_boost=self._gmm_config.fatal_prior_boost,
        )

        y_train_pred = classifier.predict(Z_train)
        y_val_pred = classifier.predict(Z_val)
        y_test_pred = classifier.predict(Z_test)

        ein_macro_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
        eval_macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        eout_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

        val_fatal_mask = y_val == 2
        eval_fatal_recall = (
            float((y_val_pred[val_fatal_mask] == 2).sum() / val_fatal_mask.sum())
            if val_fatal_mask.sum() > 0 else 0.0
        )
        test_fatal_mask = y_test == 2
        eout_fatal_recall = (
            float((y_test_pred[test_fatal_mask] == 2).sum() / test_fatal_mask.sum())
            if test_fatal_mask.sum() > 0 else 0.0
        )

        with mlflow.start_run(run_name=f"gmm_seed_{seed}") as run:
            mlflow.log_params({
                "seed": seed,
                "n_components_pdo": self._gmm_config.n_components[0],
                "n_components_injury": self._gmm_config.n_components[1],
                "n_components_fatal": self._gmm_config.n_components[2],
                "covariance_type": self._gmm_config.covariance_type,
                "reg_covar": self._gmm_config.reg_covar,
                "max_iter": self._gmm_config.max_iter,
                "n_init": self._gmm_config.n_init,
                "fatal_prior_boost": self._gmm_config.fatal_prior_boost,
            })
            mlflow.log_metrics({
                "ein_macro_f1": ein_macro_f1,
                "eval_macro_f1": eval_macro_f1,
                "eval_fatal_recall": eval_fatal_recall,
                "eout_macro_f1": eout_macro_f1,
                "eout_fatal_recall": eout_fatal_recall,
                "generalisation_gap": ein_macro_f1 - eout_macro_f1,
            })

            matrix_path = Path("per_class_matrix.json")
            matrix_path.write_text(
                json.dumps(per_class_matrix(y_test, y_test_pred, _CLASS_NAMES), indent=2)
            )
            mlflow.log_artifact(str(matrix_path))
            matrix_path.unlink()

            log_confusion_matrix(y_test, y_test_pred, _CLASS_NAMES)
            log_roc_curve(y_test, self._soft_posteriors(classifier, Z_test), _CLASS_NAMES)

            model_path = f"models/best_gmm_model_seed{seed}.pkl"
            Path("models").mkdir(exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(classifier, f)

            run_id = run.info.run_id

        return GMMTrainResult(
            run_id=run_id,
            model_path=model_path,
            best_seed=seed,
            eout_macro_f1=eout_macro_f1,
            eout_fatal_recall=eout_fatal_recall,
            eval_macro_f1=eval_macro_f1,
        )

    def _soft_posteriors(self, classifier: GMMClassifier, Z: np.ndarray) -> np.ndarray:
        """Softmax-normalize log posteriors to probabilities for ROC curve."""
        log_p = classifier.predict_log_posteriors(Z)
        log_p -= log_p.max(axis=1, keepdims=True)
        exp_p = np.exp(log_p)
        return exp_p / exp_p.sum(axis=1, keepdims=True)
