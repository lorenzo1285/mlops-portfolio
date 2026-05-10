"""Isolated Optuna HPO experiment for GMM classifier.

Searches GMM hyperparameters on the pre-computed Z-space artifacts.
Fitness: 0.6 * eval_macro_f1 + 0.4 * eval_fatal_recall (same as OptunaTuner).
Results logged to MLflow experiment: crash-severity-gmm-optuna.

Run:
    uv run python experiments/gmm_optuna_experiment.py
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

from src.config import load_config
from src.train_gmm.trainer import GMMClassifier

_EXPERIMENT_NAME = "crash-severity-gmm-optuna"
_N_TRIALS = 100
_CLASS_NAMES = ["PDO", "Injury", "Fatal"]
_N_CLASSES = 3


def _load_data(processed_dir: Path) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    Z_train = np.load(processed_dir / "Z_train_augmented.npy")
    y_train = np.load(processed_dir / "y_train_augmented.npy")
    Z_val = np.load(processed_dir / "Z_val.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    Z_test = np.load(processed_dir / "Z_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    return Z_train, y_train, Z_val, y_val, Z_test, y_test


def _objective(
    trial: optuna.Trial,
    Z_train: np.ndarray,
    y_train: np.ndarray,
    Z_val: np.ndarray,
    y_val: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    tracking_uri: str,
) -> float:
    n_comp_pdo = trial.suggest_int("n_components_pdo", 1, 8)
    n_comp_injury = trial.suggest_int("n_components_injury", 1, 10)
    n_comp_fatal = trial.suggest_int("n_components_fatal", 1, 5)
    covariance_type = trial.suggest_categorical(
        "covariance_type", ["full", "tied", "diag", "spherical"]
    )
    reg_covar = trial.suggest_float("reg_covar", 1e-6, 1e-1, log=True)
    fatal_prior_boost = trial.suggest_float("fatal_prior_boost", 0.5, 20.0)
    n_init = trial.suggest_int("n_init", 3, 15)

    priors = np.array(
        [(y_train == c).sum() / len(y_train) for c in range(_N_CLASSES)]
    )
    log_priors = np.log(np.clip(priors, 1e-9, None))

    n_comps = [n_comp_pdo, n_comp_injury, n_comp_fatal]

    try:
        gmms: dict[int, GaussianMixture] = {}
        for class_label in range(_N_CLASSES):
            X_class = Z_train[y_train == class_label]
            # Guard: can't have more components than samples
            n_comp = min(n_comps[class_label], len(X_class))
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                max_iter=200,
                n_init=n_init,
                random_state=42,
            )
            gmm.fit(X_class)
            gmms[class_label] = gmm

        classifier = GMMClassifier(
            gmms=gmms,
            log_priors=log_priors,
            fatal_prior_boost=fatal_prior_boost,
        )

        y_val_pred = classifier.predict(Z_val)
        eval_macro_f1 = float(
            f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        )
        val_fatal_mask = y_val == 2
        eval_fatal_recall = (
            float((y_val_pred[val_fatal_mask] == 2).sum() / val_fatal_mask.sum())
            if val_fatal_mask.sum() > 0
            else 0.0
        )
        fitness = 0.6 * eval_macro_f1 + 0.4 * eval_fatal_recall

        y_test_pred = classifier.predict(Z_test)
        eout_macro_f1 = float(
            f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        )
        test_fatal_mask = y_test == 2
        eout_fatal_recall = (
            float((y_test_pred[test_fatal_mask] == 2).sum() / test_fatal_mask.sum())
            if test_fatal_mask.sum() > 0
            else 0.0
        )
        ein_macro_f1 = float(
            f1_score(
                y_train,
                classifier.predict(Z_train),
                average="macro",
                zero_division=0,
            )
        )

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"gmm_trial_{trial.number}"):
            mlflow.log_params({
                "n_components_pdo": n_comp_pdo,
                "n_components_injury": n_comp_injury,
                "n_components_fatal": n_comp_fatal,
                "covariance_type": covariance_type,
                "reg_covar": reg_covar,
                "fatal_prior_boost": fatal_prior_boost,
                "n_init": n_init,
            })
            mlflow.log_metrics({
                "fitness": fitness,
                "eval_macro_f1": eval_macro_f1,
                "eval_fatal_recall": eval_fatal_recall,
                "eout_macro_f1": eout_macro_f1,
                "eout_fatal_recall": eout_fatal_recall,
                "ein_macro_f1": ein_macro_f1,
                "generalisation_gap": ein_macro_f1 - eout_macro_f1,
            })

        return fitness

    except Exception as exc:
        print(f"  Trial {trial.number} failed: {exc}")
        return 0.0


def main() -> None:
    config = load_config()
    processed_dir = Path("data/processed")

    Z_train, y_train, Z_val, y_val, Z_test, y_test = _load_data(processed_dir)
    print(f"Z_train: {Z_train.shape}  Z_val: {Z_val.shape}  Z_test: {Z_test.shape}")
    print(f"Class distribution (train): { {c: int((y_train==c).sum()) for c in range(_N_CLASSES)} }")
    print(f"Running {_N_TRIALS} Optuna trials -> MLflow: {_EXPERIMENT_NAME}\n")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="gmm-optuna",
        direction="maximize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: _objective(
            trial,
            Z_train, y_train,
            Z_val, y_val,
            Z_test, y_test,
            config.mlflow.tracking_uri,
        ),
        n_trials=_N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial: #{best.number}")
    print(f"  Fitness:           {best.value:.4f}")
    print(f"  eval_macro_f1:     {best.user_attrs.get('eval_macro_f1', 'see MLflow')}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    print(f"\nTop 5 trials by fitness:")
    top5 = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:5]
    for t in top5:
        print(f"  #{t.number:3d}  fitness={t.value:.4f}  params={t.params}")


if __name__ == "__main__":
    main()
