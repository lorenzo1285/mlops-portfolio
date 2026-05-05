"""Optuna-based hyperparameter optimization for VAE pipeline."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.config import MLflowConfig, ModelConfig, TuneConfig, VAEConfig
from src.encode.encoder import LatentEncoder
from src.train_ml.trainer import MLTrainer
from src.train_vae.vae_trainer import DVAETrainer
from src.tune.tuner import TuneResult


class OptunaTuner:
    """Optuna-based VAE hyperparameter optimization with early pruning.

    Optimizes 5 VAE hyperparameters (beta_max, latent_dim, warmup_epochs, lr, dropout_p)
    using TPE sampler and median pruner. Each trial trains VAE → encodes data → trains
    classifier → evaluates on validation set.

    Public interface
    ----------------
    tune() → TuneResult
        Runs Optuna study for n_trials; returns best hyperparameters.
    """

    def __init__(
        self,
        tune_config: TuneConfig,
        vae_config: VAEConfig,
        mlflow_config: MLflowConfig,
        data_dir: Path | str,
        model_config: ModelConfig | None = None,
    ) -> None:
        """Initialize tuner and load featurized data.

        Args:
            tune_config: Tune configuration with Optuna settings
            vae_config: Base VAE configuration (will be cloned with suggested params)
            mlflow_config: MLflow tracking configuration
            data_dir: Directory containing processed data files
            model_config: Model configuration for classifier training (uses defaults if None)
        """
        self._tune_config = tune_config
        self._vae_config = vae_config
        self._mlflow_config = mlflow_config
        self._data_dir = Path(data_dir)
        
        # Use provided model_config or create minimal defaults
        if model_config is None:
            from src.config import ModelConfig
            model_config = ModelConfig(
                n_classes=3,
                n_select=3,
                macro_f1_threshold=0.35,
                fatal_recall_threshold=0.5,
                fatal_threshold=0.5,
            )
        self._model_config = model_config

        # Load featurized data at construction
        processed_dir = self._data_dir / "processed"
        # X_train (pre-augmentation) for VAE unsupervised training — matches pipeline design
        self._X_train = np.load(processed_dir / "X_train.npy")
        self._X_train_aug = np.load(processed_dir / "X_train_augmented.npy")
        self._y_train_aug = np.load(processed_dir / "y_train_augmented.npy")
        self._X_val = np.load(processed_dir / "X_val.npy")
        self._y_val = np.load(processed_dir / "y_val.npy")
        self._X_test = np.load(processed_dir / "X_test.npy")
        self._y_test = np.load(processed_dir / "y_test.npy")

        # Storage for best run tracking
        self._best_run_id: str | None = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for a single Optuna trial.

        Args:
            trial: Optuna trial object for hyperparameter suggestions

        Returns:
            Validation macro F1 score (fitness metric to maximize)
        """
        # Get search space from config
        search_space = self._tune_config.optuna.search_space

        # Suggest hyperparameters
        beta_max = trial.suggest_float(
            "beta_max",
            search_space.beta_max_low,
            search_space.beta_max_high,
            log=True,
        )
        latent_dim = trial.suggest_categorical(
            "latent_dim",
            search_space.latent_dim_choices,
        )
        warmup_epochs = trial.suggest_int(
            "warmup_epochs",
            search_space.warmup_epochs_low,
            search_space.warmup_epochs_high,
        )
        lr = trial.suggest_float(
            "lr",
            search_space.lr_low,
            search_space.lr_high,
            log=True,
        )
        dropout_p = trial.suggest_float(
            "dropout_p",
            search_space.dropout_p_low,
            search_space.dropout_p_high,
        )

        # Clone VAE config with suggested hyperparameters
        trial_vae_config = replace(
            self._vae_config,
            beta_max=beta_max,
            latent_dim=latent_dim,
            warmup_epochs=warmup_epochs,
            lr=lr,
            dropout_p=dropout_p,
        )

        # Combine original (pre-augmentation) train/val/test for unsupervised VAE training
        y_train_orig = np.load(self._data_dir / "processed" / "y_train.npy")
        X_all = np.vstack([self._X_train, self._X_val, self._X_test])
        y_all = np.hstack([y_train_orig, self._y_val, self._y_test])

        # MLflow nested run for this trial
        mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)
        mlflow.set_experiment(self._mlflow_config.experiment_name_tune)

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}") as run:
            # Tag as Optuna trial
            mlflow.set_tag("trial_type", "optuna")
            mlflow.set_tag("trial_number", trial.number)

            # Log trial hyperparameters
            mlflow.log_params({
                "beta_max": beta_max,
                "latent_dim": latent_dim,
                "warmup_epochs": warmup_epochs,
                "lr": lr,
                "dropout_p": dropout_p,
            })

            try:
                # Train VAE with Optuna pruning support
                vae_trainer = DVAETrainer(
                    vae_config=trial_vae_config,
                    mlflow_config=self._mlflow_config,
                    run_name=f"vae_trial_{trial.number}",
                )
                vae_result = vae_trainer.train(
                    X_all=X_all,
                    y_all=y_all,
                    optuna_trial=trial,  # Enable pruning
                )

                # Encode data to latent space
                encoder = LatentEncoder(
                    encoder_path=vae_result.encoder_path,
                    latent_dim=latent_dim,
                )
                encode_result = encoder.encode(
                    X_train_augmented=self._X_train_aug,
                    y_train_augmented=self._y_train_aug,
                    X_val=self._X_val,
                    X_test=self._X_test,
                )

                # Train classifier on latent space (single seed for speed)
                ml_trainer = MLTrainer(
                    mlflow_config=self._mlflow_config,
                    model_config=self._model_config,
                    seeds=[0],  # Single seed for fast trials
                )
                ml_result = ml_trainer.train(
                    Z_train=encode_result.Z_train_augmented,
                    y_train=self._y_train_aug,
                    Z_val=encode_result.Z_val,
                    y_val=self._y_val,
                    Z_test=encode_result.Z_test,
                    y_test=self._y_test,
                )

                # Validation fitness metric (maximize): penalise 50% if fatal recall gate missed
                recall_penalty = 1.0 if ml_result.eval_fatal_recall >= 0.50 else 0.5
                val_fitness = ml_result.eval_macro_f1 * recall_penalty

                # Log fitness
                mlflow.log_metric("val_fitness", val_fitness)
                mlflow.log_metric("val_macro_f1", ml_result.eval_macro_f1)
                mlflow.log_metric("val_fatal_recall", ml_result.eval_fatal_recall)
                mlflow.log_metric("test_macro_f1", ml_result.eout_macro_f1)

                # Track best run
                if self._best_run_id is None or val_fitness > self._best_val_fitness:
                    self._best_run_id = run.info.run_id
                    self._best_val_fitness = val_fitness

                return val_fitness

            except optuna.TrialPruned:
                # Trial was pruned by median pruner - log and re-raise
                mlflow.log_metric("trial_pruned", 1.0)
                mlflow.set_tag("trial_status", "pruned")
                raise

    def tune(self) -> TuneResult:
        """Run Optuna hyperparameter optimization study.

        Returns:
            TuneResult with best hyperparameters and metrics
        """
        # End any lingering MLflow runs
        mlflow.end_run()
        
        # Initialize tracking for best trial
        self._best_run_id = None
        self._best_val_fitness = -float("inf")

        # Create Optuna study
        pruner = MedianPruner(
            n_startup_trials=self._tune_config.optuna.pruner.n_startup_trials,
            n_warmup_steps=self._tune_config.optuna.pruner.n_warmup_steps,
        )
        sampler = TPESampler(seed=42)

        study = optuna.create_study(
            study_name=self._tune_config.optuna.study_name,
            direction=self._tune_config.optuna.direction,
            sampler=sampler,
            pruner=pruner,
        )

        # Optimize
        study.optimize(
            self._objective,
            n_trials=self._tune_config.optuna.n_trials,
            show_progress_bar=True,
        )

        # Extract best trial
        best_trial = study.best_trial

        # Return TuneResult
        return TuneResult(
            best_params={
                "beta_max": best_trial.params["beta_max"],
                "latent_dim": best_trial.params["latent_dim"],
                "warmup_epochs": best_trial.params["warmup_epochs"],
                "lr": best_trial.params["lr"],
                "dropout_p": best_trial.params["dropout_p"],
            },
            best_value=best_trial.value,
            n_trials=len(study.trials),
            best_run_id=self._best_run_id or "unknown",
        )
