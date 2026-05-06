from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class FeaturesConfig:
    columns: list[str]
    numeric_columns: list[str]
    danger_index_features: bool = False
    forbidden_columns: list[str] = field(default_factory=list)
    target_column: str = "CRASHSEVER"
    sentinel_columns: list[str] = field(default_factory=list)
    ordinal_columns: dict[str, list[str]] = field(default_factory=dict)
    cyclical_columns: dict[str, int] = field(default_factory=dict)


@dataclass
class DataConfig:
    raw_path: str
    processed_dir: str
    train_size: float
    val_size: float
    test_size: float
    random_state: int
    sentinel_value: int


@dataclass
class ModelConfig:
    n_classes: int
    n_select: int
    macro_f1_threshold: float
    fatal_recall_threshold: float
    focal_loss_enabled: bool = False
    focal_loss_gamma: float = 2.0
    fatal_threshold: float = 0.5


@dataclass
class DLConfig:
    input_dim: int
    hidden_dim: int
    dropout_p: float
    epochs: int
    patience: int
    batch_size: int
    lr: float
    experiment_name: str
    focal_loss_enabled: bool
    focal_loss_gamma: float


@dataclass
class VAEConfig:
    encoder_dims: list[int]
    latent_dim: int
    beta_start: float
    beta_max: float
    warmup_epochs: int
    dropout_p: float
    epochs: int
    patience: int
    batch_size: int
    lr: float
    experiment_name: str
    cyclical_annealing: bool = False
    cycle_epochs: int = 50
    
    @property
    def beta(self) -> float:
        """Backward compatibility: return beta_max for old code."""
        return self.beta_max


@dataclass
class EncodeConfig:
    random_state: int = 42


@dataclass
class AugmentConfig:
    tvae_epochs: int
    target_fatal_ratio: float
    random_state: int


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name_ml: str
    experiment_name_dl: str
    experiment_name_vae: str
    experiment_name_tune: str
    model_name: str


@dataclass
class ABTestConfig:
    seeds: list[int]
    alpha: float
    tiebreak: str


@dataclass
class OptunaSearchSpace:
    beta_max_low: float
    beta_max_high: float
    latent_dim_choices: list[int]
    warmup_epochs_low: int
    warmup_epochs_high: int
    lr_low: float
    lr_high: float
    dropout_p_low: float
    dropout_p_high: float


@dataclass
class OptunaPrunerConfig:
    n_startup_trials: int = 5
    n_warmup_steps: int = 15


@dataclass
class OptunaConfig:
    n_trials: int
    study_name: str
    direction: str
    pruner: OptunaPrunerConfig
    search_space: OptunaSearchSpace


@dataclass
class TuneConfig:
    experiment_name: str
    max_trials: int
    namespace: str
    max_dl_trial_epochs: int = 50
    optuna: OptunaConfig | None = None


@dataclass
class FeatureSelectionConfig:
    method: str = "none"  # one of: none, mutual_info, rfe, correlation, vif
    n_features: int = 10  # used by supervised methods
    threshold: float = 0.95  # used by unsupervised methods


@dataclass
class ColumnContract:
    dtype: str
    mostly: float = 1.0
    min: float | None = None
    max: float | None = None
    allowed_values: list[Any] | None = None
    sentinel_values: list[Any] | None = None


@dataclass
class ValidationConfig:
    columns: dict[str, ColumnContract] = field(default_factory=dict)


@dataclass
class GreatExpectationsConfig:
    suite_name: str
    datasource_name: str


@dataclass
class ProjectConfig:
    features: FeaturesConfig
    data: DataConfig
    model: ModelConfig
    dl: DLConfig
    vae: VAEConfig
    encode: EncodeConfig
    augment: AugmentConfig
    mlflow: MLflowConfig
    ab_test: ABTestConfig
    tune: TuneConfig
    feature_selection: FeatureSelectionConfig
    great_expectations: GreatExpectationsConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def _load_tune_config(raw_tune: dict[str, Any]) -> TuneConfig:
    optuna_raw = raw_tune.pop("optuna", None)
    optuna: OptunaConfig | None = None
    if optuna_raw is not None:
        pruner_raw = optuna_raw.pop("pruner", {})
        search_raw = optuna_raw.pop("search_space", {})
        optuna = OptunaConfig(
            pruner=OptunaPrunerConfig(**pruner_raw),
            search_space=OptunaSearchSpace(**search_raw),
            **optuna_raw,
        )
    return TuneConfig(**raw_tune, optuna=optuna)


def load_config(path: str | None = None) -> ProjectConfig:
    path = path or os.getenv("PARAMS_PATH", "params.yaml")
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    required = {"features", "data", "model", "dl", "vae", "augment", "mlflow", "ab_test", "tune", "great_expectations"}
    missing = required - raw.keys()
    if missing:
        raise KeyError(f"params.yaml missing required sections: {missing}")

    return ProjectConfig(
        features=FeaturesConfig(**raw["features"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        dl=DLConfig(**raw["dl"]),
        vae=VAEConfig(**raw["vae"]),
        encode=EncodeConfig(**raw.get("encode", {"random_state": 42})),
        augment=AugmentConfig(**raw["augment"]),
        mlflow=MLflowConfig(**raw["mlflow"]),
        feature_selection=FeatureSelectionConfig(
            **raw.get("feature_selection", {})
        ),
        ab_test=ABTestConfig(**raw["ab_test"]),
        tune=_load_tune_config(raw["tune"]),
        great_expectations=GreatExpectationsConfig(**raw["great_expectations"]),
        validation=ValidationConfig(
            columns={
                col: ColumnContract(**spec)
                for col, spec in raw.get("validation", {}).get("columns", {}).items()
            }
        ),
    )
