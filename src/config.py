from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class FeaturesConfig:
    columns: list[str]
    numeric_columns: list[str]
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
    feature_selection: FeatureSelectionConfig
    great_expectations: GreatExpectationsConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def load_config(path: str | None = None) -> ProjectConfig:
    path = path or os.getenv("PARAMS_PATH", "params.yaml")
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    required = {"features", "data", "model", "dl", "vae", "augment", "mlflow", "ab_test", "great_expectations"}
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
        great_expectations=GreatExpectationsConfig(**raw["great_expectations"]),
        validation=ValidationConfig(
            columns={
                col: ColumnContract(**spec)
                for col, spec in raw.get("validation", {}).get("columns", {}).items()
            }
        ),
    )
