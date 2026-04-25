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
    class_weight_neg: float
    class_weight_pos: float
    n_select: int
    macro_f1_threshold: float
    minority_recall_threshold: float


@dataclass
class DLConfig:
    epochs: int
    patience: int
    batch_size: int
    lr: float
    hidden_1: int
    hidden_2: int
    dropout: float
    weight_decay: float = 1e-4


@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name_ml: str
    experiment_name_dl: str
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


@dataclass
class ValidationConfig:
    columns: dict[str, ColumnContract] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    features: FeaturesConfig
    data: DataConfig
    model: ModelConfig
    dl: DLConfig
    mlflow: MLflowConfig
    ab_test: ABTestConfig
    feature_selection: FeatureSelectionConfig
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def load_config(path: str | None = None) -> ProjectConfig:
    path = path or os.getenv("PARAMS_PATH", "params.yaml")
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    required = {"features", "data", "model", "dl", "mlflow", "ab_test"}
    missing = required - raw.keys()
    if missing:
        raise KeyError(f"params.yaml missing required sections: {missing}")

    return ProjectConfig(
        features=FeaturesConfig(**raw["features"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        dl=DLConfig(**raw["dl"]),
        mlflow=MLflowConfig(**raw["mlflow"]),
        feature_selection=FeatureSelectionConfig(
            **raw.get("feature_selection", {})
        ),
        ab_test=ABTestConfig(**raw["ab_test"]),
        validation=ValidationConfig(
            columns={
                col: ColumnContract(**spec)
                for col, spec in raw.get("validation", {}).get("columns", {}).items()
            }
        ),
    )
