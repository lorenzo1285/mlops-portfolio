from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.featurize.selector import FeatureSelector


@dataclass
class FeaturizeResult:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    selector: FeatureSelector | None
    feature_cols: list[str]
    selected_cols: list[str]
    n_features_raw: int
    n_params: int
    samples_per_param_ratio: float


_CRASHSEVER_MAP: dict[str, int] = {
    "Property Damage Only": 0,
    "Injury": 1,
    "Fatal": 2,
}


class Featurizer:
    """Encode, 3-way-split (train/val/test), and scale a tabular dataset.

    All column names, split ratios, and sentinel handling come from the
    constructor — no hardcoded dataset assumptions inside the class.
    Reusable for any binary-classification tabular problem.

    Public interface
    ----------------
    fit_transform(df) → FeaturizeResult
        Fits the preprocessing pipeline on the train split only,
        transforms all three splits, and returns all arrays plus
        sample-complexity diagnostics.
    """

    FORBIDDEN_COLUMNS = frozenset(["NUMOFKILL", "NUMOFINJ", "NUMOFUNINJ"])

    def __init__(
        self,
        feature_cols: list[str],
        numeric_cols: list[str],
        target_col: str,
        train_size: float,
        val_size: float,
        test_size: float,
        random_state: int,
        sentinel_value: int | float | None = None,
        sentinel_cols: list[str] | None = None,
        ordinal_cols: dict[str, list[str]] | None = None,
        cyclical_cols: dict[str, int] | None = None,
        feature_selector: FeatureSelector | None = None,
        danger_index_features: bool = False,
        forbidden_columns: list[str] | None = None,
    ) -> None:
        self._forbidden_columns = list(forbidden_columns) if forbidden_columns else list(self.FORBIDDEN_COLUMNS)
        
        # Guard: forbidden columns must never be in feature_cols
        intersection = set(feature_cols) & set(self._forbidden_columns)
        if intersection:
            raise ValueError(f"Requested feature_cols contains forbidden leakage columns: {intersection}")

        self._feature_cols = feature_cols
        self._numeric_cols = numeric_cols
        self._ordinal_cols = dict(ordinal_cols) if ordinal_cols else {}
        self._cyclical_cols = dict(cyclical_cols) if cyclical_cols else {}
        self._categorical_cols = [
            c for c in feature_cols
            if c not in numeric_cols and c not in self._ordinal_cols
        ]
        self._target_col = target_col
        self._train_size = train_size
        self._val_size = val_size
        self._test_size = test_size
        self._random_state = random_state
        self._sentinel_value = sentinel_value
        self._sentinel_cols = sentinel_cols or []
        self._feature_selector = feature_selector
        self._danger_index_features = danger_index_features

    def fit_transform(self, df: pd.DataFrame) -> FeaturizeResult:
        df = self._select_and_recode(df)
        X, y = self._separate_target(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self._split(X, y)
        preprocessor, X_tr, X_vl, X_te = self._fit_preprocess(X_train, X_val, X_test)

        feature_names = list(preprocessor.get_feature_names_out())
        n_features_raw = len(feature_names)
        selected_cols = feature_names
        active_selector: FeatureSelector | None = None

        if self._feature_selector is not None:
            self._feature_selector.fit(X_tr, y_train.values, feature_names)
            X_tr, sel_result = self._feature_selector.transform(X_tr)
            X_vl, _ = self._feature_selector.transform(X_vl)
            X_te, _ = self._feature_selector.transform(X_te)
            selected_cols = sel_result.selected_cols
            active_selector = self._feature_selector

        n_params, ratio = self._sample_complexity(len(X_train), X_tr.shape[1])
        return FeaturizeResult(
            X_train=X_tr,
            X_val=X_vl,
            X_test=X_te,
            y_train=y_train.values,
            y_val=y_val.values,
            y_test=y_test.values,
            preprocessor=preprocessor,
            selector=active_selector,
            feature_cols=list(X_train.columns),
            selected_cols=selected_cols,
            n_features_raw=n_features_raw,
            n_params=n_params,
            samples_per_param_ratio=ratio,
        )

    def _select_and_recode(self, df: pd.DataFrame) -> pd.DataFrame:
        all_cols = self._feature_cols + [self._target_col]
        # Temporarily retain NUMOFVEHIC if needed for engineering
        if self._danger_index_features and "NUMOFVEHIC" not in all_cols:
            all_cols.append("NUMOFVEHIC")

        df = df[[c for c in all_cols if c in df.columns]].copy()

        if self._sentinel_value is not None:
            for col in self._sentinel_cols:
                if col in df.columns:
                    df[col] = df[col].replace(self._sentinel_value, np.nan)
        
        if self._danger_index_features:
            df = self._compute_danger_index(df)

        df = self._apply_cyclical(df)
        return df.dropna(subset=[self._target_col])

    def _compute_danger_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer safety-critical interaction features."""
        # solo_highspeed: 1 if single vehicle crash at >= 45mph
        if "NUMOFVEHIC" in df.columns and "SPEEDLIMIT" in df.columns:
            df["solo_highspeed"] = ((df["NUMOFVEHIC"] == 1) & (df["SPEEDLIMIT"] >= 45)).astype(int)
        
        # vulnerability_interaction: 1 if young/old driver at > 40mph
        if "DRIVER1AGE" in df.columns and "SPEEDLIMIT" in df.columns:
            df["vulnerability_interaction"] = (
                ((df["DRIVER1AGE"] < 25) | (df["DRIVER1AGE"] > 70)) & 
                (df["SPEEDLIMIT"] > 40)
            ).astype(int)
        
        # Drop NUMOFVEHIC - it is replaced by these interaction features (Fix C)
        if "NUMOFVEHIC" in df.columns:
            df = df.drop(columns=["NUMOFVEHIC"])
            
        return df

    def _apply_cyclical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sin/cos encoding to cyclical columns (HOUR, MONTH).
        
        For each column in cyclical_columns dict:
            - Create {col}_sin = sin(2π × value / period)
            - Create {col}_cos = cos(2π × value / period)
            - Drop original column
        """
        _MONTH_MAP = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }
        _DAYOFWEEK_MAP = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }

        for col, period in self._cyclical_cols.items():
            if col not in df.columns:
                continue

            if col == "MONTH":
                values = df[col].map(_MONTH_MAP) - 1
            elif col == "DAYOFWEEK":
                values = df[col].map(_DAYOFWEEK_MAP)
            else:
                values = pd.to_numeric(df[col], errors='coerce')
            
            # Compute sin and cos transformations
            df[f"{col}_sin"] = values.apply(lambda x: math.sin(2 * math.pi * x / period) if pd.notna(x) else np.nan)
            df[f"{col}_cos"] = values.apply(lambda x: math.cos(2 * math.pi * x / period) if pd.notna(x) else np.nan)
            
            # Drop original column
            df = df.drop(columns=[col])
        
        return df

    def _separate_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        # Build feature column list, excluding target and including cyclical replacements
        feature_cols = []
        for c in self._feature_cols:
            if c in df.columns:
                feature_cols.append(c)
            elif c in self._cyclical_cols:
                # If this was a cyclical column that was replaced, add the sin/cos pairs
                if f"{c}_sin" in df.columns:
                    feature_cols.append(f"{c}_sin")
                if f"{c}_cos" in df.columns:
                    feature_cols.append(f"{c}_cos")
        
        # Add engineered features if enabled and present
        if self._danger_index_features:
            for eng_col in ["solo_highspeed", "vulnerability_interaction"]:
                if eng_col in df.columns and eng_col not in feature_cols:
                    feature_cols.append(eng_col)
        
        X = df[feature_cols]
        y = df[self._target_col].map(_CRASHSEVER_MAP)
        return X, y

    def _split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.Series, pd.Series, pd.Series,
    ]:
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y,
            test_size=self._test_size,
            random_state=self._random_state,
            stratify=y,
        )
        relative_val = self._val_size / (1.0 - self._test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=relative_val,
            random_state=self._random_state,
            stratify=y_tv,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _fit_preprocess(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[ColumnTransformer, np.ndarray, np.ndarray, np.ndarray]:
        cols = list(X_train.columns)
        
        # Separate numeric columns from cyclical sin/cos columns
        num = [c for c in self._numeric_cols if c in cols]

        # Append engineered features to numeric group if present
        for eng_col in ["solo_highspeed", "vulnerability_interaction"]:
            if eng_col in cols and eng_col not in num:
                num.append(eng_col)
        
        # Collect cyclical sin/cos columns
        cyc = []
        for col in self._cyclical_cols:
            if f"{col}_sin" in cols:
                cyc.append(f"{col}_sin")
            if f"{col}_cos" in cols:
                cyc.append(f"{col}_cos")
        
        # Remove cyclical columns from ordinal list (MONTH was previously ordinal)
        ord_names = [c for c in self._ordinal_cols if c in cols and c not in self._cyclical_cols]
        cat = [c for c in self._categorical_cols if c in cols]

        transformers = [
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )),
            ]), cat),
        ]
        
        # Add cyclical transformer group - impute but do NOT scale (already bounded [-1, 1])
        if cyc:
            transformers.append((
                "cyc",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                ]),
                cyc,
            ))

        if ord_names:
            # Explicit category order preserves semantic meaning (Mon=0…Sun=6).
            # StandardScaler normalises the integers alongside numeric features.
            ord_categories = [self._ordinal_cols[c] for c in ord_names]
            transformers.append((
                "ord",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(
                        categories=ord_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        dtype=int,
                    )),
                ]),
                ord_names,
            ))

        preprocessor = ColumnTransformer(transformers)
        return (
            preprocessor,
            preprocessor.fit_transform(X_train),
            preprocessor.transform(X_val),
            preprocessor.transform(X_test),
        )

    @staticmethod
    def _sample_complexity(n_train: int, n_features: int) -> tuple[int, float]:
        n_params = (n_features * 128 + 128) + (128 * 64 + 64) + (64 * 1 + 1)
        return n_params, n_train / n_params
