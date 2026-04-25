"""
ge_context_builder.py — Business Layer: builds the GE context from params.yaml validation config.

This module is the CONFIGURATION layer. It reads the project's params.yaml
validation contract (ProjectConfig.validation.columns) and creates or updates
the GE context with datasources, assets, suites, and column-level expectations.

Layer architecture:
    ┌─────────────────────────────────────────────────┐
    │  ge_context_builder.py      (BUSINESS LAYER)    │
    │  → reads params.yaml validation section          │
    │  → creates datasources, assets, suites           │
    │  → run once before the validate stage            │
    ├─────────────────────────────────────────────────┤
    │  ge_manager.py               (ENGINE LAYER)     │
    │  → reads existing config, runs validations       │
    │  → reusable across any project / pipeline        │
    ├─────────────────────────────────────────────────┤
    │  src/validate/run.py         (EXECUTION LAYER)  │
    │  → loads data, calls the engine, gets results    │
    └─────────────────────────────────────────────────┘

params.yaml validation.columns schema (one entry per column):
    ┌──────────────────┬────────────────────────────────────────────────┐
    │ dtype            │ Expected dtype: int, float, str                │
    │ mostly           │ Fraction of non-null values required (0.0–1.0) │
    │ min              │ Optional minimum value (numeric columns)       │
    │ max              │ Optional maximum value (numeric columns)       │
    │ allowed_values   │ Optional list of valid values (categorical)    │
    └──────────────────┴────────────────────────────────────────────────┘

Generated expectations per column:
    - ExpectColumnValuesToNotBeNull       (always; uses contract.mostly)
    - ExpectColumnValuesToBeBetween       (when min or max is set)
    - ExpectColumnValuesToBeInSet         (when allowed_values is set)

Flow:
    1. builder = GEContextBuilder(config.validation.columns, ...)
    2. builder.build()

    Or step by step:
    1. builder = GEContextBuilder(config.validation.columns, ...)
    2. builder.load_config()
    3. builder.inspect_config()
    4. builder.build()
"""

from __future__ import annotations

import json
from typing import Optional

import great_expectations as gx
import pandas as pd
from great_expectations.data_context import FileDataContext

from src.config import ColumnContract

# Tolerance for range/value-set checks — handles known sentinel values in raw data
# (e.g. HOUR=99, SPEEDLIMIT=0/99, DRIVER ages with 0s). Null tolerance per column
# is controlled by contract.mostly, not this constant.
_RANGE_MOSTLY = 0.99


class GEContextBuilder:
    """
    Business Layer: reads params.yaml validation config and builds the GE context.

    This class creates and updates datasources, assets, and expectation suites.
    It is the only class allowed to modify the GE context.
    """

    def __init__(
        self,
        validation_columns: dict[str, ColumnContract],
        datasource_name: str,
        asset_name: str,
        suite_name: str,
        context_root_dir: str,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the builder.

        :param validation_columns: Per-column contracts from config.validation.columns.
        :param datasource_name: Name to assign to the GE pandas datasource.
        :param asset_name: Name to assign to the GE dataframe asset.
        :param suite_name: Name of the expectation suite to create or update.
        :param context_root_dir: Root directory of the Great Expectations file context.
        :param verbose: If True, prints detailed logs.
        """
        self.validation_columns = validation_columns
        self._datasource_name = datasource_name
        self._asset_name = asset_name
        self._suite_name = suite_name
        self.verbose = verbose
        self.config_df: Optional[pd.DataFrame] = None

        self._separator("1. INITIALIZING CONTEXT BUILDER")
        self._log(f"   Datasource   : {datasource_name}")
        self._log(f"   Asset        : {asset_name}")
        self._log(f"   Suite        : {suite_name}")
        self._log(f"   Context root : {context_root_dir}")
        self._log(f"   Columns      : {len(validation_columns)}")

        try:
            self.context: FileDataContext = gx.get_context(context_root_dir=str(context_root_dir))
            self._log("✅ Context loaded (initialized if previously missing).")
        except Exception as e:
            self._log(f"❌ Failed to load or initialize GE context: {e}")
            raise

    def load_config(self) -> pd.DataFrame:
        """
        Build the expectations DataFrame from params.yaml ColumnContract objects.

        Each ColumnContract generates up to 3 expectations:
          - not-null (always)
          - range (if min or max is set)
          - value set (if allowed_values is set)
        """
        self._separator("2. LOADING CONFIG")

        rows = []
        for col_name, contract in self.validation_columns.items():
            # Not-null — uses contract.mostly for per-column null tolerance
            rows.append({
                "datasource": self._datasource_name,
                "asset_name": self._asset_name,
                "suite_name": self._suite_name,
                "column_name": col_name,
                "expectation": "ExpectColumnValuesToNotBeNull",
                "value_set": None,
                "row_condition": None,
                "rule": f"{col_name}_not_null",
                "kwargs": json.dumps({"mostly": contract.mostly}),
            })

            # Range check — _RANGE_MOSTLY handles known sentinel values in raw data
            if contract.min is not None or contract.max is not None:
                rows.append({
                    "datasource": self._datasource_name,
                    "asset_name": self._asset_name,
                    "suite_name": self._suite_name,
                    "column_name": col_name,
                    "expectation": "ExpectColumnValuesToBeBetween",
                    "value_set": None,
                    "row_condition": None,
                    "rule": f"{col_name}_range",
                    "kwargs": json.dumps({
                        "min_value": contract.min,
                        "max_value": contract.max,
                        "mostly": _RANGE_MOSTLY,
                    }),
                })

            # Allowed values
            if contract.allowed_values is not None:
                rows.append({
                    "datasource": self._datasource_name,
                    "asset_name": self._asset_name,
                    "suite_name": self._suite_name,
                    "column_name": col_name,
                    "expectation": "ExpectColumnValuesToBeInSet",
                    "value_set": "|".join(str(v) for v in contract.allowed_values),
                    "row_condition": None,
                    "rule": f"{col_name}_values",
                    "kwargs": None,
                })

        self.config_df = pd.DataFrame(rows)

        self._log(f"✅ Generated {len(rows)} expectations for {len(self.validation_columns)} columns.")

        unknown = [
            exp_type for exp_type in self.config_df["expectation"].unique()
            if not hasattr(gx.expectations, exp_type)
        ]
        if unknown:
            raise ValueError(
                f"Unknown expectation type(s): {unknown}\n"
                f"Ensure the expectation column contains exact GE class names."
            )

        return self.config_df

    def inspect_config(self) -> None:
        """Print summary of the config."""
        self._require_config()
        self._separator("3. CONFIG SUMMARY")

        self._log(f"   Suites      : {self.config_df['suite_name'].nunique()}")
        self._log(f"   Total rules : {len(self.config_df)}")

        for col_name, col_group in self.config_df.groupby("column_name"):
            self._log(f"\n   📋 {col_name}")
            for _, row in col_group.iterrows():
                value_info = f" → {row['value_set']}" if pd.notna(row.get("value_set")) else ""
                self._log(f"      • {row['expectation']}{value_info}")

    def build(self) -> None:
        """Build or update the GE context from the validation config."""
        if self.config_df is None:
            self.load_config()

        self.inspect_config()
        self._separator("4. BUILDING CONTEXT")

        suite_count = 0
        rule_count = 0

        for ds_name, ds_group in self.config_df.groupby("datasource"):
            self._log(f"\n   📦 Creating/updating datasource: {ds_name}")
            datasource = self.context.data_sources.add_or_update_pandas(name=ds_name)

            for asset_name, asset_group in ds_group.groupby("asset_name"):
                suite_name = asset_group["suite_name"].iloc[0]
                self._log(f"      🔹 Asset: {asset_name}")

                asset = datasource.add_dataframe_asset(name=asset_name)

                suite = gx.ExpectationSuite(
                    name=suite_name,
                    meta={"asset": asset_name, "datasource": ds_name},
                )

                expectations_built = 0
                for _, row in asset_group.iterrows():
                    try:
                        exp = self._build_expectation(row)
                        if exp is not None:
                            suite.add_expectation(exp)
                            rule_count += 1
                            expectations_built += 1
                    except Exception as e:
                        self._log(f"⚠️  Error building rule for {asset_name}: {e}")

                self.context.suites.add_or_update(suite)
                suite_count += 1
                self._log(f"         ✅ Suite '{suite_name}' saved ({expectations_built} expectations).")

        self._separator("BUILD COMPLETE")
        self._log(f"✅ Context ready: {suite_count} Suite(s), {rule_count} rule(s).")

    def _build_expectation(self, row: pd.Series):
        exp_type = row["expectation"]
        column = row["column_name"]
        value_set_raw = row.get("value_set")
        row_condition = row.get("row_condition")
        rule = row.get("rule", "")
        kwargs_raw = row.get("kwargs")

        exp_class = getattr(gx.expectations, exp_type, None)
        if exp_class is None:
            self._log(f"⚠️  Unknown expectation type: {exp_type}")
            return None

        # Introspect the class so we only pass kwargs it actually declares.
        # Works with both Pydantic v2 (model_fields) and v1 (__fields__).
        if hasattr(exp_class, "model_fields"):
            declared = set(exp_class.model_fields.keys())
        elif hasattr(exp_class, "__fields__"):
            declared = set(exp_class.__fields__.keys())
        else:
            declared = set()

        kwargs = {}
        if "column" in declared:
            kwargs["column"] = column

        meta = {"rule": rule} if rule else {}
        if meta:
            kwargs["meta"] = meta

        # Generic kwargs column (JSON string) — merged before dedicated columns
        if pd.notna(kwargs_raw) and str(kwargs_raw).strip() not in ("", "null"):
            try:
                extra = json.loads(str(kwargs_raw))
                if not isinstance(extra, dict):
                    raise ValueError("kwargs column must be a JSON object, e.g. {\"min_value\": 0}")
                kwargs.update(extra)
            except (json.JSONDecodeError, ValueError) as e:
                self._log(f"⚠️  Invalid JSON in 'kwargs' for {exp_type} [{column}]: {e}")
                return None

        if "value_set" in declared and pd.notna(value_set_raw) and str(value_set_raw).strip() != "":
            kwargs["value_set"] = [v.strip() for v in str(value_set_raw).split("|")]

        if "row_condition" in declared and pd.notna(row_condition):
            kwargs["row_condition"] = str(row_condition)

        return exp_class(**kwargs)

    def _require_config(self) -> None:
        if self.config_df is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _separator(self, title: str) -> None:
        if self.verbose:
            print(f"\n{'='*60}\n {title}\n{'='*60}")
