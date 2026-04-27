"""
ge_manager.py — Suite + Preparation Layer for Great Expectations.

This module is the SUITE + PREPARATION layer. It creates expectation suites  
from ColumnContract objects and prepares the context for validation execution.

Layer architecture:
    ┌─────────────────────────────────────────────────┐
    │  GEContextBuilder       (INFRASTRUCTURE LAYER)  │
    │  → initializes FileDataContext                   │
    │  → creates datasource + dataframe asset          │
    │  → NO suite creation, NO expectations            │
    ├─────────────────────────────────────────────────┤
    │  GEManager                (SUITE + PREP LAYER)  │
    │  → build_suite() generates ExpectationSuite      │
    │  → select_asset_and_suite() binds context       │
    │  → pre_validate() sanity-checks DataFrame        │
    ├─────────────────────────────────────────────────┤
    │  GECheckpointRunner       (EXECUTION LAYER)     │
    │  → run(df) executes validation via Checkpoint   │
    │  → returns CheckpointRunResult                   │
    └─────────────────────────────────────────────────┘

Flow:
    1. manager = GEManager(context_root_dir, verbose)
    2. manager.build_suite(suite_name, asset_name, datasource_name, validation_columns)
    3. manager.select_asset_and_suite(asset_name, suite_name, datasource_name)
    4. manager.set_batch_definition()
    5. manager.pre_validate(df)
    6. runner = GECheckpointRunner(context, batch_definition, verbose)
    7. result = runner.run(df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

# Import from installed great_expectations package (not local directory)
import great_expectations as gx_installed
import pandas as pd
from great_expectations.data_context import FileDataContext

from src.config import ColumnContract


class GEManager:
    """
    Suite + Preparation Layer: creates expectation suites and prepares context for validation.

    This class is responsible for:
    - build_suite(): Generates ExpectationSuite from ColumnContract objects
    - select_asset_and_suite(): Binds to existing asset and suite
    - set_batch_definition(): Prepares batch definition
    - pre_validate(): Sanity-checks DataFrame before validation
    """

    BATCH_METHODS = ("whole_dataframe",)

    def __init__(
        self,
        context_root_dir: str,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self._separator("1. LOADING CONTEXT")

        root = Path(context_root_dir)
        if not root.exists():
            raise FileNotFoundError(f"GE context root not found: {root}")
        if not (root / "great_expectations.yml").exists():
            raise FileNotFoundError(f"great_expectations.yml missing in: {root}")

        self.context: FileDataContext = gx_installed.get_context(context_root_dir=str(root))
        self._log(f"[OK] Context loaded from: {root}")

        # Print existing datasources and suites
        datasource_names = list(self.context.data_sources.all())
        self._log(f"   Datasources: {datasource_names}")
        for ds_name in datasource_names:
            ds = self.context.data_sources.get(ds_name)
            if ds:
                asset_names = [asset.name for asset in ds.assets]
                self._log(f"   [{ds.name}] assets: {asset_names}")

        suites = [s.name for s in self.context.suites.all()]
        self._log(f"   Suites: {suites}")

        # State — set by later steps
        self.datasource = None
        self.data_asset = None
        self.suite = None
        self.batch_definition = None

    # ===================================================================
    # 2. BUILD SUITE
    # ===================================================================
    def build_suite(
        self,
        suite_name: str,
        asset_name: str,
        datasource_name: str,
        validation_columns: dict[str, ColumnContract],
    ) -> None:
        """
        Generate an ExpectationSuite from ColumnContract objects.

        For each column contract, generates:
        - ExpectColumnValuesToNotBeNull (always; uses contract.mostly for null tolerance)
        - ExpectColumnValuesToBeBetween (when min/max set; uses row_condition per sentinel)
        - ExpectColumnValuesToBeInSet (when allowed_values set)

        Sentinel handling:
        - If contract.sentinel_values is non-empty, range checks add a row_condition
          per sentinel to exclude those rows explicitly (no global mostly tolerance).
        - If no sentinels, range check is strict (no mostly).

        Parameters
        ----------
        suite_name : str
            Name of the expectation suite to create or update.
        asset_name : str
            Name of the asset this suite will validate against.
        datasource_name : str
            Name of the datasource containing the asset.
        validation_columns : dict[str, ColumnContract]
            Per-column contracts from params.yaml validation.columns.
        """
        self._separator("2. BUILDING SUITE")
        self._log(f"   Suite: {suite_name}")
        self._log(f"   Asset: {asset_name}")
        self._log(f"   Datasource: {datasource_name}")
        self._log(f"   Columns: {len(validation_columns)}")

        suite = gx_installed.ExpectationSuite(
            name=suite_name,
            meta={"asset": asset_name, "datasource": datasource_name},
        )

        expectations_count = 0

        for col_name, contract in validation_columns.items():
            self._log(f"\n   [col] {col_name}")

            # Not-null — uses contract.mostly for per-column null tolerance
            suite.add_expectation(
                gx_installed.expectations.ExpectColumnValuesToNotBeNull(
                    column=col_name,
                    mostly=contract.mostly,
                    meta={"rule": f"{col_name}_not_null"},
                )
            )
            expectations_count += 1
            self._log(f"      -ExpectColumnValuesToNotBeNull (mostly={contract.mostly})")

            # Range check — strict; sentinels excluded via combined row_condition
            if contract.min is not None or contract.max is not None:
                if contract.sentinel_values:
                    # One expectation, one condition that excludes ALL sentinels simultaneously.
                    # GE v1 pandas engine uses DataFrame.query() syntax — no col() wrapper.
                    condition = " and ".join(
                        f"{col_name} != {repr(s)}" for s in contract.sentinel_values
                    )
                    suite.add_expectation(
                        gx_installed.expectations.ExpectColumnValuesToBeBetween(
                            column=col_name,
                            min_value=contract.min,
                            max_value=contract.max,
                            mostly=contract.mostly,
                            row_condition=condition,
                            condition_parser="pandas",
                            meta={"rule": f"{col_name}_range"},
                        )
                    )
                    expectations_count += 1
                    self._log(
                        f"      -ExpectColumnValuesToBeBetween [{contract.min}, {contract.max}] "
                        f"excluding sentinels {contract.sentinel_values} (mostly={contract.mostly})"
                    )
                else:
                    # No sentinels — use mostly tolerance
                    suite.add_expectation(
                        gx_installed.expectations.ExpectColumnValuesToBeBetween(
                            column=col_name,
                            min_value=contract.min,
                            max_value=contract.max,
                            mostly=contract.mostly,
                            meta={"rule": f"{col_name}_range"},
                        )
                    )
                    expectations_count += 1
                    self._log(f"      -ExpectColumnValuesToBeBetween [{contract.min}, {contract.max}] (mostly={contract.mostly})")

            # Allowed values — use mostly tolerance
            if contract.allowed_values is not None:
                suite.add_expectation(
                    gx_installed.expectations.ExpectColumnValuesToBeInSet(
                        column=col_name,
                        value_set=contract.allowed_values,
                        mostly=contract.mostly,
                        meta={"rule": f"{col_name}_values"},
                    )
                )
                expectations_count += 1
                self._log(f"      -ExpectColumnValuesToBeInSet ({len(contract.allowed_values)} values, mostly={contract.mostly})")

        self.context.suites.add_or_update(suite)
        self._log(f"\n[OK] Suite '{suite_name}' created with {expectations_count} expectations")

    # ===================================================================
    # 3. SELECT ASSET + SUITE
    # ===================================================================
    def select_asset_and_suite(
        self,
        asset_name: str,
        suite_name: str,
        datasource_name: Optional[str] = None,
    ) -> None:
        """
        Select an existing asset and suite from the business-defined context.
        Read-only — never creates datasources, assets, or suites.

        Parameters
        ----------
        asset_name : str
            Name of the data asset to select (must exist in the context).
        suite_name : str
            Name of the expectation suite to validate against (must exist in the context).
        datasource_name : str, optional
            Name of the datasource that contains the asset.
            If None, the datasource is auto-detected from the context.
            Must be provided explicitly when the same asset name exists
            in multiple datasources to avoid ambiguity.

        Raises
        ------
        KeyError
            If the asset or suite is not found in the context.
        ValueError
            If the asset name is found in more than one datasource and
            datasource_name was not specified.
        """
        self._separator("2. SELECTING ASSET + SUITE")

        # Build available map using GE v1 API
        datasource_names = list(self.context.data_sources.all())
        available: dict[str, list[str]] = {}
        for ds_name in datasource_names:
            ds = self.context.data_sources.get(ds_name)
            if ds:
                available[ds.name] = [asset.name for asset in ds.assets]

        # Auto-detect datasource — raise on ambiguity
        if datasource_name is None:
            matches = [ds for ds, assets in available.items() if asset_name in assets]

            if len(matches) > 1:
                raise ValueError(
                    f"Asset '{asset_name}' found in multiple datasources: {matches}.\n"
                    f"Pass datasource_name= explicitly to disambiguate."
                )
            elif len(matches) == 1:
                datasource_name = matches[0]

        if datasource_name is None:
            all_assets = [a for assets in available.values() for a in assets]
            raise KeyError(
                f"Asset '{asset_name}' not found in any datasource.\n"
                f"Available assets: {all_assets}"
            )

        self.datasource = self.context.data_sources.get(datasource_name)
        self._log(f"[OK] Datasource: {datasource_name}")

        self.data_asset = self.datasource.get_asset(asset_name)
        self._log(f"[OK] Asset: {asset_name}")

        self.suite = self.context.suites.get(suite_name)
        self._log(f"[OK] Suite: {suite_name} ({len(self.suite.expectations)} expectations)")

        existing = [bd.name for bd in self.data_asset.batch_definitions]
        if existing:
            self._log(f"   Existing batch definitions: {existing}")
        else:
            self._log("   No batch definitions yet — call set_batch_definition() next.")

    # ===================================================================
    # 3. INSPECT SUITE (read-only)
    # ===================================================================
    def inspect_suite(self) -> pd.DataFrame:
        """
        Read-only view of the business-defined expectations in the suite.
        Shows what rules exist, which columns they check, and any conditions.

        Returns a pandas DataFrame for easy viewing.
        """
        self._require("suite", "select_asset_and_suite()")
        self._separator("3. INSPECTING SUITE")

        rows = []
        for i, exp in enumerate(self.suite.expectations, 1):
            column = getattr(exp, "column", "N/A")
            meta = getattr(exp, "meta", None) or {}
            condition = getattr(exp, "row_condition", "N/A")

            config = exp.configuration if hasattr(exp, "configuration") else {}
            extra_params = {
                k: v for k, v in config.items()
                if k not in ("column", "row_condition", "condition_parser", "batch_id")
            }

            rows.append({
                "#": i,
                "Expectation": exp.__class__.__name__,
                "Column": column,
                "Rule": meta.get("rule", "") if isinstance(meta, dict) else "",
                "Condition": condition if condition else "N/A",
                "Parameters": extra_params,
            })

        report = pd.DataFrame(rows)

        self._log(f"[OK] Suite '{self.suite.name}' contains {len(rows)} expectations:")
        if self.verbose:
            for _, row in report.iterrows():
                rule_label = f"[{row['Rule']}] " if row["Rule"] else ""
                self._log(
                    f"   {row['#']}. {rule_label}{row['Expectation']} "
                    f"on [{row['Column']}]"
                )
                if row["Condition"] != "N/A":
                    self._log(f"      condition: {row['Condition']}")

        return report

    # ===================================================================
    # 4. SET BATCH DEFINITION
    # ===================================================================
    def set_batch_definition(
        self,
        name: Optional[str] = None,
        method: str = "whole_dataframe",
    ) -> None:
        """
        Get an existing batch definition or create a new one.
        """
        self._require("data_asset", "select_asset_and_suite()")
        self._separator("4. SETTING BATCH DEFINITION")

        if method not in self.BATCH_METHODS:
            raise ValueError(
                f"Unknown batch method '{method}'. Supported: {self.BATCH_METHODS}"
            )

        if name is None:
            name = f"{self.data_asset.name}_batch_def"

        self._log(f"   Method: {method}")

        for bd in self.data_asset.batch_definitions:
            if bd.name == name:
                self.batch_definition = bd
                self._log(f"[OK] Batch Definition (existing): {name}")
                return

        if method == "whole_dataframe":
            self.batch_definition = (
                self.data_asset.add_batch_definition_whole_dataframe(name=name)
            )

        self._log(f"[+] Batch Definition (created): {name}")

    # ===================================================================
    # 5. PRE-VALIDATE
    # ===================================================================
    def pre_validate(self, df: pd.DataFrame) -> bool:
        """
        Check that the DataFrame is safe to validate:
          - Not empty (0 rows would give a misleading "passed")
          - Has all columns the suite expects (missing columns cause errors, not failures)

        Returns True if safe, False if not. Prints warnings either way.
        """
        self._require("suite", "select_asset_and_suite()")
        self._separator("5. PRE-VALIDATION CHECKS")

        is_safe = True

        row_count = len(df)
        self._log(f"   Row count: {row_count}")

        if row_count == 0:
            self._log("[WARN]  DataFrame is EMPTY — validation would pass on 0 rows.")
            is_safe = False

        expected_columns = {
            getattr(exp, "column", None)
            for exp in self.suite.expectations
            if getattr(exp, "column", None)
        }
        actual_columns = set(df.columns.tolist())
        missing = expected_columns - actual_columns

        if missing:
            self._log(f"[ERROR] Missing columns in DataFrame: {missing}")
            self._log(f"   Suite expects: {sorted(expected_columns)}")
            self._log(f"   DataFrame has: {sorted(actual_columns)}")
            is_safe = False
        else:
            self._log(f"[OK] All {len(expected_columns)} expected columns present.")

        extra = actual_columns - expected_columns
        if extra and self.verbose:
            self._log(f"   [INFO] {len(extra)} extra columns not checked by suite: {sorted(extra)}")

        if is_safe:
            self._log("[OK] Pre-validation passed — safe to run.")
        else:
            self._log("[WARN]  Pre-validation has warnings — review before running.")

        return is_safe

    # ===================================================================
    # INTERNAL HELPERS
    # ===================================================================
    def _require(self, attr: str, step: str) -> None:
        if getattr(self, attr, None) is None:
            raise RuntimeError(f"'{attr}' not set. Run {step} first.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _separator(self, title: str) -> None:
        if self.verbose:
            print(f"\n{'='*60}")
            print(f" {title}")
            print(f"{'='*60}")
