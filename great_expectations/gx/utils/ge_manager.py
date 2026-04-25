"""
ge_manager.py — Reusable Great Expectations engine for pandas DataFrames.

This module is the EXECUTION layer. It reads the configuration defined
by GEContextBuilder (datasources, assets, expectation suites) and runs
validations against pandas DataFrames. It never creates or modifies expectations.

Layer architecture:
    ┌─────────────────────────────────────────────────┐
    │  ge_context_builder.py      (BUSINESS LAYER)    │
    │  → defines datasources, assets, suites           │
    │  → run once before the validate stage            │
    ├─────────────────────────────────────────────────┤
    │  ge_manager.py               (ENGINE LAYER)     │
    │  → reads existing config, runs validations       │
    │  → reusable across any project / pipeline        │
    ├─────────────────────────────────────────────────┤
    │  src/validate/run.py         (EXECUTION LAYER)  │
    │  → loads data, calls the engine, gets results    │
    └─────────────────────────────────────────────────┘

Flow:
    1. ge = GEManager(context_root_dir)
    2. ge.select_asset_and_suite("crash_data_asset", "crash_data_suite")
    3. ge.inspect_suite()
    4. ge.set_batch_definition()
    5. ge.pre_validate(df)
    6. report = ge.run_validation(df)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import great_expectations as gx
import pandas as pd
from great_expectations.data_context import FileDataContext


class GEManager:
    """
    Read-only engine that validates pandas DataFrames against business-defined rules.

    This class NEVER creates or modifies expectations / suites.
    All rules are defined upstream in the context by GEContextBuilder.
    """

    BATCH_METHODS = ("whole_dataframe",)
    RESULT_FORMATS = ("BOOLEAN_ONLY", "BASIC", "SUMMARY", "COMPLETE")

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

        self.context: FileDataContext = gx.get_context(context_root_dir=str(root))
        self._log(f"✅ Context loaded from: {root}")

        # Print what GEContextBuilder has configured
        for ds in self.context.data_sources.all():
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
    # 2. SELECT ASSET + SUITE
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
        available: dict[str, list[str]] = {
            ds.name: [asset.name for asset in ds.assets]
            for ds in self.context.data_sources.all()
        }

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
        self._log(f"✅ Datasource: {datasource_name}")

        self.data_asset = self.datasource.get_asset(asset_name)
        self._log(f"✅ Asset: {asset_name}")

        self.suite = self.context.suites.get(suite_name)
        self._log(f"✅ Suite: {suite_name} ({len(self.suite.expectations)} expectations)")

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
            column = getattr(exp, "column", "—")
            meta = getattr(exp, "meta", None) or {}
            condition = getattr(exp, "row_condition", "—")

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
                "Condition": condition if condition else "—",
                "Parameters": extra_params,
            })

        report = pd.DataFrame(rows)

        self._log(f"✅ Suite '{self.suite.name}' contains {len(rows)} expectations:")
        if self.verbose:
            for _, row in report.iterrows():
                rule_label = f"[{row['Rule']}] " if row["Rule"] else ""
                self._log(
                    f"   {row['#']}. {rule_label}{row['Expectation']} "
                    f"on [{row['Column']}]"
                )
                if row["Condition"] != "—":
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
                self._log(f"✅ Batch Definition (existing): {name}")
                return

        if method == "whole_dataframe":
            self.batch_definition = (
                self.data_asset.add_batch_definition_whole_dataframe(name=name)
            )

        self._log(f"➕ Batch Definition (created): {name}")

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
            self._log("⚠️  DataFrame is EMPTY — validation would pass on 0 rows.")
            is_safe = False

        expected_columns = {
            getattr(exp, "column", None)
            for exp in self.suite.expectations
            if getattr(exp, "column", None)
        }
        actual_columns = set(df.columns.tolist())
        missing = expected_columns - actual_columns

        if missing:
            self._log(f"❌ Missing columns in DataFrame: {missing}")
            self._log(f"   Suite expects: {sorted(expected_columns)}")
            self._log(f"   DataFrame has: {sorted(actual_columns)}")
            is_safe = False
        else:
            self._log(f"✅ All {len(expected_columns)} expected columns present.")

        extra = actual_columns - expected_columns
        if extra and self.verbose:
            self._log(f"   ℹ️  {len(extra)} extra columns not checked by suite: {sorted(extra)}")

        if is_safe:
            self._log("✅ Pre-validation passed — safe to run.")
        else:
            self._log("⚠️  Pre-validation has warnings — review before running.")

        return is_safe

    # ===================================================================
    # 6. RUN VALIDATION
    # ===================================================================
    def run_validation(
        self,
        df: pd.DataFrame,
        validation_name: Optional[str] = None,
        result_format: str = "COMPLETE",
        partial_unexpected_count: int = 5,
    ) -> pd.DataFrame:
        """
        Validate a pandas DataFrame against the selected suite.

        Parameters
        ----------
        df : pd.DataFrame
            The data to validate.
        validation_name : str, optional
            Custom name for traceable runs (e.g. "reprocess_jan_2026").
            None reuses the default persistent validation.
        result_format : str
            One of: BOOLEAN_ONLY, BASIC, SUMMARY, COMPLETE (default).
        partial_unexpected_count : int
            Number of bad value examples to include (default 5).

        Returns
        -------
        pd.DataFrame
            One row per expectation with: Expectation, Column, Rule,
            Success, Total Rows, Bad Rows, Success %, Examples.
        """
        self._require("batch_definition", "set_batch_definition()")
        self._require("suite", "select_asset_and_suite()")

        self._separator("6. RUNNING VALIDATION")

        if result_format not in self.RESULT_FORMATS:
            raise ValueError(
                f"Unknown result_format '{result_format}'. "
                f"Supported: {self.RESULT_FORMATS}"
            )

        if validation_name is None:
            validation_name = f"{self.suite.name}_validation"
            self._log(f"   Mode: persistent (reusing '{validation_name}')")
        else:
            self._log(f"   Mode: named run ('{validation_name}')")

        try:
            validation_def = self.context.validation_definitions.get(validation_name)
            self._log(f"✅ Validation Definition (existing): {validation_name}")
        except Exception:
            validation_def = gx.ValidationDefinition(
                name=validation_name,
                data=self.batch_definition,
                suite=self.suite,
            )
            self.context.validation_definitions.add(validation_def)
            self._log(f"➕ Validation Definition (created): {validation_name}")

        self._log(f"   Result format: {result_format}")
        self._log(f"   Partial unexpected count: {partial_unexpected_count}")
        self._log("   Running expectations...")

        result = validation_def.run(
            batch_parameters={"dataframe": df},
            result_format={
                "result_format": result_format,
                "partial_unexpected_count": partial_unexpected_count,
            },
        )

        report = self._parse_results(result)
        self._print_verdict(report)

        return report

    # ===================================================================
    # INTERNAL HELPERS
    # ===================================================================
    def _parse_results(self, result, silent: bool = False) -> pd.DataFrame:
        """Parse a GE validation result into a clean pandas DataFrame."""
        rows = []
        for r in result.results:
            stats = (r.result or {})
            config = r.expectation_config

            config_kwargs = config.kwargs if hasattr(config, "kwargs") else {}
            config_meta = config.meta if hasattr(config, "meta") else {}

            if not stats or len(stats) == 0:
                rows.append({
                    "Expectation": getattr(config, "type", "unknown_expectation"),
                    "Column": config_kwargs.get("column", "Table"),
                    "Rule": config_meta.get("rule", "") if isinstance(config_meta, dict) else "",
                    "Success": r.success,
                    "Total Rows": "N/A",
                    "Bad Rows": "N/A",
                    "Success %": "Pass" if r.success else "Fail",
                    "Examples": [],
                })
                continue

            total_rows = stats.get("element_count")
            if total_rows is None:
                total_rows = stats.get("nonnull_count")

            bad_rows = stats.get("unexpected_count")
            if bad_rows is None and bool(getattr(r, "success", False)):
                bad_rows = 0

            unexpected_pct = stats.get("unexpected_percent")
            if (
                total_rows is None
                and bad_rows is not None
                and unexpected_pct not in (None, 0)
            ):
                total_rows = int(round((bad_rows * 100.0) / float(unexpected_pct)))

            success_pct = "N/A"
            if total_rows not in (None, 0):
                bad = int(bad_rows or 0)
                ok_pct = max(0.0, 100.0 * (1.0 - (bad / float(total_rows))))
                success_pct = f"{ok_pct:.2f}%"
            elif unexpected_pct is not None:
                success_pct = f"{max(0.0, 100.0 - float(unexpected_pct)):.2f}%"

            rows.append({
                "Expectation": getattr(config, "type", "unknown_expectation"),
                "Column": config_kwargs.get("column", "Table"),
                "Rule": config_meta.get("rule", "") if isinstance(config_meta, dict) else "",
                "Success": r.success,
                "Total Rows": total_rows,
                "Bad Rows": bad_rows if bad_rows is not None else 0,
                "Success %": success_pct,
                "Examples": stats.get("partial_unexpected_list", []),
            })

        return pd.DataFrame(rows)

    def _print_verdict(self, report: pd.DataFrame) -> None:
        """Print pass/fail summary from a parsed report."""
        self._separator("RESULTS")

        if report.empty:
            self._log("⚠️  No results to report.")
            return

        total = len(report)
        ok = int(report["Success"].sum())

        if report["Success"].all():
            self._log(f"✅ PASSED — {ok}/{total} expectations met.")
        else:
            self._log(f"❌ FAILED — {ok}/{total} expectations met.")
            failed = report[~report["Success"]]
            for _, row in failed.iterrows():
                rule_label = f"[{row['Rule']}] " if row["Rule"] else ""
                self._log(
                    f"   • {rule_label}{row['Expectation']} on "
                    f"[{row['Column']}] — {row['Bad Rows']} bad rows "
                    f"— {row['Success %']} ok"
                )
                if row["Examples"]:
                    self._log(f"     Examples: {row['Examples']}")

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
