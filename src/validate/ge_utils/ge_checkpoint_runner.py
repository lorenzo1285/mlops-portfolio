"""
ge_checkpoint_runner.py — Execution Layer: runs validation via GE Checkpoint.

This module is the EXECUTION layer. It takes a prepared GE context and
batch definition, runs validation via a Checkpoint with actions, and returns results.

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
    1. builder = GEContextBuilder(...).build()
    2. manager = GEManager(...)
    3. manager.build_suite(...)
    4. manager.select_asset_and_suite() → set_batch_definition() → pre_validate(df)
    5. runner = GECheckpointRunner(context, batch_definition, suite, verbose)
    6. result = runner.run(df)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import great_expectations as gx_installed
import pandas as pd
from great_expectations.data_context import FileDataContext


@dataclass
class CheckpointRunResult:
    """Result of running a GE Checkpoint."""

    success: bool
    failed_expectations: list[str]
    data_docs_path: str | None


class GECheckpointRunner:
    """
    Execution Layer: runs validation via a GE Checkpoint with Data Docs generation.

    Builds or updates a Checkpoint with UpdateDataDocsAction + StoreValidationResultAction,
    runs it against a DataFrame, and returns CheckpointRunResult.
    """

    def __init__(
        self,
        context: FileDataContext,
        batch_definition: Any,
        suite: Any,
        checkpoint_name: str = "validation_checkpoint",
        verbose: bool = True,
    ) -> None:
        self.context = context
        self.batch_definition = batch_definition
        self.suite = suite
        self.checkpoint_name = checkpoint_name
        self.verbose = verbose

        self._log("=" * 60)
        self._log("GE CHECKPOINT RUNNER — EXECUTION LAYER")
        self._log("=" * 60)
        self._log(f"Checkpoint : {checkpoint_name}")
        self._log(f"Suite      : {suite.name}")

    def run(self, df: pd.DataFrame) -> CheckpointRunResult:
        """
        Run validation using GE v1 ValidationDefinition.

        In GE v1, ValidationDefinition.run() automatically:
        - Executes the expectations
        - Stores validation results
        - Generates Data Docs

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Returns
        -------
        CheckpointRunResult
            Contains success status, failed expectation names, and Data Docs path.
        """
        self._log(f"\nRUNNING VALIDATION — {len(df)} rows × {len(df.columns)} cols")

        # Create validation definition
        validation_name = f"{self.checkpoint_name}_validation"

        try:
            validation_def = self.context.validation_definitions.get(validation_name)
            self._log(f"[OK] Using existing ValidationDefinition: {validation_name}")
        except Exception:
            validation_def = gx_installed.ValidationDefinition(
                name=validation_name,
                data=self.batch_definition,
                suite=self.suite,
            )
            self.context.validation_definitions.add(validation_def)
            self._log(f"[+] Created ValidationDefinition: {validation_name}")

        # Run validation (automatically stores results and builds Data Docs)
        self._log("   Executing validation...")
        result = validation_def.run(
            batch_parameters={"dataframe": df},
        )

        # Build Data Docs explicitly
        try:
            self.context.build_data_docs()
            self._log("[OK] Data Docs built")
        except Exception as e:
            self._log(f"[WARN]  Data Docs build warning: {e}")

        # Parse results
        success = result.success
        failed_expectations = self._extract_failed_expectations(result)

        # Get Data Docs path
        data_docs_path = self._get_data_docs_path()

        self._log(f"\n{'[OK] VALIDATION PASSED' if success else '[ERROR] VALIDATION FAILED'}")
        if not success:
            self._log(f"Failed expectations ({len(failed_expectations)}):")
            for exp in failed_expectations:
                self._log(f"  - {exp}")
        if data_docs_path:
            self._log(f"Data Docs: {data_docs_path}")

        return CheckpointRunResult(
            success=success,
            failed_expectations=failed_expectations,
            data_docs_path=data_docs_path,
        )

    def _extract_failed_expectations(self, result: Any) -> list[str]:
        """
        Extract failed expectation names from validation result.

        In GE v1, ValidationDefinition.run() returns ExpectationSuiteValidationResult
        which has a `results` attribute with the list of expectation results.

        Parameters
        ----------
        result : ExpectationSuiteValidationResult
            The result from ValidationDefinition.run()

        Returns
        -------
        list[str]
            List of failed expectation descriptions
        """
        failed = []

        # GE v1 ValidationResult has results directly
        for result_item in result.results:
            if not result_item.success:
                exp_type = result_item.expectation_config.type
                column = result_item.expectation_config.kwargs.get("column", "")
                failed.append(f"{exp_type} [{column}]")

        return failed

    def _get_data_docs_path(self) -> str | None:
        """Return path to Data Docs index.html if it exists."""
        try:
            index_path = (
                Path(self.context.root_directory)
                / "uncommitted"
                / "data_docs"
                / "local_site"
                / "index.html"
            )
            return str(index_path) if index_path.exists() else None
        except Exception as e:
            self._log(f"[WARN]  Could not locate Data Docs: {e}")
            return None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
