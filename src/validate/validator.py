from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.validate.ge_utils.ge_context_builder import GEContextBuilder
from src.validate.ge_utils.ge_manager import GEManager
from src.validate.ge_utils.ge_checkpoint_runner import (
    GECheckpointRunner,
    CheckpointRunResult,
)
from src.config import ColumnContract


@dataclass
class ValidationResult:
    success: bool
    failed_expectations: list[str]
    data_docs_path: str | None


class DataValidator:
    """
    Orchestrates the three-layer GE validation workflow.

    Layer architecture:
        1. GEContextBuilder — infrastructure (datasource + asset)
        2. GEManager — suite building + preparation
        3. GECheckpointRunner — execution

    Public interface
    ----------------
    validate(df) → ValidationResult
        Runs all expectations against df and writes Data Docs HTML.
        Returns success=False and lists failing expectation names on failure.
    """

    def __init__(
        self,
        context_root_dir: str,
        validation_columns: dict[str, ColumnContract],
        datasource_name: str,
        asset_name: str,
        suite_name: str,
        run_name: str | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the validator.

        Parameters
        ----------
        context_root_dir : str
            Path to GE file context root.
        validation_columns : dict[str, ColumnContract]
            Per-column contracts from params.yaml validation.columns.
        datasource_name : str
            Name of the GE datasource.
        asset_name : str
            Name of the GE dataframe asset.
        suite_name : str
            Name of the expectation suite.
        run_name : str | None
            Optional name for this validation run. Organizes Data Docs by run name.
            Defaults to None (GE uses "__none__" placeholder).
        verbose : bool
            If True, prints detailed logs.
        """
        self.context_root_dir = context_root_dir
        self.validation_columns = validation_columns
        self.datasource_name = datasource_name
        self.asset_name = asset_name
        self.suite_name = suite_name
        self.run_name = run_name
        self.verbose = verbose

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run the complete validation workflow.

        Workflow:
            1. GEContextBuilder.build() — create infrastructure
            2. GEManager.build_suite() — generate expectations
            3. GEManager preparation (select, batch def, pre-validate)
            4. GECheckpointRunner.run() — execute validation

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Returns
        -------
        ValidationResult
            Contains success status, failed expectations, and Data Docs path.
        """
        # Layer 1: Infrastructure
        builder = GEContextBuilder(
            context_root_dir=self.context_root_dir,
            datasource_name=self.datasource_name,
            asset_name=self.asset_name,
            verbose=self.verbose,
        )
        context = builder.build()

        # Layer 2: Suite building + preparation (reuse builder's context — single open)
        manager = GEManager(
            context_root_dir=self.context_root_dir,
            verbose=self.verbose,
            context=context,
        )

        # Build suite from column contracts
        manager.build_suite(
            suite_name=self.suite_name,
            asset_name=self.asset_name,
            datasource_name=self.datasource_name,
            validation_columns=self.validation_columns,
        )

        # Prepare for validation
        manager.select_asset_and_suite(
            asset_name=self.asset_name,
            suite_name=self.suite_name,
            datasource_name=self.datasource_name,
        )
        manager.set_batch_definition()
        is_safe = manager.pre_validate(df)

        if not is_safe:
            raise RuntimeError(
                "Pre-validation failed. Check DataFrame shape and columns."
            )

        # Layer 3: Execution
        runner = GECheckpointRunner(
            context=context,
            batch_definition=manager.batch_definition,
            suite=manager.suite,
            run_name=self.run_name,
            verbose=self.verbose,
        )

        result: CheckpointRunResult = runner.run(df)

        return ValidationResult(
            success=result.success,
            failed_expectations=result.failed_expectations,
            data_docs_path=result.data_docs_path,
        )
