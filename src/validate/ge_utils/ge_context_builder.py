"""
ge_context_builder.py — Infrastructure Layer: initializes GE context, datasource, and empty asset only.

This module is the INFRASTRUCTURE layer. It initializes the Great Expectations  
file context and creates the datasource + dataframe asset. It does NOT create
expectation suites or add expectations — that is the responsibility of GEManager.

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
    1. builder = GEContextBuilder(context_root_dir, datasource_name, asset_name, verbose)
    2. builder.build()  # creates datasource + asset
    3. manager = GEManager(context_root_dir, validation_columns, verbose)
    4. manager.build_suite(suite_name, asset_name, datasource_name)  # generates expectations
    5. manager.select_asset_and_suite() → manager.set_batch_definition() → manager.pre_validate(df)
    6. runner = GECheckpointRunner(context, batch_definition, verbose)
    7. result = runner.run(df)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import from installed great_expectations package (not local directory)
import great_expectations as gx_installed
from great_expectations.data_context import FileDataContext


class GEContextBuilder:
    """
    Infrastructure Layer: initializes GE context and creates datasource + asset.
    
    This class is responsible ONLY for:
    - Initializing the FileDataContext
    - Creating the pandas datasource  
    - Creating the dataframe asset
    
    It does NOT create expectation suites or add expectations.
    Suite creation is handled by GEManager.build_suite().
    """

    def __init__(
        self,
        context_root_dir: str,
        datasource_name: str,
        asset_name: str,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the context builder.

        :param context_root_dir: Root directory of the Great Expectations file context.
        :param datasource_name: Name to assign to the GE pandas datasource.
        :param asset_name: Name to assign to the GE dataframe asset.
        :param verbose: If True, prints detailed logs.
        """
        self._context_root_dir = context_root_dir
        self._datasource_name = datasource_name
        self._asset_name = asset_name
        self.verbose = verbose
        self.context: FileDataContext | None = None

        self._log("=" * 60)
        self._log("GE CONTEXT BUILDER — INFRASTRUCTURE LAYER")
        self._log("=" * 60)
        self._log(f"Context root : {context_root_dir}")
        self._log(f"Datasource   : {datasource_name}")
        self._log(f"Asset        : {asset_name}")

    def build(self) -> FileDataContext:
        """
        Build the GE infrastructure: context + datasource + asset.
        
        Returns the initialized FileDataContext.
        No suite creation happens here.
        """
        self._log("\n1. INITIALIZING FILE CONTEXT")
        
        root = Path(self._context_root_dir)
        root.mkdir(parents=True, exist_ok=True)
        
        try:
            self.context = gx_installed.get_context(context_root_dir=str(root))
            self._log(f"[OK] Context initialized at: {root}")
        except Exception as e:
            self._log(f"[ERROR] Failed to initialize GE context: {e}")
            raise

        self._log("\n2. CREATING DATASOURCE + ASSET")
        
        # Create or update datasource
        datasource = self.context.data_sources.add_or_update_pandas(
            name=self._datasource_name
        )
        self._log(f"[OK] Datasource: {self._datasource_name}")
        
        # Create dataframe asset — skip if already registered
        existing_assets = {asset.name for asset in datasource.assets}
        if self._asset_name not in existing_assets:
            datasource.add_dataframe_asset(name=self._asset_name)
        self._log(f"[OK] Asset: {self._asset_name}")
        
        self._log("\n" + "=" * 60)
        self._log("INFRASTRUCTURE BUILD COMPLETE")
        self._log("=" * 60)
        self._log("Next: Use GEManager.build_suite() to create expectations")
        
        return self.context

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

