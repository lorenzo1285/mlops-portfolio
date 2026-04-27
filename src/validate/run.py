import os
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.validate.validator import DataValidator


def main() -> None:
    """
    Validate stage entry point.

    Reads raw crash CSV, runs GE validation via DataValidator,
    and writes sentinel file on success.

    Exit codes:
        0: Validation passed
        1: Validation failed or error
    """
    config = load_config()

    # Read environment variables
    input_path = os.getenv("INPUT_PATH", config.data.raw_path)
    output_sentinel = os.getenv(
        "OUTPUT_SENTINEL",
        os.path.join(config.data.processed_dir, ".validation_passed"),
    )
    gx_root = os.getenv("GX_ROOT", "great_expectations/gx")

    # Validate input exists
    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Load raw data
    print(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    # Initialize validator
    validator = DataValidator(
        context_root_dir=gx_root,
        validation_columns=config.validation.columns,
        datasource_name=config.great_expectations.datasource_name,
        asset_name="crash_data_asset",
        suite_name=config.great_expectations.suite_name,
        verbose=True,
    )

    # Run validation
    print("\nRunning validation...")
    result = validator.validate(df)

    if result.success:
        # Write sentinel file on success
        sentinel_path = Path(output_sentinel)
        sentinel_path.parent.mkdir(parents=True, exist_ok=True)
        sentinel_path.write_text("validated")

        print(f"\n[OK] VALIDATION PASSED")
        print(f"Sentinel written to: {sentinel_path}")
        if result.data_docs_path:
            print(f"Data Docs: {result.data_docs_path}")
        sys.exit(0)
    else:
        print(f"\n[ERROR] VALIDATION FAILED")
        print(f"Failed expectations ({len(result.failed_expectations)}):")
        for exp in result.failed_expectations:
            print(f"  - {exp}")
        if result.data_docs_path:
            print(f"Data Docs: {result.data_docs_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
