import os
import sys
from pathlib import Path

from src.config import load_config
from src.ingest.ingester import Ingester


def main() -> int:
    try:
        config = load_config()
        input_path = os.getenv("INPUT_PATH", config.data.raw_path)
        output_path = os.getenv("OUTPUT_PATH",
                                os.path.join(config.data.processed_dir, "raw.csv"))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        result = Ingester(input_path=input_path, output_path=output_path).run()

        print(f"Ingest: {result.row_count} rows written to {result.output_path}")
        return 0

    except FileNotFoundError as e:
        print(f"ERROR: Input not found: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
