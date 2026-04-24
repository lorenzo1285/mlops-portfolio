import os
import sys

from src.config import load_config
from src.ingest.ingester import Ingester


def main() -> None:
    config = load_config()
    input_path = os.getenv("INPUT_PATH", config.data.raw_path)
    output_path = os.getenv("OUTPUT_PATH", config.data.processed_dir + "raw.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        result = Ingester(input_path=input_path, output_path=output_path).run()
    except FileNotFoundError:
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingest: {result.row_count} rows written to {result.output_path}")


if __name__ == "__main__":
    main()
