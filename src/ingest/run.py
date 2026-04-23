import os
import sys
import pandas as pd
from src.utils import load_params


def main():
    params = load_params()
    input_path = os.getenv("INPUT_PATH", params["data"]["raw_path"])
    output_path = os.getenv("OUTPUT_PATH", params["data"]["processed_dir"] + "raw.csv")

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(input_path, low_memory=False)
    except Exception as e:
        print(f"ERROR: Could not read {input_path}: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Ingest complete: {len(df)} rows written to {output_path}")


if __name__ == "__main__":
    main()
