import os
import sys
from pathlib import Path

from src.config import load_config


def main() -> None:
    # TODO T029-T030: replace stub with full GE three-class workflow
    # (GEContextBuilder → GEManager → GECheckpointRunner)
    config = load_config()
    input_path = os.getenv("INPUT_PATH", config.data.raw_path)
    sentinel_path = os.getenv(
        "SENTINEL_PATH", os.path.join(config.data.processed_dir, ".validation_passed")
    )

    if not Path(input_path).exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    Path(sentinel_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sentinel_path).write_text("validated")
    print(f"validate: sentinel written to {sentinel_path}")


if __name__ == "__main__":
    main()
