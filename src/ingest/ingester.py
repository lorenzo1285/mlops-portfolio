from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class IngestResult:
    row_count: int
    output_path: str


class Ingester:
    """Copy raw CSV from source path to output path, schema-agnostic."""

    def __init__(self, input_path: str, output_path: str) -> None:
        self._input_path = Path(input_path)
        self._output_path = Path(output_path)

    def run(self) -> IngestResult:
        df = pd.read_csv(self._input_path, low_memory=False)
        df.to_csv(self._output_path, index=False)
        return IngestResult(row_count=len(df), output_path=str(self._output_path))
