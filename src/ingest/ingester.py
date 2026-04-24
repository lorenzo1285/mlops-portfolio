from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class IngestResult:
    row_count: int
    output_path: str


class Ingester:
    """Copy raw tabular data from source path to processing location.

    Schema-agnostic: validates the source is readable and reports row count.
    Reusable for any CSV dataset by passing different input/output paths.
    """

    def __init__(self, input_path: str, output_path: str) -> None:
        self._input_path = input_path
        self._output_path = output_path

    def run(self) -> IngestResult:
        df = pd.read_csv(self._input_path, low_memory=False)
        df.to_csv(self._output_path, index=False)
        return IngestResult(row_count=len(df), output_path=self._output_path)
