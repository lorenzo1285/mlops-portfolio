from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationResult:
    success: bool
    failed_expectations: list[str]
    data_docs_path: str


class DataValidator:
    """Build and run a GE expectation suite from params; write Data Docs.

    Expectations are generated programmatically from validation_config —
    no committed suite JSON. The same class works for any dataset whose
    column contract is encoded in params.yaml under the validation section.

    Public interface
    ----------------
    validate(df) → ValidationResult
        Runs all expectations against df and writes Data Docs HTML.
        Returns success=False and lists failing expectation names on failure.
    """

    def __init__(self, ge_context_root: str, validation_config: dict) -> None:
        self._ge_context_root = ge_context_root
        self._validation_config = validation_config

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        raise NotImplementedError
