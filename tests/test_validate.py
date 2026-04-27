"""Boundary tests for the validate stage (constitution XVI — raw data, not processed)."""
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

RAW_CSV = Path("data/raw/CGR_Crash_Data.csv")
GX_ROOT = "great_expectations/gx"
DATA_DOCS_INDEX = (
    Path(GX_ROOT) / "uncommitted" / "data_docs" / "local_site" / "index.html"
)


def _run_validate(input_path: str, sentinel_path: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "INPUT_PATH": input_path,
        "OUTPUT_SENTINEL": sentinel_path,
        "GX_ROOT": GX_ROOT,
    }
    return subprocess.run(
        ["uv", "run", "python", "-m", "src.validate.run"],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture(scope="module")
def raw_csv_path():
    """Raw CSV must exist — validate runs on raw data before ingest (constitution XVI)."""
    if not RAW_CSV.exists():
        pytest.skip(f"Raw data not found: {RAW_CSV}")
    return str(RAW_CSV)


# ---------------------------------------------------------------------------
# Clean data — happy path
# ---------------------------------------------------------------------------

class TestValidateCleanData:
    def test_exits_0_on_clean_csv(self, raw_csv_path, tmp_path):
        result = _run_validate(raw_csv_path, str(tmp_path / ".validation_passed"))
        assert result.returncode == 0, (
            f"Expected exit 0 on clean CSV\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_writes_sentinel_on_success(self, raw_csv_path, tmp_path):
        sentinel = tmp_path / ".validation_passed"
        _run_validate(raw_csv_path, str(sentinel))
        assert sentinel.exists(), "Sentinel file must be written after successful validation"

    def test_creates_data_docs_html(self, raw_csv_path, tmp_path):
        _run_validate(raw_csv_path, str(tmp_path / ".validation_passed"))
        assert DATA_DOCS_INDEX.exists(), (
            f"Data Docs HTML not created at {DATA_DOCS_INDEX}"
        )


# ---------------------------------------------------------------------------
# Corrupt data — failure path
# ---------------------------------------------------------------------------

class TestValidateCorruptData:
    @pytest.fixture
    def corrupt_csv(self, raw_csv_path, tmp_path):
        """200-row subset with SPEEDLIMIT=500 injected in 10 rows.
        10/200 = 5% failure rate, well above the 1.1% mostly=0.989 tolerance."""
        df = pd.read_csv(raw_csv_path, nrows=200)
        df.loc[:9, "SPEEDLIMIT"] = 500
        path = tmp_path / "corrupt.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_exits_1_on_out_of_range_value(self, corrupt_csv, tmp_path):
        result = _run_validate(corrupt_csv, str(tmp_path / ".validation_passed"))
        assert result.returncode == 1, (
            f"Expected exit 1 on corrupt CSV\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_names_failed_expectation_in_stdout(self, corrupt_csv, tmp_path):
        result = _run_validate(corrupt_csv, str(tmp_path / ".validation_passed"))
        combined = result.stdout + result.stderr
        assert "SPEEDLIMIT" in combined or "ExpectColumnValuesToBeBetween" in combined, (
            f"Expected failed expectation name in output\nstdout:\n{result.stdout}"
        )

    def test_no_sentinel_written_on_failure(self, corrupt_csv, tmp_path):
        sentinel = tmp_path / ".validation_passed"
        _run_validate(corrupt_csv, str(sentinel))
        assert not sentinel.exists(), "Sentinel must NOT be written when validation fails"


# ---------------------------------------------------------------------------
# Missing input
# ---------------------------------------------------------------------------

class TestValidateMissingInput:
    def test_exits_1_when_input_path_missing(self, tmp_path):
        result = _run_validate("/nonexistent/path.csv", str(tmp_path / ".validation_passed"))
        assert result.returncode == 1
