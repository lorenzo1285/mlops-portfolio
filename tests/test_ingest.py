"""Tests for ingest stage - schema-agnostic CSV copy."""
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest


class TestIngest:
    """Boundary tests for the ingest stage through run.py entry point."""

    def test_ingest_copies_csv_with_same_row_count(self, tmp_path):
        """Given a valid CSV, stage exits 0 and produces output with same row count."""
        # Arrange: Create a source CSV with known row count
        input_csv = tmp_path / "source.csv"
        output_csv = tmp_path / "output.csv"
        
        test_data = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        test_data.to_csv(input_csv, index=False)
        input_row_count = len(test_data)
        
        # Act: Run ingest stage
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_PATH"] = str(output_csv)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.ingest.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: Exit 0 and output CSV exists with same row count
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}. stderr: {result.stderr}"
        assert output_csv.exists(), "Output CSV was not created"
        
        output_data = pd.read_csv(output_csv)
        assert len(output_data) == input_row_count, \
            f"Expected {input_row_count} rows, got {len(output_data)}"

    def test_ingest_exits_1_when_input_missing(self, tmp_path):
        """Given a missing input file, stage exits 1 without writing output."""
        # Arrange: Non-existent input path
        input_csv = tmp_path / "nonexistent.csv"
        output_csv = tmp_path / "output.csv"
        
        assert not input_csv.exists(), "Input should not exist for this test"
        
        # Act: Run ingest stage
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_PATH"] = str(output_csv)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.ingest.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: Exit 1 and no output written
        assert result.returncode == 1, \
            f"Expected exit 1 for missing input, got {result.returncode}. stdout: {result.stdout}"
        assert not output_csv.exists(), \
            "Output CSV should not be created when input is missing"

    def test_ingest_preserves_schema(self, tmp_path):
        """Output CSV has the same columns as input (schema-agnostic)."""
        # Arrange: CSV with specific column names
        input_csv = tmp_path / "source.csv"
        output_csv = tmp_path / "output.csv"
        
        test_data = pd.DataFrame({
            "HOUR": [1, 2, 3],
            "WEATHER": ["Clear", "Rain", "Snow"],
            "CRASHSEVER": [0, 1, 0]
        })
        test_data.to_csv(input_csv, index=False)
        
        # Act: Run ingest stage
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_PATH"] = str(output_csv)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.ingest.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: Schema preserved
        assert result.returncode == 0
        output_data = pd.read_csv(output_csv)
        assert list(output_data.columns) == list(test_data.columns), \
            "Output schema should match input schema exactly"

    def test_ingest_creates_output_directory_if_missing(self, tmp_path):
        """Stage creates the output directory if it doesn't exist."""
        # Arrange: Input exists, output dir does not
        input_csv = tmp_path / "source.csv"
        output_dir = tmp_path / "nested" / "output" / "dir"
        output_csv = output_dir / "data.csv"
        
        test_data = pd.DataFrame({"col": [1, 2, 3]})
        test_data.to_csv(input_csv, index=False)
        
        assert not output_dir.exists(), "Output dir should not exist initially"
        
        # Act: Run ingest stage
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_PATH"] = str(output_csv)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.ingest.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: Directory created and CSV written
        assert result.returncode == 0
        assert output_dir.exists(), "Output directory should be created"
        assert output_csv.exists(), "Output CSV should be written"
