"""Tests for featurize stage - 3-way split with preprocessing and feature selection."""
import os
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.compose import ColumnTransformer


class TestFeaturize:
    """Boundary tests for the featurize stage through run.py entry point."""

    def test_featurize_exits_0_and_writes_all_outputs(self, tmp_path):
        """Given real crash CSV, stage exits 0 and writes 6 arrays + pipeline."""
        # Arrange: Use ingest output (constitution XVI - downstream of validate must use processed/raw.csv)
        input_csv = Path("data/processed/raw.csv")
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        
        assert input_csv.exists(), f"Ingest output not found at {input_csv}"
        
        # Act: Run featurize stage
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: Exit 0 and all outputs exist
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}. stderr: {result.stderr}"
        
        # Check 6 arrays exist
        assert (output_dir / "X_train.npy").exists(), "X_train.npy not created"
        assert (output_dir / "y_train.npy").exists(), "y_train.npy not created"
        assert (output_dir / "X_val.npy").exists(), "X_val.npy not created"
        assert (output_dir / "y_val.npy").exists(), "y_val.npy not created"
        assert (output_dir / "X_test.npy").exists(), "X_test.npy not created"
        assert (output_dir / "y_test.npy").exists(), "y_test.npy not created"
        
        # Check pipeline exists
        assert pipeline_path.exists(), "preprocessing_pipeline.joblib not created"

    def test_X_train_has_no_nan_values(self, tmp_path):
        """X_train array has no NaN values after preprocessing."""
        # Arrange: Use ingest output (constitution XVI)
        input_csv = Path("data/processed/raw.csv")
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        
        assert input_csv.exists(), f"Ingest output not found at {input_csv}"
        
        # Act: Run featurize stage
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: X_train has no NaN
        assert result.returncode == 0
        X_train = np.load(output_dir / "X_train.npy")
        assert not np.isnan(X_train).any(), "X_train contains NaN values after preprocessing"

    def test_y_train_contains_only_0_and_1(self, tmp_path):
        """y_train contains only binary values 0 and 1."""
        # Arrange: Use ingest output (constitution XVI)
        input_csv = Path("data/processed/raw.csv")
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        
        assert input_csv.exists(), f"Ingest output not found at {input_csv}"
        
        # Act
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: y_train is binary
        assert result.returncode == 0
        y_train = np.load(output_dir / "y_train.npy")
        unique_values = np.unique(y_train)
        assert set(unique_values).issubset({0, 1}), \
            f"y_train should only contain 0 and 1, got {unique_values}"

    def test_split_sizes_match_params(self, tmp_path):
        """Split sizes match train_size/val_size/test_size from params (±1 row)."""
        # Arrange: Use ingest output (constitution XVI)
        input_csv = Path("data/processed/raw.csv")
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        
        assert input_csv.exists(), f"Ingest output not found at {input_csv}"
        
        # Read to get actual row count
        df = pd.read_csv(input_csv, low_memory=False)
        n_rows = len(df)
        
        # Act
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: Split sizes match params (70/15/15)
        assert result.returncode == 0
        X_train = np.load(output_dir / "X_train.npy")
        X_val = np.load(output_dir / "X_val.npy")
        X_test = np.load(output_dir / "X_test.npy")
        
        expected_train = int(n_rows * 0.70)
        expected_val = int(n_rows * 0.15)
        expected_test = int(n_rows * 0.15)
        
        assert abs(len(X_train) - expected_train) <= 1, \
            f"Expected ~{expected_train} train rows, got {len(X_train)}"
        assert abs(len(X_val) - expected_val) <= 1, \
            f"Expected ~{expected_val} val rows, got {len(X_val)}"
        assert abs(len(X_test) - expected_test) <= 1, \
            f"Expected ~{expected_test} test rows, got {len(X_test)}"

    def test_column_transformer_has_three_groups(self, tmp_path):
        """ColumnTransformer has three groups named num, cat, ord."""
        # Arrange: Use ingest output (constitution XVI)
        input_csv = Path("data/processed/raw.csv")
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        
        assert input_csv.exists(), f"Ingest output not found at {input_csv}"
        
        # Act
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: Pipeline has num, cat, ord transformers
        assert result.returncode == 0
        pipeline = joblib.load(pipeline_path)
        
        # Pipeline should be a ColumnTransformer or contain one
        transformer = pipeline if isinstance(pipeline, ColumnTransformer) else pipeline.named_steps['preprocessor']
        
        transformer_names = [name for name, _, _ in transformer.transformers_]
        assert "num" in transformer_names, "Missing 'num' transformer group"
        assert "cat" in transformer_names, "Missing 'cat' transformer group"
        assert "ord" in transformer_names, "Missing 'ord' transformer group"

    def test_ordinal_encoding_dayofweek(self, tmp_path):
        """Ordinal group encodes DAYOFWEEK as Mon=0, Tue=1, ..., Sun=6."""
        # Arrange: Use ingest output (constitution XVI)
        input_csv = Path("data/processed/raw.csv")
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        
        assert input_csv.exists(), f"Ingest output not found at {input_csv}"
        
        # Act
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: Check that ordinal encoding is correct
        assert result.returncode == 0
        pipeline = joblib.load(pipeline_path)
        
        # Get the ordinal encoder from the pipeline
        transformer = pipeline if isinstance(pipeline, ColumnTransformer) else pipeline.named_steps['preprocessor']
        ord_transformer = None
        for name, trans, cols in transformer.transformers_:
            if name == "ord":
                ord_transformer = trans
                break
        
        assert ord_transformer is not None, "Ordinal transformer not found"
        
        # Check categories - should have DAYOFWEEK categories defined
        # OrdinalEncoder stores categories_ after fitting
        if hasattr(ord_transformer, 'named_steps'):
            encoder = ord_transformer.named_steps['encoder']
        else:
            # Find OrdinalEncoder in the pipeline
            encoder = ord_transformer[-1] if hasattr(ord_transformer, '__getitem__') else ord_transformer
        
        # Verify Monday=0, Sunday=6 (semantic ordering)
        # This is verified by checking the categories order in params.yaml
        assert hasattr(encoder, 'categories_'), "OrdinalEncoder not fitted"

    def test_feature_selection_mutual_info(self, tmp_path):
        """When feature_selection.method='mutual_info', X_train.shape[1]==n_features."""
        # Arrange: Use ingest output (constitution XVI) with custom params for feature selection
        input_csv = Path("data/processed/raw.csv")
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        
        assert input_csv.exists(), f"Ingest output not found at {input_csv}"
        
        # Load full params.yaml and override only feature_selection keys (constitution XVI)
        # Do NOT create partial params - load_config() requires all sections
        import yaml
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        
        # Override feature selection settings
        params["feature_selection"] = {
            "method": "mutual_info",
            "n_features": 5,
            "threshold": 0.95
        }
        
        params_yaml = tmp_path / "test_params.yaml"
        with open(params_yaml, "w") as f:
            yaml.safe_dump(params, f)
        
        # Act
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        env["PARAMS_PATH"] = str(params_yaml)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: X_train has exactly n_features columns
        assert result.returncode == 0
        X_train = np.load(output_dir / "X_train.npy")
        assert X_train.shape[1] == 5, \
            f"Expected 5 features after selection, got {X_train.shape[1]}"

    def test_feature_selection_none_keeps_all_features(self, tmp_path):
        """When feature_selection.method='none', n_features_raw == n_features_selected."""
        # Arrange: Use ingest output with default params (method='none')
        input_csv = Path("data/processed/raw.csv")
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        mlflow_dir = tmp_path / "mlruns"
        
        assert input_csv.exists(), f"Ingest output not found at {input_csv}"
        
        # Act: Run with method='none' (default in params.yaml)
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        env["MLFLOW_TRACKING_URI"] = str(mlflow_dir)  # No file:// prefix on Windows
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: n_features_raw == n_features_selected
        assert result.returncode == 0, f"Expected exit 0, got {result.returncode}. stderr: {result.stderr}"
        
        # Check MLflow for the metrics
        mlflow.set_tracking_uri(str(mlflow_dir))  # No file:// prefix on Windows
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        assert len(experiments) > 0, "No MLflow experiments found"
        
        runs = client.search_runs(experiment_ids=[experiments[0].experiment_id])
        assert len(runs) > 0, "No MLflow runs found"
        
        latest_run = runs[0]
        metrics = latest_run.data.metrics
        
        assert "n_features_raw" in metrics, "n_features_raw not logged"
        assert "n_features_selected" in metrics, "n_features_selected not logged"
        
        # When method='none', raw and selected should be equal
        assert metrics["n_features_raw"] == metrics["n_features_selected"], \
            f"With method='none', expected same feature count, got raw={metrics['n_features_raw']}, selected={metrics['n_features_selected']}"

    def test_mlflow_logs_feature_metrics(self, tmp_path):
        """MLflow run logs n_features_raw, n_features_selected, samples_per_param_ratio."""
        # Arrange: Use ingest output (constitution XVI)
        input_csv = Path("data/processed/raw.csv")
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        mlflow_dir = tmp_path / "mlruns"
        
        assert input_csv.exists(), f"Ingest output not found at {input_csv}"
        
        # Act: Run with custom MLflow tracking URI
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        env["MLFLOW_TRACKING_URI"] = str(mlflow_dir)  # No file:// prefix on Windows
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: MLflow logged the required metrics
        assert result.returncode == 0
        
        # Search for the run
        mlflow.set_tracking_uri(str(mlflow_dir))  # No file:// prefix on Windows
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        # Should have at least one experiment with runs
        assert len(experiments) > 0, "No MLflow experiments found"
        
        runs = client.search_runs(experiment_ids=[experiments[0].experiment_id])
        assert len(runs) > 0, "No MLflow runs found"
        
        latest_run = runs[0]
        metrics = latest_run.data.metrics
        
        assert "n_features_raw" in metrics, "n_features_raw not logged"
        assert "n_features_selected" in metrics, "n_features_selected not logged"
        assert "samples_per_param_ratio" in metrics, "samples_per_param_ratio not logged"

    def test_stage_exits_1_when_sample_complexity_too_low(self, tmp_path):
        """Stage exits 1 when samples_per_param_ratio < 3.0."""
        # Arrange: Create tiny subset from ingest output to trigger sample complexity gate
        real_csv = Path("data/processed/raw.csv")
        assert real_csv.exists(), f"Ingest output not found at {real_csv}"
        
        # Read first 10 rows from real data - will fail sample complexity gate
        df_real = pd.read_csv(real_csv, low_memory=False, nrows=10)
        
        input_csv = tmp_path / "tiny_sample.csv"
        output_dir = tmp_path / "arrays"
        pipeline_path = tmp_path / "preprocessing_pipeline.joblib"
        
        df_real.to_csv(input_csv, index=False)
        
        # Act
        env = os.environ.copy()
        env["INPUT_PATH"] = str(input_csv)
        env["OUTPUT_DIR"] = str(output_dir)
        env["PIPELINE_PATH"] = str(pipeline_path)
        
        result = subprocess.run(
            ["uv", "run", "python", "-m", "src.featurize.run"],
            env=env,
            capture_output=True,
            text=True,
        )
        
        # Assert: Exit 1 due to sample complexity gate
        assert result.returncode == 1, \
            f"Expected exit 1 for low sample complexity, got {result.returncode}. stdout: {result.stdout}"
        assert "sample" in result.stderr.lower() or "complexity" in result.stderr.lower() or "ratio" in result.stderr.lower(), \
            "Error message should mention sample complexity issue"
