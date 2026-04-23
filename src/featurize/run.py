import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from src.utils import load_params

FEATURE_COLS = [
    "HOUR", "DAYOFWEEK", "MONTH", "YEAR",
    "WEATHER", "SURFCOND", "LIGHTING",
    "SPEEDLIMIT", "RDNUMLANES", "RDWIDTH",
    "ROUTECLASS", "TRUNKLINE", "RDSUBTYPE",
    "DRIVER1AGE", "DRIVER1SEX", "DRIVER2AGE", "DRIVER2SEX",
    "VEH1TYPE", "VEH1USE", "VEH2TYPE", "VEH2USE",
    "CRASHTYPE", "TRAFCTLDEV", "NONTRAFFIC",
]
TARGET_COL = "CRASHSEVER"

NUMERIC_COLS = [
    "HOUR", "DAYOFWEEK", "MONTH", "YEAR",
    "SPEEDLIMIT", "RDNUMLANES", "RDWIDTH",
    "DRIVER1AGE", "DRIVER2AGE",
]
CATEGORICAL_COLS = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]


def encode_target(series):
    """Map CRASHSEVER → binary: 0 = PDO, 1 = injury or fatal."""
    return (series != "PDO").astype(int)


def main():
    params = load_params()
    input_path = os.getenv("INPUT_PATH", params["data"]["processed_dir"] + "raw.csv")
    output_dir = os.getenv("OUTPUT_DIR", params["data"]["processed_dir"])
    pipeline_path = os.getenv("PIPELINE_PATH", "models/preprocessing_pipeline.joblib")
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    sentinel = params["data"]["sentinel_value"]

    df = pd.read_csv(input_path, low_memory=False)
    initial_rows = len(df)

    # Select relevant columns (features + target)
    available = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        print(f"WARNING: Missing columns: {missing}", file=sys.stderr)

    df = df[available].copy()

    # Recode sentinel values to NaN
    for col in ["DRIVER1AGE", "DRIVER2AGE"]:
        if col in df.columns:
            df[col] = df[col].replace(sentinel, np.nan)

    # Drop rows with no target
    df = df.dropna(subset=[TARGET_COL])
    dropped = initial_rows - len(df)
    drop_pct = dropped / initial_rows
    if drop_pct > 0.05:
        print(f"ERROR: Dropped {drop_pct:.1%} of rows (threshold 5%)", file=sys.stderr)
        sys.exit(1)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols]
    y = encode_target(df[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    num_cols = [c for c in NUMERIC_COLS if c in feature_cols]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train_processed)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test_processed)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test.values)

    os.makedirs(os.path.dirname(pipeline_path) or ".", exist_ok=True)
    joblib.dump({"pipeline": preprocessor, "feature_cols": feature_cols}, pipeline_path)

    print(
        f"Featurize complete: {len(X_train)} train / {len(X_test)} test rows, "
        f"{X_train_processed.shape[1]} features, dropped {drop_pct:.1%}"
    )


if __name__ == "__main__":
    main()
