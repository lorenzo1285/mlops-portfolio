"""
Baseline comparison — answers: does the VAE latent space earn its complexity?

Trains four models on the same training data and evaluates on the same test set.
All models use X_train_augmented (same data as the pipeline) except the dummy.

Run:
    uv run python baseline_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier

DATA = Path("data/processed")
PIPELINE_REPORT = Path("docs/evaluation_report.json")
CLASS_NAMES = ["PDO", "Injury", "Fatal"]
FATAL_CLASS = 2


def load_arrays() -> dict[str, np.ndarray]:
    keys = [
        "X_train_augmented", "y_train_augmented",
        "X_val", "y_val",
        "X_test", "y_test",
        "Z_train_augmented",
        "Z_val", "Z_test",
    ]
    return {k: np.load(DATA / f"{k}.npy") for k in keys}


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    return weights


def report(name: str, y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    fatal_recall = recall_score(y_test, y_pred, labels=[FATAL_CLASS], average="macro", zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0, labels=[0, 1, 2])

    row = {
        "model": name,
        "macro_f1": round(macro_f1, 4),
        "fatal_recall": round(fatal_recall, 4),
        "pdo_f1": round(per_class_f1[0], 4),
        "injury_f1": round(per_class_f1[1], 4),
        "fatal_f1": round(per_class_f1[2], 4),
        "fatal_gate": "PASS" if fatal_recall > 0.50 else "FAIL",
        "f1_gate": "PASS" if macro_f1 > 0.35 else "FAIL",
    }

    print(f"\n{'-'*55}")
    print(f"  {name}")
    print(f"{'-'*55}")
    f1_flag = "PASS" if macro_f1 > 0.35 else "FAIL"
    rc_flag = "PASS" if fatal_recall > 0.50 else "FAIL"
    print(f"  macro F1      : {macro_f1:.4f}  [{f1_flag}] (gate > 0.35)")
    print(f"  fatal recall  : {fatal_recall:.4f}  [{rc_flag}] (gate > 0.50)")
    print(f"  PDO F1        : {per_class_f1[0]:.4f}")
    print(f"  Injury F1     : {per_class_f1[1]:.4f}")
    print(f"  Fatal F1      : {per_class_f1[2]:.4f}")

    fatal_mask = y_test == FATAL_CLASS
    n_fatal = fatal_mask.sum()
    n_caught = (y_pred[fatal_mask] == FATAL_CLASS).sum()
    print(f"  Fatal caught  : {n_caught} / {n_fatal} ({n_caught/n_fatal*100:.1f}%)")
    return row


def main() -> None:
    print("Loading arrays...")
    d = load_arrays()
    X_tr, y_tr = d["X_train_augmented"], d["y_train_augmented"].astype(int)
    X_te, y_te = d["X_test"], d["y_test"].astype(int)
    Z_tr = d["Z_train_augmented"]
    Z_te = d["Z_test"]

    class_weights = compute_class_weights(y_tr)
    sample_weights = class_weights[y_tr]

    print(f"\nTraining set : {X_tr.shape[0]:,} rows, {X_tr.shape[1]} features")
    print(f"Test set     : {X_te.shape[0]:,} rows")
    classes, counts = np.unique(y_te, return_counts=True)
    for c, n in zip(classes, counts):
        print(f"  {CLASS_NAMES[c]}: {n} ({n/len(y_te)*100:.2f}%)")

    results = []

    # 1 — Majority class dummy
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    results.append(report("Dummy (majority class)", y_te, dummy.predict(X_te)))

    # 2 — Logistic regression on raw X (no VAE)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)
    results.append(report("Logistic Regression on X (no VAE)", y_te, lr.predict(X_te)))

    # 3 — XGBoost on raw X (no VAE) — same hyperparams as pipeline
    xgb_raw = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    xgb_raw.fit(X_tr, y_tr, sample_weight=sample_weights)
    results.append(report("XGBoost on X (no VAE, no latent space)", y_te, xgb_raw.predict(X_te)))

    # 4 — XGBoost on Z (our pipeline — for reference)
    xgb_z = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    z_weights = compute_class_weights(y_tr)[y_tr]
    xgb_z.fit(Z_tr, y_tr, sample_weight=z_weights)
    results.append(report("XGBoost on Z (VAE latent space — pipeline)", y_te, xgb_z.predict(Z_te)))

    # 5 — Pipeline champion result (from evaluation_report.json)
    if PIPELINE_REPORT.exists():
        with open(PIPELINE_REPORT) as f:
            rpt = json.load(f)
        winner = rpt.get("winner", "unknown")
        macro = rpt.get(f"{winner}_mean_macro_f1", rpt.get("macro_f1"))
        recall = rpt.get(f"{winner}_mean_fatal_recall", rpt.get("fatal_recall"))
        if macro and recall:
            print(f"\n{'-'*55}")
            print(f"  Pipeline champion ({winner}, 10 seeds, from evaluation_report.json)")
            print(f"{'-'*55}")
            f1_flag = "PASS" if macro > 0.35 else "FAIL"
            rc_flag = "PASS" if recall > 0.50 else "FAIL"
            print(f"  macro F1      : {macro:.4f}  [{f1_flag}]")
            print(f"  fatal recall  : {recall:.4f}  [{rc_flag}]")

    print(f"\n{'='*55}")
    print("  SUMMARY TABLE")
    print(f"{'='*55}")
    header = f"  {'Model':<42} {'F1':>6}  {'FatalRec':>8}"
    print(header)
    print(f"  {'-'*42} {'-'*6}  {'-'*8}")
    for r in results:
        flag = "[PASS]" if r["fatal_gate"] == "PASS" else "[FAIL]"
        print(f"  {r['model']:<42} {r['macro_f1']:>6.4f}  {r['fatal_recall']:>8.4f} {flag}")

    print(f"\n  Gate thresholds: macro_f1 > 0.35, fatal_recall > 0.50"  )
    print(f"  Fatal test cases: {(y_te == FATAL_CLASS).sum()} / {len(y_te)}")
    print()


if __name__ == "__main__":
    main()
