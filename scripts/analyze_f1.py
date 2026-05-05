import mlflow
import numpy as np
from sklearn.metrics import classification_report
import joblib

mlflow.set_tracking_uri("file:///C:/Users/loren/Documents/mlops-portfolio/mlruns")

# Get the latest train_ml run
run = mlflow.get_run("c88d7141ad9546cf99b5ac9554ed96ba")

print("=" * 80)
print("MACRO F1 ANALYSIS — Threshold Optimization (τ=0.15)")
print("=" * 80)

# Load the model and test data
model = joblib.load("models/best_ml_model.pkl")
Z_test = np.load("data/processed/Z_test.npy")
y_test = np.load("data/processed/y_test.npy")

# Get predictions with threshold
test_probs = model.predict_proba(Z_test)

# Apply threshold logic (same as trainer.py)
y_pred = np.zeros(len(test_probs), dtype=int)
for i, probs in enumerate(test_probs):
    if probs[2] >= 0.15:  # fatal_threshold
        y_pred[i] = 2
    else:
        y_pred[i] = np.argmax(probs[:2])

# Standard argmax for comparison
y_pred_argmax = np.argmax(test_probs, axis=1)

print("\n" + "=" * 80)
print("WITH THRESHOLD (τ=0.15)")
print("=" * 80)
report_threshold = classification_report(y_test, y_pred, target_names=["PDO", "Injury", "Fatal"], digits=4)
print(report_threshold)

print("\n" + "=" * 80)
print("WITHOUT THRESHOLD (argmax)")
print("=" * 80)
report_argmax = classification_report(y_test, y_pred_argmax, target_names=["PDO", "Injury", "Fatal"], digits=4)
print(report_argmax)

# Extract macro F1 values
from sklearn.metrics import f1_score
macro_f1_threshold = f1_score(y_test, y_pred, average='macro')
macro_f1_argmax = f1_score(y_test, y_pred_argmax, average='macro')

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"Macro F1 (threshold):  {macro_f1_threshold:.4f}")
print(f"Macro F1 (argmax):     {macro_f1_argmax:.4f}")
print(f"Delta:                 {(macro_f1_threshold - macro_f1_argmax):+.4f}")
print(f"\nGate status (>0.35):   {'PASS ✅' if macro_f1_threshold > 0.35 else 'FAIL ❌'}")
print(f"Gap to gate:           {(macro_f1_threshold - 0.35):.4f}")

# Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR MACRO F1 IMPROVEMENT")
print("=" * 80)
if macro_f1_threshold < 0.35:
    gap = 0.35 - macro_f1_threshold
    print(f"\nNeed to improve macro F1 by {gap:.4f} ({gap*100:.2f}%) to pass gate.")
    print("\nBest options (from vae_fix_plan.md priority matrix):")
    print("  1. Tomek Links (T124) - Low complexity, Medium impact")
    print("     - Sharpens decision boundaries → improves precision across all classes")
    print("     - Should improve PDO/Injury F1 without hurting Fatal recall")
    print("  2. Danger Index Features (T123) - Medium complexity, High impact")
    print("     - Adds discriminative features → improves separation of all classes")
    print("     - Should lift F1 for PDO/Injury/Fatal together")
    print("  3. Tune threshold (T102-T103) - Scan τ ∈ [0.10, 0.20] on val set")
    print("     - Find optimal balance between Fatal recall and overall precision")
