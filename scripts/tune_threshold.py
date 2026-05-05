"""
Threshold Tuning for Fatal Class Prediction

Scans τ ∈ [0.15, 0.50] to find optimal threshold that satisfies BOTH gates:
- Fatal recall >= 0.50
- Macro F1 > 0.35

Logs all candidate thresholds to MLflow and writes optimal τ* to params.yaml.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score
import mlflow
import yaml


def apply_threshold(probs: np.ndarray, tau: float) -> np.ndarray:
    """
    Apply threshold τ to probability array.
    
    Logic:
    - If P(Fatal|X) >= τ → predict Fatal (class 2)
    - Else → argmax over PDO/Injury (classes 0/1)
    
    Args:
        probs: (N, 3) probability array
        tau: fatal decision threshold
        
    Returns:
        (N,) predicted labels
    """
    y_pred = np.zeros(len(probs), dtype=int)
    for i, p in enumerate(probs):
        if p[2] >= tau:
            y_pred[i] = 2
        else:
            y_pred[i] = np.argmax(p[:2])
    return y_pred


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "best_ml_model.pkl"
    Z_val_path = project_root / "data" / "processed" / "Z_val.npy"
    y_val_path = project_root / "data" / "processed" / "y_val.npy"
    Z_test_path = project_root / "data" / "processed" / "Z_test.npy"
    y_test_path = project_root / "data" / "processed" / "y_test.npy"
    params_path = project_root / "params.yaml"
    
    # Load model and data
    print("Loading model and data...")
    model = joblib.load(model_path)
    Z_val = np.load(Z_val_path)
    y_val = np.load(y_val_path)
    Z_test = np.load(Z_test_path)
    y_test = np.load(y_test_path)
    
    # Get probabilities
    val_probs = model.predict_proba(Z_val)
    test_probs = model.predict_proba(Z_test)
    
    # MLflow setup
    mlflow.set_tracking_uri(f"file:///{project_root}/mlruns")
    experiment_name = "crash-severity-threshold-tuning"
    mlflow.set_experiment(experiment_name)
    
    # Threshold scan
    tau_candidates = np.arange(0.15, 0.51, 0.01)
    
    print(f"\nScanning {len(tau_candidates)} threshold candidates...")
    print("=" * 80)
    print(f"{'Threshold':>10} | {'Val F1':>8} | {'Val Recall':>11} | {'Gates':>6} | {'Test F1':>8} | {'Test Recall':>12}")
    print("=" * 80)
    
    results = []
    
    with mlflow.start_run(run_name="threshold_scan"):
        for tau in tau_candidates:
            # Validation set
            y_val_pred = apply_threshold(val_probs, tau)
            val_f1 = f1_score(y_val, y_val_pred, average='macro')
            val_recall = recall_score(y_val, y_val_pred, labels=[2], average='macro', zero_division=0)
            
            # Test set
            y_test_pred = apply_threshold(test_probs, tau)
            test_f1 = f1_score(y_test, y_test_pred, average='macro')
            test_recall = recall_score(y_test, y_test_pred, labels=[2], average='macro', zero_division=0)
            
            # Gate checks
            val_passes_recall = val_recall >= 0.50
            val_passes_f1 = val_f1 > 0.35
            both_pass = val_passes_recall and val_passes_f1
            
            gate_status = "✓✓" if both_pass else ("✓R" if val_passes_recall else ("✓F" if val_passes_f1 else "✗✗"))
            
            print(f"{tau:>10.2f} | {val_f1:>8.4f} | {val_recall:>11.4f} | {gate_status:>6} | {test_f1:>8.4f} | {test_recall:>12.4f}")
            
            results.append({
                'tau': tau,
                'val_f1': val_f1,
                'val_recall': val_recall,
                'test_f1': test_f1,
                'test_recall': test_recall,
                'both_pass': both_pass,
                'val_passes_recall': val_passes_recall,
                'val_passes_f1': val_passes_f1
            })
            
            # Log to MLflow
            mlflow.log_metric(f"val_f1_tau_{tau:.2f}", val_f1)
            mlflow.log_metric(f"val_recall_tau_{tau:.2f}", val_recall)
            mlflow.log_metric(f"test_f1_tau_{tau:.2f}", test_f1)
            mlflow.log_metric(f"test_recall_tau_{tau:.2f}", test_recall)
    
    print("=" * 80)
    
    # Find optimal threshold
    passing_candidates = [r for r in results if r['both_pass']]
    
    if passing_candidates:
        # Choose threshold with highest val_f1 among those that pass both gates
        best = max(passing_candidates, key=lambda r: r['val_f1'])
        print(f"\n✅ SOLUTION FOUND: τ* = {best['tau']:.2f}")
        print(f"   Val:  F1={best['val_f1']:.4f} (gate: >0.35), Recall={best['val_recall']:.4f} (gate: >=0.50)")
        print(f"   Test: F1={best['test_f1']:.4f}, Recall={best['test_recall']:.4f}")
    else:
        # Pareto-optimal: find threshold closest to satisfying both gates
        # Metric: minimize distance to (recall=0.50, f1=0.35) target
        print("\n⚠️  NO EXACT SOLUTION — Finding Pareto-optimal threshold...")
        
        for r in results:
            recall_gap = max(0, 0.50 - r['val_recall'])
            f1_gap = max(0, 0.35 - r['val_f1'])
            r['gap_distance'] = np.sqrt(recall_gap**2 + f1_gap**2)
        
        best = min(results, key=lambda r: r['gap_distance'])
        print(f"\n⚙️  PARETO-OPTIMAL: τ* = {best['tau']:.2f}")
        print(f"   Val:  F1={best['val_f1']:.4f} (gap: {max(0, 0.35 - best['val_f1']):.4f})")
        print(f"         Recall={best['val_recall']:.4f} (gap: {max(0, 0.50 - best['val_recall']):.4f})")
        print(f"   Test: F1={best['test_f1']:.4f}, Recall={best['test_recall']:.4f}")
    
    # Write to params.yaml
    print(f"\nUpdating params.yaml: model.fatal_threshold = {best['tau']:.2f}")
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    
    params['model']['fatal_threshold'] = float(best['tau'])
    
    with open(params_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Done. Run `dvc repro train_ml evaluate` to retrain with optimal threshold.")
    
    # Summary for MLflow
    with mlflow.start_run(run_name="threshold_optimal"):
        mlflow.log_param("optimal_tau", best['tau'])
        mlflow.log_metric("val_macro_f1", best['val_f1'])
        mlflow.log_metric("val_fatal_recall", best['val_recall'])
        mlflow.log_metric("test_macro_f1", best['test_f1'])
        mlflow.log_metric("test_fatal_recall", best['test_recall'])
        mlflow.log_metric("val_passes_recall_gate", float(best['val_passes_recall']))
        mlflow.log_metric("val_passes_f1_gate", float(best['val_passes_f1']))
        mlflow.log_metric("both_gates_pass", float(best['both_pass']))


if __name__ == "__main__":
    main()
