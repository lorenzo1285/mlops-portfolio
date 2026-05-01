"""Extract and compare XGBoost vs MLP MLflow metrics."""
import json
import mlflow

mlflow.set_tracking_uri("mlruns/")
client = mlflow.tracking.MlflowClient()

xgb_runs = mlflow.search_runs(
    experiment_names=["crash-severity-ml"],
    order_by=["metrics.eout_macro_f1 DESC"],
)
dl_runs = mlflow.search_runs(
    experiment_names=["crash-severity-dl"],
    order_by=["metrics.eout_macro_f1 DESC"],
)

xgb_run_id = xgb_runs.iloc[0]["run_id"]
dl_run_id = dl_runs.iloc[0]["run_id"]

tmp_xgb = client.download_artifacts(xgb_run_id, "per_class_matrix.json", "/tmp/xgb")
tmp_dl  = client.download_artifacts(dl_run_id,  "per_class_matrix.json", "/tmp/dl")
with open(tmp_xgb) as f:
    xgb_matrix = json.load(f)
with open(tmp_dl) as f:
    dl_matrix = json.load(f)

HDR = f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
SEP = "-" * 52

print("=" * 52)
print("XGBoost -- per-class breakdown (test set)")
print("=" * 52)
print(HDR); print(SEP)
for cls, v in xgb_matrix.items():
    print(f"{cls:<10} {v['precision']:>10.3f} {v['recall']:>10.3f} {v['f1']:>10.3f} {v['support']:>10}")

print()
print("=" * 52)
print("MLP -- per-class breakdown (best seed, test set)")
print("=" * 52)
print(HDR); print(SEP)
for cls, v in dl_matrix.items():
    print(f"{cls:<10} {v['precision']:>10.3f} {v['recall']:>10.3f} {v['f1']:>10.3f} {v['support']:>10}")

print()
print("=" * 52)
print("Fatal class head-to-head")
print("=" * 52)
for metric in ["precision", "recall", "f1"]:
    xv = xgb_matrix["Fatal"][metric]
    dv = dl_matrix["Fatal"][metric]
    winner = "XGB" if xv > dv else "MLP"
    print(f"  Fatal {metric:<12}  XGB={xv:.4f}  MLP={dv:.4f}  delta={xv-dv:+.4f}  [{winner}]")

print()
print("=" * 52)
print("ROC AUC comparison")
print("=" * 52)
for cls in ["pdo", "injury", "fatal"]:
    col = f"metrics.roc_auc_{cls}"
    xv = xgb_runs[col].iloc[0] if col in xgb_runs.columns else float("nan")
    dv = dl_runs[col].iloc[0]   if col in dl_runs.columns  else float("nan")
    note = ""
    if col not in dl_runs.columns:
        note = " (MLP pre-dates plots.py)"
    print(f"  roc_auc_{cls:<8}  XGB={xv:.4f}  MLP={dv:.4f}  delta={xv-dv:+.4f}{note}")

print()
print("=" * 52)
print("Seed variance")
print("=" * 52)
xgb_unique = xgb_runs["metrics.eout_macro_f1"].unique()
print(f"  XGBoost: {len(xgb_unique)} unique F1 value(s) -- {'DETERMINISTIC' if len(xgb_unique) == 1 else 'variable'}")
dl_std = dl_runs["metrics.eout_macro_f1"].std()
print(f"  MLP:     std={dl_std:.4f}  min={dl_runs['metrics.eout_macro_f1'].min():.4f}  max={dl_runs['metrics.eout_macro_f1'].max():.4f}")

print()
print("=" * 52)
print("Overfitting check")
print("=" * 52)
for name, runs in [("XGBoost", xgb_runs), ("MLP    ", dl_runs)]:
    ein  = runs["metrics.ein_macro_f1"].mean()
    eout = runs["metrics.eout_macro_f1"].mean()
    gap  = runs["metrics.generalisation_gap"].mean()
    print(f"  {name}  ein={ein:.4f}  eout={eout:.4f}  gap={gap:.4f}")

print()
print("=" * 52)
print("Constitution VI gate check")
print("=" * 52)
gate_f1, gate_recall = 0.35, 0.50
for name, runs in [("XGBoost", xgb_runs), ("MLP    ", dl_runs)]:
    f1     = runs["metrics.eout_macro_f1"].mean()
    recall = runs["metrics.eout_fatal_recall"].mean()
    f1_pass = "PASS" if f1 > gate_f1 else "FAIL"
    rc_pass = "PASS" if recall > gate_recall else "FAIL"
    print(f"  {name}  macro_f1={f1:.4f} [{f1_pass}]   fatal_recall={recall:.4f} [{rc_pass}]")
