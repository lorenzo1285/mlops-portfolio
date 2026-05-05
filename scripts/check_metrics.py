import mlflow

mlflow.set_tracking_uri("file:///C:/Users/loren/Documents/mlops-portfolio/mlruns")
run = mlflow.get_run("c88d7141ad9546cf99b5ac9554ed96ba")

print("=" * 60)
print("TRAIN_ML METRICS (XGBoost with threshold=0.15)")
print("=" * 60)
print(f"eout_fatal_recall:  {run.data.metrics.get('eout_fatal_recall', 'N/A')}")
print(f"eout_macro_f1:      {run.data.metrics.get('eout_macro_f1', 'N/A')}")
print(f"eval_fatal_recall:  {run.data.metrics.get('eval_fatal_recall', 'N/A')}")
print(f"eval_macro_f1:      {run.data.metrics.get('eval_macro_f1', 'N/A')}")
print(f"fatal_threshold:    {run.data.metrics.get('fatal_threshold', 'N/A')}")
print(f"generalisation_gap: {run.data.metrics.get('generalisation_gap', 'N/A')}")
print("=" * 60)
print(f"\nGATE CHECK (eout_fatal_recall >= 0.50): ", end="")
recall = run.data.metrics.get('eout_fatal_recall', 0)
if recall >= 0.50:
    print(f"PASS ({recall:.4f})")
else:
    print(f"FAIL ({recall:.4f})")
