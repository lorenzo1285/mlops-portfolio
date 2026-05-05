# Tune Stage Manual Verification Guide (T059)

## Overview

The tune stage infrastructure is complete. This guide provides steps to manually verify the end-to-end Katib HPO integration (T059).

## Prerequisites

1. **Kubernetes cluster running** with Docker Desktop
   ```powershell
   kubectl get nodes
   # Should show: docker-desktop   Ready
   ```

2. **Katib installed** in kubeflow namespace
   ```powershell
   kubectl get pods -n kubeflow | Select-String katib
   # Should show 4 pods: katib-controller, katib-db-manager, katib-mysql, katib-ui (all Running)
   ```

3. **PVC mounted** at /app in cluster
   ```powershell
   kubectl get pvc mlops-portfolio-pvc
   # Should show: Bound
   ```

4. **Docker image built** and available
   ```powershell
   docker images mlops-portfolio
   # Should show: latest tag
   ```

5. **Pipeline stages completed** through evaluate
   ```powershell
   uv run dvc status evaluate
   # Should show: nothing to execute or outputs exist
   ```

## Verification Steps

### 1. Ensure Katib UI is accessible

```powershell
# Port-forward Katib UI (run in background or separate terminal)
kubectl port-forward -n kubeflow svc/katib-ui 8080:80
```

Open browser: http://localhost:8080

### 2. Run the tune stage

```powershell
uv run dvc repro tune
```

**Expected behavior:**
- Script reads `docs/evaluation_report.json` to get winner (ml or dl)
- If gates PASSED: skips HPO, writes `{"skipped": true}` to `docs/katib_best_params.json`, exits 0
- If gates FAILED:
  - Loads `k8s/katib/vae_experiment.yaml`
  - Injects `{{winner}}` placeholder with actual winner value
  - Submits Experiment CRD to Kubernetes
  - Polls experiment status every 10 seconds
  - Prints status updates to console
  - Waits for all trials to complete (max_trials=15 in params.yaml)
  - Extracts best beta_max and latent_dim from currentOptimalTrial
  - Updates `params.yaml` with best hyperparameters under `vae.beta_max` and `vae.latent_dim`
  - Writes `docs/katib_best_params.json` with results
  - Exits 0

### 3. Monitor Katib UI

**Navigate to**: http://localhost:8080

**Verify**:
- New experiment named `vae-hyperparameter-tuning` appears
- Status shows Running → Succeeded
- Trials: 15 total (or fewer if early stopping)
- Optimal Trial shows best beta_max and latent_dim

### 4. Verify MLflow runs

```powershell
uv run mlflow ui
```

**Navigate to**: http://localhost:5000

**Verify**:
- Experiment `crash-severity-tune` exists
- 15 runs (one per trial) with tags:
  - `trial_type=katib`
  - `beta_max=<value>`
  - `latent_dim=<value>`
  - `winner=ml` or `winner=dl`
- Each run has metrics:
  - `val_fitness` (objective for Katib)
  - `val_macro_f1`
  - `val_fatal_recall`

### 5. Verify params.yaml updated

```powershell
Select-String -Path params.yaml -Pattern "beta_max|latent_dim" -Context 0,1
```

**Expected**:
```yaml
vae:
  beta_max: <optimal_value>  # e.g., 0.2, 0.5, 1.0, etc.
  latent_dim: <optimal_value>  # e.g., 8, 16, or 32
```

### 6. Verify DVC output created

```powershell
cat docs/katib_best_params.json
```

**Expected (if tuning ran)**:
```json
{
  "beta_max": 0.5,
  "latent_dim": 16,
  "val_fitness": 0.4234,
  "n_trials": 15,
  "winner": "ml"
}
```

**Expected (if skipped)**:
```json
{
  "skipped": true,
  "reason": "gates_passed"
}
```

## Troubleshooting

### Experiment stuck in Running

```powershell
# Check trial pods
kubectl get pods -n default -l job-role=trial

# View trial logs
kubectl logs -n default -l job-role=trial --tail=50
```

**Common issues**:
- Image pull errors → rebuild Docker image
- PVC mount failures → verify PVC exists and is Bound
- Python errors in trial.py → check logs for traceback

### No trials created

```powershell
# Check experiment status
kubectl get experiment vae-hyperparameter-tuning -o yaml
```

**Look for**:
- `status.conditions` with type=Failed
- `status.failedTrialList` for failure reasons

### tune stage exits with error

**Check**:
1. Kubernetes config: `kubectl config current-context` should show `docker-desktop`
2. Katib CRDs: `kubectl get crd experiments.kubeflow.org`
3. Trial script exists: `ls src/tune/trial.py`
4. Config valid: `uv run python -c "from src.config import load_config; load_config()"`

## Success Criteria (T059)

✅ Katib Experiment appears in UI at http://localhost:8080  
✅ 15 MLflow runs in `crash-severity-tune` experiment  
✅ `params.yaml` updated with best `beta_max` and `latent_dim`  
✅ `docs/katib_best_params.json` created with results  
✅ DVC stage completes with exit code 0  

## Next Steps

After T059 verification passes:

**T060**: Run `dvc repro` (full pipeline) to retrain with optimized hyperparameters  
**T051**: Run `dvc repro register` to promote champion model to registry
