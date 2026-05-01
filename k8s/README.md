# Kubernetes Resources

## Installed Components

- **Kubeflow Pipelines (KFP)** v2.0.5-pns - Platform-agnostic with PNS executor
- **Katib** v0.17.0 - Hyperparameter optimization operator

## PersistentVolumeClaim

The `pvc.yaml` mounts the project root at `/app` in all pods:
```bash
kubectl apply -f k8s/pvc.yaml
kubectl get pv,pvc  # Verify Bound status
```

## Port Forwarding

### Katib UI (HPO experiments)
```bash
kubectl port-forward -n kubeflow svc/katib-ui 8080:80
# Access at http://localhost:8080
```

### ML Pipeline API (programmatic access)
```bash
kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888
# API endpoint: http://localhost:8888
```

### ML Pipeline UI (web frontend)
*Note: ml-pipeline-ui pod has ImagePullBackOff due to deprecated GCR image tags.  
KFP API is fully functional for programmatic pipeline submission.*

```bash
# Will not work until image is patched:
# kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

## Pod Status

Check running pods:
```bash
kubectl get pods -n kubeflow
```

Core components running:
- `ml-pipeline-*` - KFP API server, persistence agent, scheduled workflow
- `katib-controller`, `katib-db-manager`, `katib-mysql`, `katib-ui` - Katib HPO
- `minio-*` - Object storage for artifacts
- `mysql-*` - Metadata database
- `workflow-controller` - Argo Workflows
- `metadata-grpc-deployment` - ML Metadata

## Troubleshooting

### Image Pull Errors

If pods show `ImagePullBackOff`, patch deployment with working image:
```bash
# Example: Minio image fix
kubectl set image deployment/minio -n kubeflow minio=minio/minio:RELEASE.2021-02-14T04-01-33Z
```

### Restart Failed Pods

```bash
kubectl delete pod -n kubeflow -l app=<app-name>
# Deployment will recreate the pod
```

### Clean Reinstall

```bash
kubectl delete namespace kubeflow
# Then reapply manifests from tasks.md T070
```

## Trial Pod Configuration

Katib trial pods will mount `mlops-portfolio-pvc` at `/app` to access:
- `data/` - input splits, artifacts
- `params.yaml` - configuration
- `src/` - pipeline code
- `mlruns/` - MLflow tracking
- `models/` - trained model artifacts

See `k8s/katib/vae_experiment.yaml` for trial template configuration.

## Katib Hyperparameter Optimization

### Submit VAE Experiment

The `tune` stage submits the Katib Experiment CRD with `{{winner}}` injected:

```bash
# HyperparamTuner.tune() submits this via Python Kubernetes client
kubectl apply -f k8s/katib/vae_experiment.yaml
```

**Search space**:
- `beta_max`: [0.05, 0.1, 0.2, 0.5, 1.0] — KL divergence weight (lower = more discriminative)
- `latent_dim`: [8, 16, 32] — bottleneck width (wider = preserves feature interactions)

**Algorithm**: Bayesian optimization (15 trials max, 1 parallel)

**Fitness**: `val_fitness = val_macro_f1 * (1.0 if val_fatal_recall >= 0.50 else 0.5)`

### Monitor Experiment

```bash
# Via Katib UI
kubectl port-forward -n kubeflow svc/katib-ui 8080:80
# Open http://localhost:8080

# Via CLI
kubectl get experiments -n default
kubectl get trials -n default
kubectl describe experiment vae-hyperparameter-tuning -n default
```

### Trial Logs

```bash
# List trial pods
kubectl get pods -l trial -n default

# View trial logs
kubectl logs <trial-pod-name> -n default
```

Each trial runs `src/tune/trial.py`, which:
1. Loads candidate hyperparameters (beta_max, latent_dim)
2. Trains VAE with candidate params
3. Encodes all splits with frozen encoder
4. Trains winner classifier (XGBoost or MLP)
5. Evaluates on Z_val
6. Logs metrics to MLflow (`crash-severity-tune` experiment)
7. Prints `val_fitness=<float>` to stdout for Katib metrics collector

