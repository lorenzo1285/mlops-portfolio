# ADR 0002: KFP Infrastructure Fixes — Docker Desktop Kubernetes

**Date**: 2026-05-10  
**Status**: Accepted

## Context

During Phase O4 (T073–T074), the KFP pipeline was uploaded and started on Docker Desktop
Kubernetes (KFP 2.2.0, kfp-kubernetes SDK 2.16.1). Four blocking infrastructure issues
prevented any pod from reaching completion. All four were diagnosed and resolved, and the
pipeline ran successfully (run `crash-severity-pipeline-9w4bl`,
ID `4ac2e9c8-dbda-4059-a328-fec2b90d1306`).

---

## Issue 1 — `pvcNameParameter` Unknown Field (KFP SDK/Cluster Version Mismatch)

**Symptom**

All pods failed immediately with:

```
F... KFP driver: failed to unmarshal Kubernetes config, error:
  unknown field "pvcNameParameter" in kfp_kubernetes.PvcMount
```

**Root Cause**

`kfp-kubernetes` SDK 2.16.1 compiles `mount_pvc()` calls into the platforms YAML section
using a `pvcNameParameter` wrapper object (camelCase, proto-incompatible wrapper). The KFP
2.2.0 cluster driver deserialises via the older proto where only the `constant` field
(oneof pvc_reference, field 2) is recognised.

**Fix**

Post-compilation YAML patch in `pipelines/kubeflow/pipeline.py`:

```python
def _patch_pipeline_yaml(path: str) -> None:
    """Strip pvcNameParameter from compiled KFP-Kubernetes extension YAML.
    kfp-kubernetes >= 2.x generates pvcNameParameter which KFP 2.2.0
    cluster driver does not recognise. `constant` is the correct field.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    parts = raw.split("---\n", 1)
    if len(parts) != 2:
        return
    main_part, k8s_part = parts
    k8s_doc = yaml.safe_load(k8s_part)
    executors = (k8s_doc.get("platforms", {}).get("kubernetes", {})
                 .get("deploymentSpec", {}).get("executors", {}))
    for executor_spec in executors.values():
        for mount in executor_spec.get("pvcMount", []):
            mount.pop("pvcNameParameter", None)
    patched = yaml.dump(k8s_doc, default_flow_style=False, allow_unicode=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(main_part + "---\n" + patched)
```

`compiler.Compiler().compile(pipeline, out)` is always followed by
`_patch_pipeline_yaml(out)`.

**Future mitigation**: If upgrading KFP cluster to match the SDK version, this patch can
be removed. Until then, always call `_patch_pipeline_yaml` after compilation. If the patch
stops working, inspect the compiled YAML platforms section and compare against the
cluster's proto descriptor at
`venv/lib/site-packages/kfp_kubernetes/kubernetes_executor_config_pb2.py`.

---

## Issue 2 — PVC Namespace Mismatch

**Symptom**

Pods in the `kubeflow` namespace could not mount the PVC:

```
Unable to attach or mount volumes: unmounted volumes=[pipeline-root-mlops-portfolio-pvc],
  unattached volumes=[...]: timed out waiting for the condition
```

**Root Cause**

Kubernetes PVCs are namespace-scoped. The original `k8s/pvc.yaml` creates
`mlops-portfolio-pvc` in the `default` namespace. KFP schedules pipeline pods in the
`kubeflow` namespace. Cross-namespace PVC references are not permitted.

**Fix**

Created `k8s/pvc-kubeflow.yaml` with a dedicated PersistentVolume and a matching PVC in
the `kubeflow` namespace:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mlops-portfolio-pv-kf
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: manual
  hostPath:
    path: /run/desktop/mnt/host/c/Users/loren/Documents/mlops-portfolio
    type: DirectoryOrCreate
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlops-portfolio-pvc
  namespace: kubeflow
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: manual
  volumeName: mlops-portfolio-pv-kf
```

Apply with:

```powershell
kubectl apply -f k8s/pvc-kubeflow.yaml
kubectl get pvc -n kubeflow   # expect STATUS=Bound
```

**Notes**

- The `default` namespace PVC (`k8s/pvc.yaml`) remains for local development that does
  not use KFP (e.g. manual kubectl jobs).
- The Docker Desktop hostPath uses `/run/desktop/mnt/host/c/...` — NOT a Windows path.
  This maps to `C:\...` on the host via Docker Desktop's bind mount layer.
- `storageClassName: manual` must match between PV and PVC for static provisioning.

---

## Issue 3 — `:latest` Tag Forces `Always` Pull Policy

**Symptom**

```
Failed to pull image "mlops-portfolio:latest":
  denied: requested access to the resource is denied
```

**Root Cause**

Kubernetes defaults to `imagePullPolicy: Always` when the tag is `:latest`. Docker
Desktop's k8s cluster shares the Docker daemon, so locally built images exist in the
local registry. However, `Always` policy bypasses the local cache and queries the remote
registry (Docker Hub), where the image does not exist.

**Fix**

Use a non-`:latest` tag. Any non-latest tag defaults to `IfNotPresent`, which uses the
local image without a remote pull.

```python
# pipelines/kubeflow/pipeline.py
BASE_IMAGE = "mlops-portfolio:local"  # was "mlops-portfolio:latest"
```

Rebuild after any dependency change:

```powershell
docker build -t mlops-portfolio:local -f docker/Dockerfile .
```

**Notes**

- If optuna, xgboost, or other packages are updated via `uv add`, the image must be
  rebuilt — the cluster does not pick up host-side uv changes.
- Verify installed packages inside the image before submitting a run:
  ```powershell
  docker run --rm mlops-portfolio:local python -c "import optuna; print(optuna.__version__)"
  ```

---

## Issue 4 — DVC `rwlock` Contention Between Parallel Pods

**Symptom**

Parallel stages (`augment` + `train_vae`, or `train_ml` + `train_dl` + `train_gmm`) would
fail with:

```
ERROR: Unable to acquire lock
```

**Root Cause**

DVC 3.x uses a project-level read-write lock at `.dvc/tmp/rwlock`. Any `dvc repro <stage>`
call acquires this lock for the duration of the stage run. Two concurrent pods on the same
PVC (same project root `/app`) both try to acquire the lock → one fails immediately.

**Fix**

Parallel stages bypass DVC entirely and call the stage module directly:

```python
# Sequential stages (safe — no concurrent DVC calls):
@dsl.component(base_image=BASE_IMAGE)
def validate_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "validate"], check=True, cwd="/app")

# Parallel stages (must bypass DVC):
@dsl.component(base_image=BASE_IMAGE)
def train_vae_op() -> None:
    import subprocess
    # Run module directly to avoid DVC rwlock contention with concurrent pods
    subprocess.run(["python", "-m", "src.train_vae.run"], check=True, cwd="/app")
```

Affected parallel groups:

| Group | Stages | Reason |
|-------|--------|--------|
| A | `augment`, `train_vae` | both depend only on featurize outputs |
| B | `train_ml`, `train_dl`, `train_gmm` | all depend only on encode outputs |

**Consequences**

- DVC cache is NOT updated for parallel stages when run via KFP. Artifacts are written
  directly to `data/processed/` and `models/` on the PVC.
- `dvc status` after a KFP run will show parallel stages as "changed". This is expected.
- `dvc repro` (local, sequential) still works correctly — each stage acquires/releases the
  lock in sequence.
- If DVC adds a per-stage lock in a future version, revert to `dvc repro <stage>` for
  all stages.

---

## Decision

All four fixes are accepted as permanent workarounds given the constraint of running KFP
2.2.0 on Docker Desktop with locally built images. They are encoded in:

- `pipelines/kubeflow/pipeline.py` — patch function + `BASE_IMAGE` + parallel stage strategy
- `k8s/pvc-kubeflow.yaml` — kubeflow-namespace PVC (separate from `k8s/pvc.yaml`)

## Rationale

Each fix is the minimal change required. No upstream version upgrades were made (KFP
cluster upgrades are risky in a learning portfolio; SDK downgrade would lose other fixes).
The YAML patch is deterministic and tested against the compiled output.

## Consequences

- Re-running `pipelines/kubeflow/pipeline.py` always produces a KFP 2.2.0-compatible
  `pipeline.yaml` — the patch is idempotent.
- Parallel-stage DVC cache misses are acceptable for a portfolio project. Production
  pipelines would use a DVC remote cache or move to a pipeline orchestrator that does not
  share project storage.
- If KFP cluster is upgraded to match the SDK (e.g. KFP 2.16.x), remove the
  `pvcNameParameter` patch and retest.
