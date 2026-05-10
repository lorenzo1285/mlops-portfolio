"""KFP v2 pipeline — 10-stage crash-severity MLOps pipeline.

Each component calls `dvc repro <stage>` against the shared PVC-mounted
project root at /app. All components use the same base image.

Usage:
    python pipelines/kubeflow/pipeline.py   # compiles pipeline.yaml
"""
import yaml
from kfp import compiler, dsl
from kfp import kubernetes

BASE_IMAGE = "mlops-portfolio:local"
PVC_NAME = "mlops-portfolio-pvc"
MOUNT_PATH = "/app"


def _mount_pvc(task: dsl.PipelineTask) -> dsl.PipelineTask:
    return kubernetes.mount_pvc(task, pvc_name=PVC_NAME, mount_path=MOUNT_PATH)


@dsl.component(base_image=BASE_IMAGE)
def validate_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "validate"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def ingest_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "ingest"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def featurize_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "featurize"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def augment_op() -> None:
    import subprocess
    # Run module directly to avoid DVC rwlock contention with concurrent pods
    subprocess.run(["python", "-m", "src.augment.run"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def train_vae_op() -> None:
    import subprocess
    # Run module directly to avoid DVC rwlock contention with concurrent pods
    subprocess.run(["python", "-m", "src.train_vae.run"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def encode_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "encode"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def train_ml_op() -> None:
    import subprocess
    # Run module directly to avoid DVC rwlock contention with concurrent pods
    subprocess.run(["python", "-m", "src.train_ml.run"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def train_dl_op() -> None:
    import subprocess
    # Run module directly to avoid DVC rwlock contention with concurrent pods
    subprocess.run(["python", "-m", "src.train_dl.run"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def train_gmm_op() -> None:
    import subprocess
    # Run module directly to avoid DVC rwlock contention with concurrent pods
    subprocess.run(["python", "-m", "src.train_gmm.run"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def evaluate_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "evaluate"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def tune_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "tune"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def register_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "register"], check=True, cwd="/app")


@dsl.pipeline(name="crash-severity-pipeline")
def pipeline() -> None:
    validate = _mount_pvc(validate_op())

    ingest = _mount_pvc(ingest_op())
    ingest.after(validate)

    featurize = _mount_pvc(featurize_op())
    featurize.after(ingest)

    # augment and train_vae run in parallel after featurize
    augment = _mount_pvc(augment_op())
    augment.after(featurize)

    train_vae = _mount_pvc(train_vae_op())
    train_vae.after(featurize)

    encode = _mount_pvc(encode_op())
    encode.after(augment, train_vae)

    # train_ml, train_dl, train_gmm run in parallel after encode
    train_ml = _mount_pvc(train_ml_op())
    train_ml.after(encode)

    train_dl = _mount_pvc(train_dl_op())
    train_dl.after(encode)

    train_gmm = _mount_pvc(train_gmm_op())
    train_gmm.after(encode)

    evaluate = _mount_pvc(evaluate_op())
    evaluate.after(train_ml, train_dl, train_gmm)

    tune = _mount_pvc(tune_op())
    tune.after(evaluate)

    register = _mount_pvc(register_op())
    register.after(tune)


def _patch_pipeline_yaml(path: str) -> None:
    """Strip pvcNameParameter from the compiled KFP-Kubernetes extension YAML.

    kfp-kubernetes >= 2.x generates a pvcNameParameter wrapper that the
    KFP 2.2.0 cluster driver does not recognise.  This function rewrites the
    kubernetes platform section to the KFP 2.2.0 wire format:
      pvcNameParameter field removed
      constant field kept as-is (known to KFP 2.2.0 driver)
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    # pipeline.yaml has two YAML documents separated by "---"
    parts = raw.split("---\n", 1)
    if len(parts) != 2:
        return  # no kubernetes extension section; nothing to patch

    main_part, k8s_part = parts
    k8s_doc = yaml.safe_load(k8s_part)

    executors = (
        k8s_doc
        .get("platforms", {})
        .get("kubernetes", {})
        .get("deploymentSpec", {})
        .get("executors", {})
    )
    for executor_spec in executors.values():
        for mount in executor_spec.get("pvcMount", []):
            # The KFP 2.2.0 cluster driver understands 'constant' (oneof
            # pvc_reference field 2 in the proto) but NOT 'pvcNameParameter'
            # (added in kfp-kubernetes >= 2.x). Just remove the unknown field.
            mount.pop("pvcNameParameter", None)

    patched = yaml.dump(k8s_doc, default_flow_style=False, allow_unicode=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(main_part + "---\n" + patched)


if __name__ == "__main__":
    out = "pipelines/kubeflow/pipeline.yaml"
    compiler.Compiler().compile(pipeline, out)
    _patch_pipeline_yaml(out)
    print(f"Compiled (KFP 2.2.0-patched): {out}")
