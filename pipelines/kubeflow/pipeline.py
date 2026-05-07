"""KFP v2 pipeline — 10-stage crash-severity MLOps pipeline.

Each component calls `dvc repro <stage>` against the shared PVC-mounted
project root at /app. All components use the same base image.

Usage:
    python pipelines/kubeflow/pipeline.py   # compiles pipeline.yaml
"""
from kfp import compiler, dsl
from kfp import kubernetes

BASE_IMAGE = "mlops-portfolio:latest"
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
    subprocess.run(["dvc", "repro", "augment"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def train_vae_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "train_vae"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def encode_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "encode"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def train_ml_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "train_ml"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def train_dl_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "train_dl"], check=True, cwd="/app")


@dsl.component(base_image=BASE_IMAGE)
def train_gmm_op() -> None:
    import subprocess
    subprocess.run(["dvc", "repro", "train_gmm"], check=True, cwd="/app")


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


if __name__ == "__main__":
    compiler.Compiler().compile(pipeline, "pipelines/kubeflow/pipeline.yaml")
    print("Compiled: pipelines/kubeflow/pipeline.yaml")
