import os
import subprocess
import yaml


def load_params(path=None):
    """Load params.yaml, honouring the PARAMS_PATH env var."""
    path = path or os.getenv("PARAMS_PATH", "params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def get_run_name(stage: str) -> str:
    """Return '{stage}-{git_short_sha}' for MLflow run traceability.

    Ties every MLflow run to the exact git commit so you can reproduce it
    with: git checkout <sha> && dvc checkout && dvc repro <stage>

    Falls back to '{stage}-unknown' outside a git repo.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        sha = "unknown"
    return f"{stage}-{sha}"
