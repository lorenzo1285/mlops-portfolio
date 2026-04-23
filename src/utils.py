import os
import yaml


def load_params(path=None):
    """Load params.yaml, honouring the PARAMS_PATH env var."""
    path = path or os.getenv("PARAMS_PATH", "params.yaml")
    with open(path) as f:
        return yaml.safe_load(f)
