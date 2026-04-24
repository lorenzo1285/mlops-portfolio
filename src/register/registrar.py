from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegistryReceipt:
    model_name: str
    version: str
    alias: str
    run_id: str
    winner: str


class ModelRegistrar:
    """Promote the winning model's best-seed artifact to the MLflow Model Registry.

    Sets the @champion alias on the registered version and writes a
    registry_receipt.json as the DVC output of the register stage.

    Public interface
    ----------------
    register(winner, run_id) → RegistryReceipt
        Promotes the artifact from run_id; assigns alias "champion".
    """

    def __init__(self, mlflow_config) -> None:
        self._mlflow_config = mlflow_config

    def register(self, winner: str, run_id: str) -> RegistryReceipt:
        raise NotImplementedError
