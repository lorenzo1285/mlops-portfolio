from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import compute_class_weights, per_class_matrix
from src.plots import log_confusion_matrix, log_roc_curve

_CLASS_NAMES = ["PDO", "Injury", "Fatal"]


@dataclass
class DLTrainResult:
    best_epoch: int
    best_val_loss: float
    model_path: str
    run_id: str
    seed: int


class ShallowMLP(nn.Module):
    """Linear(input_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, n_classes)."""

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int, dropout_p: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class DLTrainer:
    """Shallow MLP multi-seed training on Z-space with MLflow tracking.

    Public interface
    ----------------
    train(Z_train, y_train, Z_val, y_val, Z_test, y_test) → DLTrainResult
    """

    def __init__(self, dl_config, mlflow_config, seeds: list[int], model_config) -> None:
        self._dl_config = dl_config
        self._mlflow_config = mlflow_config
        self._seeds = seeds
        self._model_config = model_config

    def train(
        self,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_val: np.ndarray,
        y_val: np.ndarray,
        Z_test: np.ndarray,
        y_test: np.ndarray,
    ) -> DLTrainResult:
        """Train shallow MLP across N seeds; return best by eout_macro_f1."""
        mlflow.autolog(disable=True)

        class_weights = compute_class_weights(y_train, n_classes=self._model_config.n_classes)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)
        mlflow.set_experiment(self._dl_config.experiment_name)

        best_f1 = -1.0
        best_result = None

        for seed in self._seeds:
            result = self._train_single_seed(
                seed=seed,
                Z_train=Z_train,
                y_train=y_train,
                Z_val=Z_val,
                y_val=y_val,
                Z_test=Z_test,
                y_test=y_test,
                class_weights_tensor=class_weights_tensor,
            )
            eout_f1 = mlflow.get_run(result.run_id).data.metrics["eout_macro_f1"]
            if eout_f1 > best_f1:
                best_f1 = eout_f1
                best_result = result

        return best_result

    def _train_single_seed(
        self,
        seed: int,
        Z_train: np.ndarray,
        y_train: np.ndarray,
        Z_val: np.ndarray,
        y_val: np.ndarray,
        Z_test: np.ndarray,
        y_test: np.ndarray,
        class_weights_tensor: torch.Tensor,
    ) -> DLTrainResult:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = ShallowMLP(
            input_dim=self._dl_config.input_dim,
            hidden_dim=self._dl_config.hidden_dim,
            n_classes=self._model_config.n_classes,
            dropout_p=self._dl_config.dropout_p,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._dl_config.lr)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(Z_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long),
            ),
            batch_size=self._dl_config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(Z_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            ),
            batch_size=self._dl_config.batch_size,
            shuffle=False,
        )

        with mlflow.start_run(run_name=f"mlp_seed_{seed}") as run:
            mlflow.log_params({
                "seed": seed,
                "input_dim": self._dl_config.input_dim,
                "hidden_dim": self._dl_config.hidden_dim,
                "dropout_p": self._dl_config.dropout_p,
                "lr": self._dl_config.lr,
                "batch_size": self._dl_config.batch_size,
                "epochs": self._dl_config.epochs,
                "patience": self._dl_config.patience,
            })

            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0
            checkpoint = {}

            for epoch in range(self._dl_config.epochs):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(X_batch)
                train_loss /= len(Z_train)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        val_loss += criterion(model(X_batch), y_batch).item() * len(X_batch)
                val_loss /= len(Z_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    checkpoint = {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self._dl_config.patience:
                        break

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                train_outputs = model(torch.tensor(Z_train, dtype=torch.float32))
                test_outputs = model(torch.tensor(Z_test, dtype=torch.float32))
                y_train_pred = train_outputs.argmax(dim=1).numpy()
                y_test_pred = test_outputs.argmax(dim=1).numpy()
                test_probs = torch.softmax(test_outputs, dim=1).numpy()

            # Mandatory metrics
            ein_macro_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
            eout_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
            fatal_mask = y_test == 2
            eout_fatal_recall = (
                float((y_test_pred[fatal_mask] == 2).sum() / fatal_mask.sum())
                if fatal_mask.sum() > 0 else 0.0
            )

            mlflow.log_metrics({
                "ein_macro_f1": ein_macro_f1,
                "eout_macro_f1": eout_macro_f1,
                "eout_fatal_recall": eout_fatal_recall,
                "generalisation_gap": ein_macro_f1 - eout_macro_f1,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            })

            # Per-class matrix JSON artifact
            matrix_path = Path("per_class_matrix.json")
            matrix_path.write_text(
                json.dumps(per_class_matrix(y_test, y_test_pred, _CLASS_NAMES), indent=2)
            )
            mlflow.log_artifact(str(matrix_path))
            matrix_path.unlink()

            # Visual diagnostics
            log_confusion_matrix(y_test, y_test_pred, _CLASS_NAMES)
            log_roc_curve(y_test, test_probs, _CLASS_NAMES)

            # Save checkpoint
            model_path = f"models/mlp_model_seed{seed}.pth"
            Path("models").mkdir(exist_ok=True)
            torch.save(checkpoint, model_path)

            return DLTrainResult(
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                model_path=model_path,
                run_id=run.info.run_id,
                seed=seed,
            )
