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


@dataclass
class DLTrainResult:
    """Result from DL training stage."""
    best_epoch: int
    best_val_loss: float
    model_path: str
    run_id: str
    seed: int


class ShallowMLP(nn.Module):
    """Shallow MLP: Linear(input_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, n_classes)."""
    
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
        x = self.fc2(x)
        return x


class DLTrainer:
    """Shallow MLP multi-seed training on Z-space with MLflow tracking.

    Trains a shallow MLP (Linear → ReLU → Dropout → Linear) on latent vectors
    with class weights and early stopping. Tracks N seeds and returns best by
    eout_macro_f1.

    Public interface
    ----------------
    train(Z_train, y_train, Z_val, y_val, Z_test, y_test) → DLTrainResult
    """

    def __init__(
        self, 
        dl_config, 
        mlflow_config, 
        seeds: list[int],
        model_config,
    ) -> None:
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
        # Disable autolog
        mlflow.autolog(disable=True)
        
        # Compute class weights (once, same for all seeds)
        class_weights = compute_class_weights(y_train, n_classes=self._model_config.n_classes)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        
        # Set MLflow experiment
        mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)
        mlflow.set_experiment(self._dl_config.experiment_name)
        
        best_f1 = -1.0
        best_result = None
        
        # Seed loop
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
            
            # Track best seed by eout_macro_f1
            run = mlflow.get_run(result.run_id)
            eout_f1 = run.data.metrics["eout_macro_f1"]
            
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
        """Train one seed and return DLTrainResult."""
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Build model
        model = ShallowMLP(
            input_dim=self._dl_config.input_dim,
            hidden_dim=self._dl_config.hidden_dim,
            n_classes=self._model_config.n_classes,
            dropout_p=self._dl_config.dropout_p,
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._dl_config.lr)
        
        # DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(Z_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self._dl_config.batch_size, 
            shuffle=True,
        )
        
        val_dataset = TensorDataset(
            torch.tensor(Z_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._dl_config.batch_size,
            shuffle=False,
        )
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log params
            mlflow.log_param("seed", seed)
            mlflow.log_param("input_dim", self._dl_config.input_dim)
            mlflow.log_param("hidden_dim", self._dl_config.hidden_dim)
            mlflow.log_param("dropout_p", self._dl_config.dropout_p)
            mlflow.log_param("lr", self._dl_config.lr)
            mlflow.log_param("batch_size", self._dl_config.batch_size)
            mlflow.log_param("epochs", self._dl_config.epochs)
            mlflow.log_param("patience", self._dl_config.patience)
            
            # Training loop with early stopping
            best_val_loss = float("inf")
            best_epoch = 0
            patience_counter = 0
            
            for epoch in range(self._dl_config.epochs):
                # Train
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * len(X_batch)
                train_loss /= len(Z_train)
                
                # Validate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item() * len(X_batch)
                val_loss /= len(Z_val)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # Save checkpoint
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
                Z_train_t = torch.tensor(Z_train, dtype=torch.float32)
                Z_test_t = torch.tensor(Z_test, dtype=torch.float32)
                
                train_outputs = model(Z_train_t)
                test_outputs = model(Z_test_t)
                
                y_train_pred = train_outputs.argmax(dim=1).numpy()
                y_test_pred = test_outputs.argmax(dim=1).numpy()
            
            # Compute mandatory metrics
            ein_macro_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
            eout_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
            generalisation_gap = ein_macro_f1 - eout_macro_f1
            
            # Fatal recall (class 2)
            fatal_mask = y_test == 2
            if fatal_mask.sum() > 0:
                eout_fatal_recall = (y_test_pred[fatal_mask] == 2).sum() / fatal_mask.sum()
            else:
                eout_fatal_recall = 0.0
            
            # Log mandatory metrics
            mlflow.log_metric("ein_macro_f1", ein_macro_f1)
            mlflow.log_metric("eout_macro_f1", eout_macro_f1)
            mlflow.log_metric("eout_fatal_recall", eout_fatal_recall)
            mlflow.log_metric("generalisation_gap", generalisation_gap)
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metric("best_val_loss", best_val_loss)
            
            # Log per-class matrix as artifact
            class_matrix = per_class_matrix(y_test, y_test_pred, ["PDO", "Injury", "Fatal"])
            matrix_path = Path("per_class_matrix.json")
            with open(matrix_path, "w") as f:
                json.dump(class_matrix, f, indent=2)
            mlflow.log_artifact(str(matrix_path))
            matrix_path.unlink()  # Clean up temp file
            
            # Save model checkpoint
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
