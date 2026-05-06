from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.config import MLflowConfig, VAEConfig
from src.metrics import compute_class_weights


@dataclass
class VAETrainResult:
    best_epoch: int
    final_elbo: float
    encoder_path: str
    decoder_path: str
    run_id: str


class Encoder(nn.Module):
    """Encoder network: Linear → LayerNorm → ReLU stack → μ and log_σ²."""

    def __init__(self, input_dim: int, encoder_dims: list[int], latent_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Build encoder stack
        prev_dim = input_dim
        for hidden_dim in encoder_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Final layer for μ and log_σ²
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder stack.
        
        Returns:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        for layer in self.layers:
            x = layer(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    """Decoder network: mirrors encoder dims in reverse."""

    def __init__(self, latent_dim: int, decoder_dims: list[int], output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Build decoder stack (reverse of encoder)
        prev_dim = latent_dim
        for hidden_dim in decoder_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Final reconstruction layer
        self.fc_out = nn.Linear(prev_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder stack."""
        for layer in self.layers:
            z = layer(z)
        return self.fc_out(z)


class DenoisingBetaVAE(nn.Module):
    """Denoising β-VAE: applies dropout noise then reconstructs clean input."""

    def __init__(
        self,
        input_dim: int,
        encoder_dims: list[int],
        latent_dim: int,
        dropout_p: float,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, encoder_dims, latent_dim)
        # Decoder dims are encoder dims reversed
        decoder_dims = list(reversed(encoder_dims))
        self.decoder = Decoder(latent_dim, decoder_dims, input_dim)
        self.dropout_p = dropout_p

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + ε * σ."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, x_clean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: corrupt → encode → reparameterize → decode.
        
        Args:
            x_clean: Original clean input
            
        Returns:
            x_hat: Reconstructed output
            mu: Latent mean
            log_var: Latent log variance
            x_corrupted: Noisy input (for logging)
        """
        # Apply dropout corruption during training
        x_corrupted = F.dropout(x_clean, p=self.dropout_p, training=self.training)
        
        # Encode corrupted input
        mu, log_var = self.encoder(x_corrupted)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode to reconstruct clean input
        x_hat = self.decoder(z)
        
        return x_hat, mu, log_var, x_corrupted

    def loss_function(
        self,
        x_hat: torch.Tensor,
        x_clean: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO loss: reconstruction + β * KL divergence.
        
        Args:
            beta: KL divergence weight (annealed during training)
        
        Returns:
            total_loss: -ELBO (to minimize)
            recon_loss: MSE reconstruction loss
            kl_loss: KL divergence
        """
        # Reconstruction loss (target is clean x, not corrupted)
        recon_loss = F.mse_loss(x_hat, x_clean, reduction="mean")
        
        # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / x_clean.size(0)  # Normalize by batch size
        
        # Total ELBO loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


class DVAETrainer:
    """Trainer for Denoising β-VAE with early stopping and MLflow logging."""

    def __init__(
        self,
        vae_config: VAEConfig,
        mlflow_config: MLflowConfig,
        run_name: str | None = None,
    ) -> None:
        self._vae_config = vae_config
        self._mlflow_config = mlflow_config
        self._run_name = run_name

    def train(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray | None = None,
        output_dir: Path | None = None,
        optuna_trial: optuna.trial.Trial | None = None,
    ) -> VAETrainResult:
        """Train VAE on full dataset with early stopping.

        Args:
            X_all: Combined augmented-train/val/test data (unsupervised).
            y_all: Class labels for X_all — used only to build a WeightedRandomSampler
                   so Fatal rows get proportional gradient share. Reconstruction target
                   is always clean X (no label leakage).
            output_dir: Directory for encoder/decoder checkpoints. Defaults to Path("models").
            optuna_trial: Optional Optuna trial for hyperparameter optimization.
                          If provided, reports ELBO per epoch and raises TrialPruned if pruned.

        Returns:
            VAETrainResult with best checkpoint paths and metrics
        """
        # Set MLflow tracking
        mlflow.set_tracking_uri(self._mlflow_config.tracking_uri)
        mlflow.set_experiment(self._vae_config.experiment_name)

        # Prepare data
        input_dim = X_all.shape[1]

        # Split into train (90%) and val (10%) for early stopping
        n_total = len(X_all)
        n_train = int(0.9 * n_total)

        X_train = X_all[:n_train]
        X_val = X_all[n_train:]

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

        # Create data loaders — weighted sampler when y_all provided
        train_dataset = TensorDataset(X_train_tensor)
        if y_all is not None:
            y_train_vae = y_all[:n_train]
            class_w = compute_class_weights(y_train_vae, n_classes=3)
            sample_w = torch.DoubleTensor(class_w[y_train_vae.astype(int)])
            sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=self._vae_config.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=self._vae_config.batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val_tensor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._vae_config.batch_size,
            shuffle=False,
        )

        # Initialize model
        model = DenoisingBetaVAE(
            input_dim=input_dim,
            encoder_dims=self._vae_config.encoder_dims,
            latent_dim=self._vae_config.latent_dim,
            dropout_p=self._vae_config.dropout_p,
        )

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self._vae_config.lr)

        # Early stopping
        best_val_elbo = float("inf")
        best_epoch = 0
        patience_counter = 0

        # Create output directory for checkpoints
        checkpoint_dir = output_dir if output_dir is not None else Path("models")
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        encoder_path = str(checkpoint_dir / "vae_encoder.pth")
        decoder_path = str(checkpoint_dir / "vae_decoder.pth")
        
        # Start MLflow run (nested if called from Optuna)
        active_run = mlflow.active_run()
        with mlflow.start_run(run_name=self._run_name, nested=active_run is not None) as run:
            # Log parameters
            mlflow.log_params({
                "encoder_dims": str(self._vae_config.encoder_dims),
                "latent_dim": self._vae_config.latent_dim,
                "beta": self._vae_config.beta,
                "dropout_p": self._vae_config.dropout_p,
                "epochs": self._vae_config.epochs,
                "patience": self._vae_config.patience,
                "batch_size": self._vae_config.batch_size,
                "lr": self._vae_config.lr,
                "input_dim": input_dim,
                "n_train_samples": n_train,
                "weighted_sampler": y_all is not None,
                "n_fatal_train": int((y_all[:n_train] == 2).sum()) if y_all is not None else 0,
            })
            
            # Training loop
            for epoch in range(self._vae_config.epochs):
                # Compute KL annealing beta for this epoch
                if self._vae_config.cyclical_annealing:
                    # Cyclical schedule: β resets to 0 every cycle_epochs
                    epoch_in_cycle = epoch % self._vae_config.cycle_epochs
                    beta_t = self._vae_config.beta_max * min(
                        1.0,
                        epoch_in_cycle / max(1, self._vae_config.warmup_epochs)
                    )
                else:
                    # Monotonic schedule: β ramps from beta_start to beta_max once
                    beta_t = min(
                        self._vae_config.beta_max,
                        self._vae_config.beta_start + 
                        (self._vae_config.beta_max - self._vae_config.beta_start) * 
                        epoch / max(1, self._vae_config.warmup_epochs)
                    )
                
                # Train
                model.train()
                train_loss = 0
                train_recon = 0
                train_kl = 0
                
                for (batch_x,) in train_loader:
                    optimizer.zero_grad()
                    x_hat, mu, log_var, _ = model(batch_x)
                    loss, recon, kl = model.loss_function(x_hat, batch_x, mu, log_var, beta_t)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * len(batch_x)
                    train_recon += recon.item() * len(batch_x)
                    train_kl += kl.item() * len(batch_x)
                
                train_loss /= len(X_train)
                train_recon /= len(X_train)
                train_kl /= len(X_train)
                
                # Validation
                model.eval()
                val_loss = 0
                val_recon = 0
                val_kl = 0
                
                with torch.no_grad():
                    for (batch_x,) in val_loader:
                        x_hat, mu, log_var, _ = model(batch_x)
                        loss, recon, kl = model.loss_function(x_hat, batch_x, mu, log_var, beta_t)
                        
                        val_loss += loss.item() * len(batch_x)
                        val_recon += recon.item() * len(batch_x)
                        val_kl += kl.item() * len(batch_x)
                
                val_loss /= len(X_val)
                val_recon /= len(X_val)
                val_kl /= len(X_val)
                
                # Log metrics to MLflow
                # ELBO is negative loss (we minimize loss, maximize ELBO)
                val_elbo = -val_loss
                mlflow.log_metric("vae_elbo", val_elbo, step=epoch)
                mlflow.log_metric("vae_reconstruction_loss", val_recon, step=epoch)
                mlflow.log_metric("vae_kl_loss", val_kl, step=epoch)
                mlflow.log_metric("train_elbo", -train_loss, step=epoch)
                mlflow.log_metric("kl_beta", beta_t, step=epoch)
                
                # Optuna trial reporting and pruning
                if optuna_trial is not None:
                    optuna_trial.report(val_elbo, epoch)
                    if optuna_trial.should_prune():
                        raise optuna.TrialPruned()
                
                # Early stopping — patience only counted after warmup to avoid false
                # triggers while KL beta is still rising (ELBO naturally increases
                # during annealing even as the model improves).
                warmup_threshold = (
                    self._vae_config.cycle_epochs
                    if self._vae_config.cyclical_annealing
                    else self._vae_config.warmup_epochs
                )

                if val_loss < best_val_elbo:
                    best_val_elbo = val_loss
                    best_epoch = epoch
                    patience_counter = 0

                    torch.save(
                        {
                            "state_dict": model.encoder.state_dict(),
                            "encoder_dims": self._vae_config.encoder_dims,
                            "latent_dim": self._vae_config.latent_dim,
                            "input_dim": input_dim,
                        },
                        encoder_path,
                    )
                    torch.save(
                        {
                            "state_dict": model.decoder.state_dict(),
                            "decoder_dims": list(reversed(self._vae_config.encoder_dims)),
                            "latent_dim": self._vae_config.latent_dim,
                            "output_dim": input_dim,
                        },
                        decoder_path,
                    )
                elif epoch >= warmup_threshold:
                    patience_counter += 1
                    if patience_counter >= self._vae_config.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            # Log final metrics
            final_elbo = -best_val_elbo
            mlflow.log_metric("final_elbo", final_elbo)
            mlflow.log_metric("best_epoch", best_epoch)
            
            run_id = run.info.run_id
        
        return VAETrainResult(
            best_epoch=best_epoch,
            final_elbo=final_elbo,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            run_id=run_id,
        )
