"""Katib trial script - trains VAE + winner classifier for HPO."""
import argparse
import dataclasses
import os
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.config import load_config
from src.encode.encoder import LatentEncoder
from src.metrics import compute_class_weights
from src.train_dl.trainer import ShallowMLP
from src.train_vae.vae_trainer import DVAETrainer


def main():
    """Katib trial: train VAE with candidate params → encode → train winner → eval on val."""
    parser = argparse.ArgumentParser(description="Katib HPO trial for VAE hyperparameters")
    parser.add_argument("--beta_max", type=float, required=True, help="VAE beta_max (KL weight)")
    parser.add_argument("--latent_dim", type=int, required=True, help="VAE latent dimension")
    parser.add_argument("--winner", type=str, required=True, choices=["ml", "dl"], help="Winner classifier (ml=XGBoost, dl=MLP)")
    args = parser.parse_args()

    config = load_config()

    # Immutable override — never mutate the shared config object
    vae_config = dataclasses.replace(config.vae, beta_max=args.beta_max, latent_dim=args.latent_dim)

    data_dir = Path(config.data.processed_dir)
    X_train_aug = np.load(data_dir / "X_train_augmented.npy")
    y_train_aug = np.load(data_dir / "y_train_augmented.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")

    # VAE trains unsupervised on X_all (constitution II exception); y_train_aug
    # used only for WeightedRandomSampler on the train portion — no test labels.
    X_all = np.vstack([X_train_aug, X_val, X_test])
    y_for_sampler = np.concatenate([y_train_aug, np.zeros(len(X_val) + len(X_test), dtype=y_train_aug.dtype)])

    print(f"Trial: beta_max={args.beta_max}, latent_dim={args.latent_dim}, winner={args.winner}")
    print(f"  X_all: {X_all.shape}, X_val: {X_val.shape}")

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name_tune)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Train VAE — encoder/decoder written to tmpdir by DVAETrainer
        print("Training VAE...")
        trainer = DVAETrainer(
            vae_config,
            config.mlflow,
            run_name=f"trial_beta{args.beta_max}_dim{args.latent_dim}",
        )
        vae_result = trainer.train(X_all, y_all=y_for_sampler, output_dir=tmpdir)

        # Encode splits using the checkpoint DVAETrainer just wrote
        print("Encoding splits...")
        encoder = LatentEncoder(
            encoder_path=str(tmpdir / "vae_encoder.pth"),
            latent_dim=vae_config.latent_dim,
        )
        encode_result = encoder.encode(X_train_aug, y_train_aug, X_val, X_test)

        Z_train_aug = encode_result.Z_train_augmented
        Z_val = encode_result.Z_val

        # Train winner classifier (seed=0)
        print(f"Training {args.winner} classifier (seed=0)...")

        if args.winner == "ml":
            sample_weights = compute_class_weights(y_train_aug, n_classes=config.model.n_classes)
            sample_weight_array = np.array([sample_weights[y] for y in y_train_aug])

            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=config.model.n_classes,
                random_state=0,
                early_stopping_rounds=10,
                eval_metric="mlogloss",
                verbosity=0,
            )
            clf.fit(
                Z_train_aug,
                y_train_aug,
                sample_weight=sample_weight_array,
                eval_set=[(Z_val, y_val)],
                verbose=False,
            )
            y_val_pred = clf.predict(Z_val)

        else:  # args.winner == "dl"
            class_weights = compute_class_weights(y_train_aug, n_classes=config.model.n_classes)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

            model = ShallowMLP(
                input_dim=vae_config.latent_dim,
                hidden_dim=config.dl.hidden_dim,
                n_classes=config.model.n_classes,
                dropout_p=config.dl.dropout_p,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            class_weights_tensor = class_weights_tensor.to(device)

            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.dl.lr)

            Z_train_tensor = torch.tensor(Z_train_aug, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_aug, dtype=torch.long)
            Z_val_tensor = torch.tensor(Z_val, dtype=torch.float32)

            train_dataset = torch.utils.data.TensorDataset(Z_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.dl.batch_size,
                shuffle=True,
            )

            # Cap epochs for trial speed; limit comes from params.yaml tune.max_dl_trial_epochs
            epochs = min(config.dl.epochs, config.tune.max_dl_trial_epochs)
            model.train()
            for epoch in range(epochs):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                y_val_pred = model(Z_val_tensor.to(device)).argmax(dim=1).cpu().numpy()

        # Compute validation metrics
        val_macro_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

        fatal_mask = y_val == 2
        val_fatal_recall = (
            float((y_val_pred[fatal_mask] == 2).sum() / fatal_mask.sum())
            if fatal_mask.sum() > 0 else 0.0
        )

        val_fitness = val_macro_f1 * (1.0 if val_fatal_recall >= 0.50 else 0.5)

        print(f"  val_macro_f1: {val_macro_f1:.4f}")
        print(f"  val_fatal_recall: {val_fatal_recall:.4f}")
        print(f"  val_fitness: {val_fitness:.4f}")

        with mlflow.start_run(run_name=f"trial_beta{args.beta_max}_dim{args.latent_dim}_{args.winner}"):
            mlflow.set_tags({"trial_type": "katib", "winner": args.winner})
            mlflow.log_params({
                "beta_max": args.beta_max,
                "latent_dim": args.latent_dim,
                "winner": args.winner,
            })
            mlflow.log_metrics({
                "val_macro_f1": val_macro_f1,
                "val_fatal_recall": val_fatal_recall,
                "val_fitness": val_fitness,
                "vae_final_elbo": vae_result.final_elbo,
            })

        # Write metric for Katib File collector (EmptyDir injected at /var/log/katib/).
        # Also print to stdout as a fallback for local runs.
        metrics_path = Path("/var/log/katib/metrics.log")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(f"val_fitness={val_fitness:.6f}\n")
        print(f"val_fitness={val_fitness:.6f}", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
