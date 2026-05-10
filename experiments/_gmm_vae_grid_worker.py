"""Worker script for one (latent_dim, beta_max) grid cell.

Called by gmm_vae_grid_experiment.py as a subprocess.
Prints a single JSON object to stdout on success; exits non-zero on failure.

Usage:
    python experiments/_gmm_vae_grid_worker.py --latent-dim 16 --beta-max 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

# torch must load before numpy/sklearn to avoid Windows DLL heap fragmentation
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
import torch  # noqa: F401 — must be first substantial import on Windows

import numpy as np
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture

from src.config import load_config
from src.encode.encoder import LatentEncoder
from src.train_gmm.trainer import GMMClassifier
from src.train_vae.vae_trainer import DVAETrainer

_N_CLASSES = 3
_BEST_GMM = dict(
    n_components={0: 5, 1: 3, 2: 4},
    covariance_type="diag",
    reg_covar=2.7177e-4,
    fatal_prior_boost=17.663,
    n_init=12,
)


def _fit_gmm(Z_train: np.ndarray, y_train: np.ndarray) -> GMMClassifier:
    priors = np.array([(y_train == c).sum() / len(y_train) for c in range(_N_CLASSES)])
    log_priors = np.log(np.clip(priors, 1e-9, None))
    gmms: dict[int, GaussianMixture] = {}
    for cls in range(_N_CLASSES):
        X_cls = Z_train[y_train == cls]
        n_comp = min(_BEST_GMM["n_components"][cls], len(X_cls))
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=_BEST_GMM["covariance_type"],
            reg_covar=_BEST_GMM["reg_covar"],
            max_iter=200,
            n_init=_BEST_GMM["n_init"],
            random_state=42,
        )
        gmm.fit(X_cls)
        gmms[cls] = gmm
    return GMMClassifier(gmms=gmms, log_priors=log_priors,
                         fatal_prior_boost=_BEST_GMM["fatal_prior_boost"])


def _metrics(clf: GMMClassifier, Z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    y_pred = clf.predict(Z)
    f1 = float(f1_score(y, y_pred, average="macro", zero_division=0))
    fatal_mask = y == 2
    recall = float((y_pred[fatal_mask] == 2).sum() / fatal_mask.sum()) if fatal_mask.sum() > 0 else 0.0
    return f1, recall


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--beta-max", type=float, required=True)
    args = parser.parse_args()

    config = load_config()
    processed = _root / "data" / "processed"

    X_train = np.load(processed / "X_train.npy")
    X_val   = np.load(processed / "X_val.npy")
    X_test  = np.load(processed / "X_test.npy")
    y_train = np.load(processed / "y_train.npy")
    y_val   = np.load(processed / "y_val.npy")
    y_test  = np.load(processed / "y_test.npy")
    X_train_aug = np.load(processed / "X_train_augmented.npy")
    y_train_aug = np.load(processed / "y_train_augmented.npy")

    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.hstack([y_train, y_val, y_test])

    trial_vae_config = replace(config.vae, latent_dim=args.latent_dim, beta_max=args.beta_max)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        trainer = DVAETrainer(
            vae_config=trial_vae_config,
            mlflow_config=config.mlflow,
            run_name=f"vae_grid_ld{args.latent_dim}_b{args.beta_max:.3f}",
        )
        vae_result = trainer.train(X_all=X_all, y_all=y_all, output_dir=output_dir)

        encoder = LatentEncoder(encoder_path=vae_result.encoder_path, latent_dim=args.latent_dim)
        enc = encoder.encode(
            X_train_augmented=X_train_aug,
            y_train_augmented=y_train_aug,
            X_val=X_val,
            X_test=X_test,
        )

    clf = _fit_gmm(enc.Z_train_augmented, y_train_aug)
    eval_f1,  eval_recall  = _metrics(clf, enc.Z_val,  y_val)
    eout_f1,  eout_recall  = _metrics(clf, enc.Z_test, y_test)
    ein_f1,   _            = _metrics(clf, enc.Z_train_augmented, y_train_aug)
    fitness = 0.6 * eval_f1 + 0.4 * eval_recall

    result = {
        "latent_dim": args.latent_dim,
        "beta_max": args.beta_max,
        "eval_macro_f1": eval_f1,
        "eval_fatal_recall": eval_recall,
        "eout_macro_f1": eout_f1,
        "eout_fatal_recall": eout_recall,
        "ein_macro_f1": ein_f1,
        "fitness": fitness,
        "vae_elbo": vae_result.final_elbo,
        "vae_best_epoch": vae_result.best_epoch,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
