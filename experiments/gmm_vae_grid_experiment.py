"""VAE x GMM grid feasibility experiment.

Retrains the VAE for each (latent_dim, beta_max) cell via a fresh subprocess,
re-encodes the data, then evaluates GMM with the best hyperparameters found
by gmm_optuna_experiment.py.
Goal: check whether a latent space optimised for GMM fitness produces meaningfully
better class separation than the XGBoost-tuned latent space.

Baseline (from gmm_optuna_experiment.py):
  latent_dim=8, beta_max=0.056 -> eout_macro_f1=0.268, eout_fatal_recall=0.619

Best GMM params (fixed across all grid cells):
  covariance_type=diag, n_components={PDO:5, Injury:3, Fatal:4}
  reg_covar=2.72e-4, fatal_prior_boost=17.66, n_init=12

Each cell is run as a subprocess (_gmm_vae_grid_worker.py) to ensure clean
Windows DLL loading for PyTorch. Results are returned as JSON on stdout.

Run:
    uv run python experiments/gmm_vae_grid_experiment.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mlflow

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.config import load_config

_EXPERIMENT_NAME = "crash-severity-gmm-vae-grid"
_WORKER = str(Path(__file__).parent / "_gmm_vae_grid_worker.py")

_LATENT_DIMS = [8, 16, 32]
_BETA_MAXES  = [0.056, 0.5, 1.0]

_BEST_GMM_PARAMS = {
    "gmm_covariance_type": "diag",
    "gmm_fatal_prior_boost": 17.663,
    "gmm_n_components_pdo": 5,
    "gmm_n_components_injury": 3,
    "gmm_n_components_fatal": 4,
}


def run_cell(latent_dim: int, beta_max: float, tracking_uri: str,
             cell_idx: int, n_cells: int) -> dict | None:
    print(f"\n[{cell_idx}/{n_cells}] latent_dim={latent_dim}  beta_max={beta_max:.4f}")

    proc = subprocess.run(
        [sys.executable, _WORKER,
         "--latent-dim", str(latent_dim),
         "--beta-max", str(beta_max)],
        capture_output=True,
        text=True,
        cwd=str(_project_root),
    )

    # Forward stderr for progress visibility
    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            if line and "FutureWarning" not in line and "deprecat" not in line.lower():
                print(f"  {line}")

    if proc.returncode != 0:
        print(f"  ERROR (exit {proc.returncode})")
        return None

    # Last non-empty line is the JSON result
    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    if not lines:
        print("  ERROR: no output from worker")
        return None

    result = json.loads(lines[-1])

    print(f"  eval_macro_f1={result['eval_macro_f1']:.4f}  "
          f"eval_fatal_recall={result['eval_fatal_recall']:.4f}  "
          f"fitness={result['fitness']:.4f}")
    print(f"  eout_macro_f1={result['eout_macro_f1']:.4f}  "
          f"eout_fatal_recall={result['eout_fatal_recall']:.4f}  "
          f"vae_elbo={result['vae_elbo']:.4f}")

    # Log to MLflow from orchestrator
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"grid_ld{latent_dim}_b{beta_max:.3f}"):
        mlflow.log_params({
            "latent_dim": latent_dim,
            "beta_max": beta_max,
            "vae_best_epoch": result.get("vae_best_epoch"),
            **_BEST_GMM_PARAMS,
        })
        mlflow.log_metrics({
            "fitness": result["fitness"],
            "eval_macro_f1": result["eval_macro_f1"],
            "eval_fatal_recall": result["eval_fatal_recall"],
            "eout_macro_f1": result["eout_macro_f1"],
            "eout_fatal_recall": result["eout_fatal_recall"],
            "ein_macro_f1": result["ein_macro_f1"],
            "vae_final_elbo": result["vae_elbo"],
        })

    return result


def main() -> None:
    config = load_config()

    cells = [(ld, bm) for ld in _LATENT_DIMS for bm in _BETA_MAXES]
    n_cells = len(cells)
    print(f"Grid: {_LATENT_DIMS} x {_BETA_MAXES} = {n_cells} cells")
    print(f"Baseline: latent_dim=8, beta_max=0.056 -> eout_macro_f1=0.268")

    results = []
    for i, (ld, bm) in enumerate(cells, 1):
        row = run_cell(ld, bm, config.mlflow.tracking_uri, i, n_cells)
        if row:
            results.append(row)

    print(f"\n{'='*72}")
    print(f"{'latent_dim':>12} {'beta_max':>10} {'eval_f1':>9} {'eout_f1':>9} "
          f"{'fatal_rec':>10} {'fitness':>9} {'vae_elbo':>10}")
    print(f"{'-'*72}")
    baseline = {"latent_dim": 8, "beta_max": 0.056, "eval_macro_f1": 0.265,
                "eout_macro_f1": 0.268, "eout_fatal_recall": 0.619,
                "fitness": 0.540, "vae_elbo": float("nan")}
    for r in [baseline] + sorted(results, key=lambda x: x["fitness"], reverse=True):
        tag = " <- baseline" if r is baseline else ""
        elbo = f"{r['vae_elbo']:.4f}" if r["vae_elbo"] == r["vae_elbo"] else "   n/a"
        print(
            f"{r['latent_dim']:>12} {r['beta_max']:>10.4f} "
            f"{r['eval_macro_f1']:>9.4f} {r['eout_macro_f1']:>9.4f} "
            f"{r['eout_fatal_recall']:>10.4f} {r['fitness']:>9.4f} {elbo:>10}{tag}"
        )


if __name__ == "__main__":
    main()
