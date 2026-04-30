"""Quick VAE audit — runs the §2/§3/§4/§5 analysis from vae_fatal_representation.ipynb."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import torch
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_vae.vae_trainer import Encoder

PROCESSED = ROOT / 'data' / 'processed'
MODELS    = ROOT / 'models'

X_train     = np.load(PROCESSED / 'X_train.npy')
y_train     = np.load(PROCESSED / 'y_train.npy')
y_val       = np.load(PROCESSED / 'y_val.npy')
y_test      = np.load(PROCESSED / 'y_test.npy')
X_val       = np.load(PROCESSED / 'X_val.npy')
X_test      = np.load(PROCESSED / 'X_test.npy')
X_all       = np.vstack([X_train, X_val, X_test])
y_all       = np.concatenate([y_train, y_val, y_test])

X_train_aug = np.load(PROCESSED / 'X_train_augmented.npy')
y_train_aug = np.load(PROCESSED / 'y_train_augmented.npy')
Z_train_aug = np.load(PROCESSED / 'Z_train_augmented.npy')

enc_ckpt = torch.load(MODELS / 'vae_encoder.pth', weights_only=True)
encoder  = Encoder(enc_ckpt['input_dim'], enc_ckpt['encoder_dims'], enc_ckpt['latent_dim'])
encoder.load_state_dict(enc_ckpt['state_dict'])
encoder.eval()

with torch.no_grad():
    mu, log_var = encoder(torch.tensor(X_all, dtype=torch.float32))

Z_all = mu.numpy()
lv    = log_var.numpy()
latent_dim = Z_all.shape[1]

# §2 — Posterior quality
class_var = np.stack([np.exp(lv[y_all == i]).mean(axis=0) for i in range(3)])
kl_per_sample = (-0.5 * (1 + lv - Z_all**2 - np.exp(lv))).sum(axis=1)
class_kl  = [kl_per_sample[y_all == i].mean() for i in range(3)]
active    = [(class_var[i] < 0.5).sum() for i in range(3)]
collapsed = [(class_var[i] > 0.9).sum() for i in range(3)]

CLASS_NAMES = ['PDO', 'Injury', 'Fatal']
print('=== §2 Posterior quality ===')
for i, name in enumerate(CLASS_NAMES):
    print(f'  {name:<8} mean_σ²={class_var[i].mean():.3f}  '
          f'active={active[i]}/{latent_dim}  collapsed={collapsed[i]}/{latent_dim}  '
          f'mean_KL={class_kl[i]:.4f}')

# §3 — Real vs synthetic Fatal alignment
n_orig       = len(X_train)
y_orig       = y_train_aug[:n_orig]
Z_orig       = Z_train_aug[:n_orig]
Z_synth      = Z_train_aug[n_orig:]
Z_real_fatal = Z_orig[y_orig == 2]
Z_real_pdo   = Z_orig[y_orig == 0]

c_real  = Z_real_fatal.mean(axis=0)
c_synth = Z_synth.mean(axis=0)
c_pdo   = Z_real_pdo.mean(axis=0)
d_rs    = float(np.linalg.norm(c_real - c_synth))
d_rp    = float(np.linalg.norm(c_real - c_pdo))

ks_results = [stats.ks_2samp(Z_real_fatal[:, d], Z_synth[:, d]) for d in range(latent_dim)]
n_ks_aligned = sum(r.statistic < 0.10 for r in ks_results)

print(f'\n=== §3 Real vs Synthetic Fatal ===')
print(f'  Real Fatal      : {len(Z_real_fatal)}')
print(f'  Synthetic Fatal : {len(Z_synth)}')
print(f'  Centroid dist real-synth : {d_rs:.4f}')
print(f'  Centroid dist real-PDO   : {d_rp:.4f}')
print(f'  Ratio (synth/PDO)        : {d_rs/d_rp:.3f}  (<0.5 = good)')
print(f'  KS-aligned dims (KS<0.10): {n_ks_aligned}/{latent_dim}')
for d, r in enumerate(ks_results):
    print(f'    z{d}: KS={r.statistic:.3f}  p={r.pvalue:.3e}')

# §4 — Separability
Z_fatal_all = np.vstack([Z_real_fatal, Z_synth])
f_mean = Z_fatal_all.mean(axis=0)
f_std  = Z_fatal_all.std(axis=0, ddof=1)
p_mean = Z_real_pdo.mean(axis=0)
sep    = np.abs(f_mean - p_mean) / np.maximum(f_std, 1e-6)
lower, upper = f_mean - 2*f_std, f_mean + 2*f_std
overlap = np.all((Z_real_pdo >= lower) & (Z_real_pdo <= upper), axis=1).mean()

print(f'\n=== §4 Fatal separability ===')
print(f'  Dims with sep>1σ : {(sep>1).sum()}/{latent_dim}')
print(f'  PDO overlap      : {overlap:.2%}  (<10% = good)')
for d in range(latent_dim):
    print(f'    z{d}: sep={sep[d]:.3f}')

# §5 — Verdict
n_active_fatal = int((class_var[2] < 0.5).sum())
ratio          = d_rs / d_rp
pct_ks         = n_ks_aligned / latent_dim
n_sep          = int((sep > 1).sum())

print(f'\n=== §5 Verdict ===')
rows = [
    ('Fatal active dims (σ²<0.5)',       f'{n_active_fatal}/{latent_dim}',  n_active_fatal >= 3, n_active_fatal >= 1),
    ('Fatal KL > PDO KL',                str(class_kl[2] > class_kl[0]),   class_kl[2] > class_kl[0], True),
    ('Centroid ratio (synth/PDO)',        f'{ratio:.3f}',                   ratio <= 0.5,  ratio <= 1.0),
    ('KS-aligned dims (KS<0.10)',         f'{n_ks_aligned}/{latent_dim}',   pct_ks >= 0.5, pct_ks >= 0.25),
    ('Fatal-PDO sep >1σ dims',           f'{n_sep}/{latent_dim}',           n_sep >= 3,    n_sep >= 1),
    ('PDO overlap in Fatal 2σ box',      f'{overlap:.2%}',                  overlap <= 0.1, overlap <= 0.3),
]
for name, val, good, ok in rows:
    status = '✅' if good else ('⚠️ ' if ok else '❌')
    print(f'  {status} {name:<40} {val}')
