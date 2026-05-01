# VAE Crash Severity: EDA Findings & Tuning Recommendations

## 1. Executive Summary
This document serves as context for adjusting the architecture and training loop of the `crash-severity-vae`. Current evaluations indicate that while the model trains stably, it is suffering from **severe posterior collapse**. The VAE is bypassing the intended dimensionality reduction by routing almost all information through a single latent dimension, resulting in blurry reconstructions, high MAPE (73.01%), and poor separability of crash severity classes.

## 2. Key Findings (Referencing `vae_eda_v2.ipynb`)

### A. Severe Posterior Collapse (Ref: Cells 7 & 10)
* **Finding:** 31 out of 32 latent dimensions are entirely collapsed ($\sigma^2 > 0.9$). 
* **Detail:** Only 1 dimension (`z30`) is actively encoding information. The KL loss (0.77 nats) is almost entirely attributed to this single dimension. 
* **Impact:** The bottleneck is far too restrictive in practice, despite having an overly large theoretical capacity (`latent_dim=32` for 26 input features).

### B. Poor Latent Separation of Target Classes (Ref: Cell 13)
* **Finding:** Fatal vs. PDO (Property Damage Only) crashes fail to separate meaningfully in the latent space.
* **Detail:** The average normalized separation is **0.462** (values $>1$ indicate distinct clusters).
* **Impact:** Because 97% of the latent dimensions are collapsed, the model does not have the "expressive space" to map out the subtle conditional factors that differentiate a minor fender-bender from a fatal crash.

### C. Suboptimal Feature Reconstruction (Ref: Cell 16)
* **Finding:** The model exhibits "mean-seeking" behavior to minimize loss, leading to an overall mean MAPE of **73.01%**.
* **Detail:** The worst reconstructed feature is `MONTH` (MSE = 15.93). 
* **Impact:** Treating cyclical temporal data (like months) as a raw continuous integer creates artificial boundaries (e.g., December "12" is mathematically far from January "1"), drastically increasing reconstruction error.

### D. Premature Convergence (Ref: Weights & Biases metrics)
* **Finding:** The model hit its `best_epoch` at 32 out of 200 epochs.
* **Impact:** The model is diving into a local minimum (the collapsed state) too quickly, settling for "average" predictions rather than learning the granular feature distributions.

---

## 3. Actionable Coding Recommendations

When refactoring the model code, implement the following changes:

### Recommendation 1: Fix the Posterior Collapse
1. **Implement KL Annealing:** Do not start with `beta = 1.0`. Modify the training loop to start with `beta = 0.0` for the first 10-15 epochs, and linearly anneal it up to a maximum of `0.1` or `0.5`. 
    * *Goal:* Allow the model to learn meaningful reconstructions *before* the KL penalty forces the dimensions into a standard normal distribution.
2. **Reduce Latent Dimensionality:** Change `latent_dim` from `32` to **`6` or `8`**.
    * *Goal:* Force the network to utilize the available dimensions efficiently rather than ignoring them.

### Recommendation 2: Address High Reconstruction Error & `MONTH` MSE
1. **Cyclical Encoding:** Update the data preprocessing pipeline. Apply a sine/cosine transformation to the `MONTH` column (and any other cyclical time variables like `HOUR` or `DAY_OF_WEEK`).
    * *Implementation Note:* Create two new features: `MONTH_sin = sin(2 * pi * MONTH / 12)` and `MONTH_cos = cos(2 * pi * MONTH / 12)`. Drop the original `MONTH` column.

### Recommendation 3: Stabilize the Training Dynamics
1. **Lower the Learning Rate:** Decrease the optimizer learning rate from `0.001` to **`0.0005`** or **`0.0001`**.
    * *Goal:* Prevent the model from rushing into the collapsed local minimum and give the KL annealing time to take effect.
