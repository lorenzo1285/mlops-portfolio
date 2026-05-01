# Implementation Plan v2: Pure Generative Classifier via 3-TVAE

## 1. Architectural Objective & Academic Context
Replace the hybrid VAE + XGBoost architecture with a **Likelihood-Based Generative Classifier**. 
* **Methodology:** We will train three independent Variational Autoencoders (`ctgan.TVAE`), one for each crash severity class. Classification is performed by calculating the Negative Log-Likelihood (Reconstruction Loss) for a new sample across all three models and selecting the minimum.
* **Academic Basis:** This utilizes **Generation-based Tabular Data Augmentation at the Row Level** (Xu et al., 2019) using Variational Gaussian Mixture (VGM) preprocessing to handle the multimodal, mixed-type tabular crash data.

## 2. Dependencies
* `ctgan` (`TVAE`)
* `torch`
* `pandas`, `numpy`
* `scipy` / `sklearn` (for synthetic data evaluation)

## 3. Phase 1: Generation-based Tabular Data Augmentation (TDA)
The `Fatal` class is severely imbalanced. We will use a TVAE strictly as a synthetic data generator to augment this class.

**Instructions for the Coding Agent:**
1. Split the raw preprocessed `X_train` into three DataFrames: `df_pdo`, `df_injury`, `df_fatal`.
2. **Augmentation Step:**
   * Initialize: `generator_fatal = TVAE(epochs=500)`
   * Fit on minority: `generator_fatal.fit(df_fatal)`
   * Generate: `synthetic_fatal = generator_fatal.sample(5000)`
3. **Data Quality Gate (Crucial):**
   * Write a helper function to check the *Diversity* and *Consistency* of `synthetic_fatal` against `df_fatal` (e.g., comparing mean/variance of key continuous columns).
4. Concatenate: `df_fatal_augmented = pd.concat([df_fatal, synthetic_fatal])`

## 4. Phase 2: Training the Likelihood Estimators
Train three separate TVAE models intended for classification via reconstruction.

**Instructions for the Coding Agent:**
1. Initialize three models: `clf_pdo`, `clf_injury`, `clf_fatal`.
2. Fit `clf_pdo` on `df_pdo`.
3. Fit `clf_injury` on `df_injury`.
4. Fit `clf_fatal` on `df_fatal_augmented`.
5. Save the three models to the DVC/MLflow registry.

## 5. Phase 3: The Custom Inference Engine (NLL Calculator)
`ctgan` does not expose a native `.reconstruction_loss()` method. We must extract it manually using the internal PyTorch components.

**Instructions for the Coding Agent (Custom Classifier Logic):**
Write a wrapper function `predict_severity(X_new_df)`:
1. **Iterate:** For `model_name, model` in `[('PDO', clf_pdo), ('Injury', clf_injury), ('Fatal', clf_fatal)]`:
2. **Transform:** `transformed_np = model._transformer.transform(X_new_df)`
3. **Tensorize:** Convert `transformed_np` to `torch.Tensor` (ensure dtype is float32 and mapped to the correct device).
4. **Forward Pass:** `recon, mu, logvar = model._model(tensor_data)`
5. **Calculate Loss (ELBO):**
   * `mse_loss = torch.mean((recon - tensor_data) ** 2, dim=1)`
   * `kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)`
   * `total_loss = mse_loss + kl_loss` (or just use MSE if KL is too unstable).
6. **Argmin Selection:** Compare the `total_loss` from all three models. 
7. **Return:** The `model_name` that produced the **lowest** loss.

## 6. Evaluation Metrics
Run the standard test set (`X_test`) through the inference engine:
* **F1-Macro Score:** Ensure the Fatal class is successfully predicted.
* **Latency Profiling:** Measure inference time per batch (Target: < 30s).
