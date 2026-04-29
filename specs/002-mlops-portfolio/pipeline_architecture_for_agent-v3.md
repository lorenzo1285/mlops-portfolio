# Architecture Document: Generative Accident Severity Pipeline (v3 - Denoising & Disentangled)
**Target Audience:** Coding Agent / AI Assistant  
**Primary Goal:** Develop a **Generative Model** that learns the true probability distribution of historical traffic accidents. This pipeline uses a Denoising Variational Autoencoder (DVAE) to handle messy/missing data via "Neural Inpainting" and extracts clean latent features for a final XGBoost classifier.

---

## 🏗️ The A/B Test Strategy
Compare a traditional statistical generative baseline against an advanced deep generative representation learning approach.

### Pipeline A: The Statistical Baseline (ML)
* **Balancing:** SMOTE on raw tabular data.
* **Density Estimation:** `scikit-learn` Gaussian Mixture Model (GMM) (`n_components=2`) in `PyCaret`.
* **Purpose:** Benchmark for inference speed and baseline generative classification.

### Pipeline B: The Hybrid DVAE + XGBoost Architecture (Main Focus)
This pipeline incorporates architectural principles specifically referenced in modern representation learning (e.g., *Arxiv Insights* on VAEs).

#### Phase 1: Feature Engineering & Distribution Mapping (PyTorch Denoising $\beta$-VAE)
* **Task:** Train a Denoising VAE to map the continuous data distribution and intelligently handle missing data.
* **Step 1: The Corruption Step (Neural Inpainting):** * *Implementation:* Before passing data into the Encoder, apply `nn.Dropout(p=0.15)` to the input tabular tensor (or add Gaussian noise to continuous variables). 
  * *Purpose:* This mathematically forces the network to reconstruct the *original clean data* from broken data, teaching it the underlying physics of crashes to fill in missing values natively (Tabular Neural Inpainting).
* **Step 2: Encoder & Bottleneck:** `nn.Linear` layers -> Splits into $\mu$ and $\log(\sigma^2)$. Uses the Reparameterization Trick ($Z = \mu + \epsilon \cdot \sigma$).
* **Step 3: Decoder:** Reconstructs the *uncorrupted* original tabular inputs.
* **Step 4: Disentanglement ($\beta$-VAE Loss):** * Add a $\beta$ hyperparameter multiplier to the Kullback-Leibler (KL) Divergence term in the loss function.
  * *Purpose:* Forces the latent space to be *disentangled*, meaning individual neurons in the latent vector learn uncorrelated, causal features (e.g., isolating "speed" independently from "weather").
* **Phase 1 Evaluation (CRITICAL):** Track the **Evidence Lower Bound (ELBO)** (Reconstruction Loss + $\beta$ * KL Divergence). Halt training and freeze the Encoder when the validation ELBO stabilizes.

#### Phase 2: Generative Upsampling (Balancing the Dataset)
* **Task:** Fix the class imbalance for "Fatal" accidents.
* **Method:** Isolate the latent space coordinates of the real "Fatal" accidents. Sample random mathematical noise from this neighborhood and pass it through the trained **Decoder** to generate brand new, physically logical rows of tabular accident data.

#### Phase 3: Feature Extraction (The Hand-off)
* **Method:** Pass the entire balanced dataset (real + synthetic) through the frozen Encoder to extract the dense, noise-free Latent Vectors ($Z$). The raw CSV features are discarded.

#### Phase 4: The Classifier (XGBoost)
* **Task:** Draw the final decision boundary.
* **Method:** Train an `xgboost.XGBClassifier` directly on the $Z$ vectors.

---

## 📊 Evaluation Metrics (By Phase)
1. **Generative Phase (DVAE Training):** * **ELBO Convergence:** Ensure the reconstruction and KL loss stabilize.
   * **Imputation Sanity Check:** Feed an accident with an artificially deleted "Temperature" column and verify the DVAE's reconstruction outputs a physically logical temperature.
2. **Discriminative Phase (Final A/B Test):** * **F1-Score (Minority Class):** Did the final pipeline successfully identify 'Fatal' accidents?
   * **ROC-AUC:** How well does the model separate the classes?
   * **Inference Latency:** Milliseconds per prediction.
