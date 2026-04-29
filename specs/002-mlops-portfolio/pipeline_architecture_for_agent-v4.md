# Architecture Document: Generative Accident Severity Pipeline (v4 - Hybrid vs Pure DL)
**Target Audience:** Coding Agent / AI Assistant  
**Primary Goal:** Develop a Generative Model using a Denoising $\beta$-VAE to learn the probability distribution of traffic accidents. The overarching A/B test will evaluate whether a traditional ML classifier (XGBoost) or a deep Neural Network (MLP) performs better when classifying the extracted latent representations.

---

## 🏗️ Phase 1: Shared Representation Learning (The Physics Engine)
Both pipelines begin by sharing the exact same deep generative foundation.

* **Task:** Train a Denoising $\beta$-VAE purely on normalized historical tabular features to learn the continuous data distribution and intelligently handle missing data (`NaN`s).
* **Step 1 (Neural Inpainting):** Apply `nn.Dropout(p=0.15)` to the input tabular tensor to force reconstruction of clean data from corrupted data.
* **Step 2 (Encoder):** Compress to a disentangled latent bottleneck using the Reparameterization Trick ($Z = \mu + \epsilon \cdot \sigma$).
* **Step 3 (Loss Function):** Track the Evidence Lower Bound (ELBO) -> (Reconstruction Loss + $\beta \times$ KL Divergence).
* **Step 4 (Generative Upsampling):** Sample noise from the "Fatal" accident latent neighborhood and pass it through the Decoder to generate synthetic minority tabular data.
* **Step 5 (Feature Extraction):** Freeze the VAE and map all balanced data (real + synthetic) into clean, disentangled Latent Vectors ($Z$).

---

## ⚔️ Phase 2: The A/B Classification Test (The Judges)
Once the dataset is balanced and converted entirely into latent $Z$ vectors, the pipeline splits to compare two distinct classification paradigms.

### Pipeline A: The Hybrid Approach (DVAE + XGBoost)
* **Task:** Draw rigid, rule-based decision boundaries through the continuous latent space.
* **Method:** Train an `xgboost.XGBClassifier` directly on the frozen $Z$ vectors.
* **Why it might win:** Tree-based models are notoriously resistant to overfitting synthetic generated data and draw highly effective box-like boundaries.

### Pipeline B: The Pure Deep Learning Approach (DVAE + PyTorch MLP)
* **Task:** Draw smooth, non-linear mathematical boundaries through the continuous latent space.
* **Method:** Attach a Multi-Layer Perceptron (MLP) (e.g., `nn.Linear -> nn.ReLU -> nn.Dropout -> nn.Linear`) directly to the VAE's latent space output. Train it using `BCEWithLogitsLoss`.
* **Why it might win (The Superpower):** End-to-End Fine-Tuning. If the MLP struggles, the agent can unfreeze the VAE Encoder and train the *entire* architecture simultaneously, allowing the classifier to subtly reshape the latent space to make classification easier.

---

## 📊 Evaluation Metrics
1. **Generative Phase (DVAE Training):** * **ELBO Convergence:** Ensure reconstruction and KL loss stabilize.
2. **Discriminative Phase (Final A/B Test on $Z$):** * **F1-Score (Minority Class):** Which classifier (XGBoost vs MLP) successfully identifies the highest percentage of rare 'Fatal' accidents without throwing false positives?
   * **ROC-AUC:** How well does the model separate the classes across probability thresholds?
   * **Inference Latency:** Milliseconds per prediction (XGBoost vs PyTorch forward pass).
