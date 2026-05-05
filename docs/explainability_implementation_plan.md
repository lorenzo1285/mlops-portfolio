# Explainability Implementation Plan: SHAP, LIME, and Counterfactuals

This document outlines the strategy for integrating explainability frameworks into the MLOps portfolio. Given the critical nature of crash severity prediction (especially "Fatal" crashes), interpretability is essential for model validation, bias detection, and stakeholder trust.

## 1. Objectives
- **Feature Attribution**: Identify which features (or latent dimensions $Z$) drive predictions.
- **Local Explanations**: Explain specific "Fatal" predictions to understand risk factors.
- **Global Interpretability**: Summarize model behavior across the entire dataset.
- **Latent Mapping**: Link abstract VAE latent dimensions back to physical crash characteristics.
- **Counterfactuals**: Understand the "tipping point" between an Injury and a Fatal crash.

## 2. Framework Selection

### A. SHAP (SHapley Additive exPlanations)
- **Use Case**: Primary tool for Global and Local attribution.
- **Why**: `XGBClassifier` (currently in `src/train_ml`) supports **TreeSHAP**, which is mathematically robust, consistent, and computationally efficient. It provides a "fair" distribution of feature importance.
- **Integration**: Log SHAP summary plots to MLflow to track model behavior across versions.

### B. LIME (Local Interpretable Model-agnostic Explanations)
- **Use Case**: "Surgical" deep-dives into specific individual predictions.
- **Why**: While SHAP is more theoretically sound for global behavior, LIME is often faster for explaining a *single* complex prediction from any model (including the DL MLP). It creates a local surrogate model that is very intuitive for "what-if" style local reasoning.
- **Synergy**: We will use SHAP for the "Big Picture" and LIME for a "Second Opinion" on critical Fatal false negatives.

### C. Partial Dependence Plots (PDP) & SHAP Dependence Plots
- **Use Case**: Visualizing the "shape" of the relationship between features and risk.
- **Why**: These plots show how the probability of a Fatal crash changes as a specific feature (like speed or hour of day) increases. SHAP Dependence plots also reveal **interactions** (e.g., how speed impact changes during the night vs. day).

### D. Global Surrogate Models
- **Use Case**: Providing a simplified, human-readable "proxy" of the complex model.
- **Why**: We can train a shallow Decision Tree to approximate the XGBoost/MLP behavior. This gives us a set of "if-then" rules that summarize the model's core logic for non-technical stakeholders.

## 3. Proposed Architecture

### Stage 1: `src/explain` (DVC Stage)
A new pipeline stage will be added after `register`.
- **Inputs**: 
    - Champion Model (XGBoost/MLP).
    - Latent Vectors (`Z_test.npy`).
    - Original Features (`X_test.csv`).
- **Outputs**:
    - `shap_values.npy`
    - `global_importance.png` (logged to MLflow).
    - `latent_meaning_report.json` (Correlation between $Z$ and $X$).

### Stage 2: Mapping $Z \rightarrow X$
Since the ML models see $Z$, the explanation will initially be in terms of "Latent Dimension 1, 2, ...". To make this human-readable, we will:
1. Compute the SHAP values for the classifier on $Z$.
2. Compute the Jacobian of the VAE Decoder to map $Z$ gradients back to $X$ features.
3. Combine them to provide explanations in terms of original features (e.g., "Weather", "Speed Limit").

## 4. Implementation Steps

1. **Task T130**: Create `src/explain/explainer.py` using `shap.TreeExplainer` for XGBoost.
2. **Task T131**: Implement a "Latent Decoder" that correlates $Z$ dimensions with raw feature columns to give "names" to the latent space.
3. **Task T132**: Update `dvc.yaml` to include the `explain` stage.
4. **Task T133**: Add SHAP summary and dependence plots to the MLflow run artifacts.
5. **Task T134**: (Advanced) Implement generative counterfactuals by perturbing $Z$ and passing through the VAE Decoder.

## 5. Success Criteria
- [ ] MLflow contains a "Global Feature Importance" plot for every Champion model.
- [ ] A JSON artifact maps the top 3 Latent Dimensions to the most correlated physical features.
- [ ] Local explanations for 10 random "Fatal" predictions are generated and stored.

---
*Note: This plan aligns with the "Superpower" of the VAE architecture, leveraging the generative nature of the latent space for deeper interpretability than standard supervised models.*
