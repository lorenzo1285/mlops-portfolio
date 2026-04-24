# Feature Specification: Crash Severity Prediction Model

**Feature Branch**: `001-crash-severity-model`
**Created**: 2026-03-30
**Status**: Draft
**Input**: Develop a ML/DL model to predict crash severity (CRASHSEVER) based on the CGR Crash Data dataset (74,309 rows, 142 columns, Grand Rapids Michigan 2008-2017).

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Baseline ML Model (Priority: P1)

A data scientist trains a baseline machine learning model on the preprocessed crash dataset and evaluates its ability to predict crash severity. The output is a scored model with metrics documented, serving as the performance benchmark for all subsequent models.

**Why this priority**: Without a working baseline there is no reference point to judge whether more complex models add value. This story alone delivers a usable prediction artifact.

**Independent Test**: Can be tested end-to-end by running the preprocessing and training pipeline on the crash CSV and verifying that evaluation metrics are produced on a held-out test set.

**Acceptance Scenarios**:

1. **Given** the raw crash CSV, **When** the preprocessing pipeline runs, **Then** a clean feature matrix and encoded target are produced with no data leakage between train and test splits.
2. **Given** the clean feature matrix, **When** the baseline model is trained, **Then** macro F1, per-class precision/recall, and a confusion matrix are reported on the test set.
3. **Given** a trained baseline model, **When** a new record is passed in, **Then** a severity prediction and confidence score are returned.

---

### User Story 2 - Model Comparison & Selection (Priority: P2)

A data scientist trains multiple ML and DL model candidates, compares them against the baseline using identical train/test splits, and selects the best performer for further use.

**Why this priority**: The baseline alone may not achieve sufficient predictive accuracy. Comparing alternatives — including at least one deep learning model — ensures the best approach is identified before the work is considered complete.

**Independent Test**: Can be tested by verifying that at least three model types are evaluated under identical conditions and that a comparison table of metrics is produced.

**Acceptance Scenarios**:

1. **Given** trained baseline and candidate models, **When** evaluated on the same held-out test set, **Then** a comparison table of macro F1, weighted F1, and AUC-ROC is produced.
2. **Given** a comparison table, **When** reviewed, **Then** the best model is clearly identified and its selection is justified by metrics.
3. **Given** a class-imbalanced target, **When** any model is trained, **Then** class imbalance is explicitly addressed (via oversampling, class weights, or equivalent) and the strategy is documented.

---

### User Story 3 - Feature Importance & Interpretability (Priority: P3)

A data scientist produces a ranked feature importance report for the best model, identifying which factors most strongly predict crash severity.

**Why this priority**: Interpretability is required to validate that the model is learning meaningful signals rather than artefacts, and to communicate findings to non-technical stakeholders.

**Independent Test**: Can be tested by verifying that a ranked list of top features with importance scores is saved and matches domain expectations from the EDA (e.g. speed limit, hazardous actions, pedestrian involvement rank highly).

**Acceptance Scenarios**:

1. **Given** a trained best model, **When** feature importance is extracted, **Then** the top 20 features are ranked and saved with their importance scores.
2. **Given** the importance report, **When** compared against EDA findings in `docs/eda_findings.md`, **Then** at least 5 of the top 10 features are consistent with EDA-identified high-risk factors.

---

### Edge Cases

- What happens when a feature column is entirely null for a subset of records?
- How does the model handle the Fatal class given its extreme rarity (0.14% of records)?
- What happens if a new record contains a categorical value not seen during training?
- How are records where driver age is unknown (sentinel value 999) treated at inference time?

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline MUST split the dataset into train and test sets before any preprocessing is fit, ensuring no data leakage.
- **FR-002**: The pipeline MUST encode all categorical features and handle missing values without dropping more than 5% of records.
- **FR-003**: The pipeline MUST address class imbalance in the training set; the chosen strategy MUST be documented.
- **FR-004**: The system MUST train a baseline ML model and report macro F1, weighted F1, per-class precision/recall, and a confusion matrix.
- **FR-005**: The system MUST train at least one deep learning model on the tabular data and evaluate it under the same conditions as the baseline.
- **FR-006**: All experiments MUST be tracked with model name, hyperparameters, and evaluation metrics.
- **FR-007**: Trained model artifacts MUST be saved so predictions can be reproduced without retraining.
- **FR-008**: The system MUST produce a feature importance ranking for the best-performing model.
- **FR-011**: Post-crash outcome columns (NUMOFINJ, NUMOFKILL, CRASHTYPE, vehicle damage extent) MUST be used for evaluation enrichment only — to validate that model predictions align with actual outcomes, support error analysis on false negatives, and cross-tabulate confidence scores against injury counts. They MUST NOT be used as model inputs.
- **FR-009**: The target variable MUST be binary — **no-injury** (Property Damage Only) vs **injury+fatal** (Injury + Fatal combined). Class imbalance (~82/18 split) MUST be addressed via class weights. This formulation was chosen over 3-class to avoid the near-impossible Fatal class prediction problem (0.14% of records).
- **FR-010**: Feature selection MUST use only pre-crash observable features — those knowable at or before the moment of the crash. Permitted feature groups: time/date (hour, day of week, month, year), weather and road conditions, lighting, speed limit, driver demographics (age, sex), vehicle type and use, road geometry (number of lanes, width, road class). Post-crash outcome columns (NUMOFINJ, NUMOFKILL, GRTINJSEVE, vehicle damage extent, harm events, violator flags) MUST be excluded to prevent data leakage.

### Key Entities

- **CrashRecord**: A single crash event with all available attributes; the unit of prediction.
- **SeverityLabel**: The encoded target derived from CRASHSEVER; represents the outcome to be predicted.
- **PreprocessingPipeline**: The sequence of transformations (imputation, encoding, scaling, resampling) applied to raw records before modelling.
- **TrainedModel**: A serialised model artifact paired with its preprocessing pipeline and metadata (metrics, feature list, training date).
- **ExperimentLog**: A record of a single training run including model type, hyperparameters, train/test metrics, and artifact path.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The best model achieves a macro F1 score above 0.55 on the held-out test set.
- **SC-002**: The deep learning model is evaluated and its results are directly comparable to the ML baseline (same split, same metrics).
- **SC-003**: Class imbalance is demonstrably addressed — the minority class recall is above 0.40 for the best model.
- **SC-004**: The top 20 predictive features are documented and at least 5 align with high-risk factors identified in the EDA.
- **SC-005**: All experiments are logged with sufficient detail to reproduce any result without retraining.
- **SC-006**: The full pipeline from raw CSV to evaluation metrics runs without manual intervention.
- **SC-007**: Predicted high-severity crashes show a measurably higher average injury count than predicted low-severity crashes, confirming the model captures real signal.

---

## Assumptions

- The EDA in `notebooks/eda.ipynb` and `docs/eda_findings.md` is complete and its feature rankings will guide initial feature selection.
- Real-time serving or deployment to production is out of scope for this phase; the deliverable is a trained, saved model artifact.
- The notebook-based workflow established during EDA will be extended with new notebooks for preprocessing, training, and evaluation.
- No additional external data sources will be introduced; the model will use only the CGR Crash Data CSV.
- The `GRTINJSEVE` column (greatest injury severity) is considered a direct proxy for CRASHSEVER and will be excluded from features to avoid leakage.
- scikit-learn and standard data science libraries are already installed in the project environment.
- Experiment tracking will use the `experiment_tracker.py` script already available in `.agents/skills/data-scientist/scripts/`.
