# Tasks: Crash Severity Prediction Model

**Input**: Design documents from `specs/001-crash-severity-model/`
**Prerequisites**: spec.md ✅ | plan.md ✅ | EDA complete ✅ (`notebooks/eda.ipynb`, `docs/eda_findings.md`)
**Note**: EDA is already done — tasks begin at preprocessing (notebook 02).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story this task belongs to (US1, US2, US3)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create directories and verify environment before notebook work begins.

- [ ] T001 Create `notebooks/` directory if not present and confirm `models/` directory exists at repo root
- [ ] T002 [P] Verify all required packages are importable: pandas, numpy, sklearn, pycaret, torch, mlflow (run `uv run python -c "import pandas, sklearn, pycaret, torch, mlflow; print('OK')"`)
- [ ] T003 [P] Initialise MLflow tracking: confirm `mlruns/` is created by running `uv run python -c "import mlflow; mlflow.set_experiment('smoke-test')"` from repo root

**Checkpoint**: Environment confirmed — notebook work can begin.

---

## Phase 2: Foundational — Preprocessing Pipeline (Blocking)

**Purpose**: Produce the shared train/test splits and fitted pipeline that ALL subsequent notebooks depend on.

**⚠️ CRITICAL**: Notebooks 03, 04, and 05 cannot start until this phase is complete.

- [ ] T004 Create `notebooks/02_preprocessing.ipynb` with markdown header cell documenting inputs (raw CSV path `data/CGR_Crash_Data.csv`) and outputs (`models/preprocessing_pipeline.joblib`, numpy arrays)
- [ ] T005 Load raw CSV, select pre-crash feature columns per FR-010: HOUR, DAYOFWEEK, MONTH, YEAR, WEATHER, SURFCOND, LIGHTING, SPEEDLIMIT, RDNUMLANES, RDWIDTH, ROUTECLASS, TRUNKLINE, RDSUBTYPE, DRIVER1AGE, DRIVER1SEX, DRIVER2AGE, DRIVER2SEX, VEH1TYPE, VEH1USE, VEH2TYPE, VEH2USE, CRASHTYPE, TRAFCTLDEV, NONTRAFFIC — in `notebooks/02_preprocessing.ipynb`
- [ ] T006 Recode sentinel value 999 → NaN in DRIVER1AGE and DRIVER2AGE columns in `notebooks/02_preprocessing.ipynb`
- [ ] T007 Encode target: CRASHSEVER → SEVERITY_BINARY (0 = "Property Damage Only", 1 = "Injury" + "Fatal") and print class distribution in `notebooks/02_preprocessing.ipynb`
- [ ] T008 Perform stratified 80/20 train/test split (`random_state=42`, `stratify=y`) — confirm no leakage: pipeline fit ONLY on train in `notebooks/02_preprocessing.ipynb`
- [ ] T009 Build sklearn `Pipeline` with: `SimpleImputer(strategy="median")` for numerics, `SimpleImputer(strategy="most_frequent")` + `OrdinalEncoder(handle_unknown="use_encoded_value")` for categoricals, `StandardScaler` for numeric output — in `notebooks/02_preprocessing.ipynb`
- [ ] T010 Fit pipeline on `X_train` only, transform both `X_train` and `X_test`; print shape and confirm < 5% rows dropped (FR-002) in `notebooks/02_preprocessing.ipynb`
- [ ] T011 Save fitted pipeline to `models/preprocessing_pipeline.joblib` using `joblib.dump()` and print `feature_names` list (post-encoding column names) in `notebooks/02_preprocessing.ipynb`

**Checkpoint**: `models/preprocessing_pipeline.joblib` exists; `X_train`, `X_test`, `y_train`, `y_test` arrays ready. All downstream notebooks can now begin.

---

## Phase 3: User Story 1 — Baseline ML Model (Priority: P1) 🎯 MVP

**Goal**: Train and evaluate a baseline ML model using PyCaret; produce a saved artifact with documented metrics.

**Independent Test**: Run notebook end-to-end and confirm `models/best_ml_model.pkl` exists and macro F1, confusion matrix, and AUC-ROC are printed on the held-out test set.

### Implementation for User Story 1

- [ ] T012 [US1] Create `notebooks/03_ml_pycaret.ipynb` with markdown header documenting inputs (pipeline joblib, train arrays) and outputs (`models/best_ml_model.pkl`, MLflow experiment `crash-severity-ml`)
- [ ] T013 [US1] Load `models/preprocessing_pipeline.joblib` and reconstruct `train_df` (pandas DataFrame with `SEVERITY_BINARY` target column) from `X_train` + `y_train` in `notebooks/03_ml_pycaret.ipynb`
- [ ] T014 [US1] Configure MLflow: `mlflow.set_experiment("crash-severity-ml")` + `mlflow.sklearn.autolog()` in `notebooks/03_ml_pycaret.ipynb`
- [ ] T015 [US1] Call PyCaret `setup()` with: `data=train_df`, `target="SEVERITY_BINARY"`, `train_size=0.8`, `fold=5`, `fold_strategy="stratifiedkfold"`, `class_weights={0: 0.61, 1: 2.74}`, `metric="F1"`, `session_id=42`, `log_experiment=True`, `experiment_name="crash-severity-ml"` in `notebooks/03_ml_pycaret.ipynb`
- [ ] T016 [US1] Run `compare_models(n_select=3, sort="F1")` and display results table in `notebooks/03_ml_pycaret.ipynb`
- [ ] T017 [US1] Run `tune_model(best_models[0])` on the top model in `notebooks/03_ml_pycaret.ipynb`
- [ ] T018 [US1] Log Ein/Eout/generalisation gap for all 3 top models via nested MLflow runs (macro F1 on `X_train` and `X_test`) in `notebooks/03_ml_pycaret.ipynb`
- [ ] T019 [US1] Evaluate tuned model on `X_test` / `y_test`: print macro F1, weighted F1, per-class precision/recall, confusion matrix, AUC-ROC in `notebooks/03_ml_pycaret.ipynb`
- [ ] T020 [US1] Save tuned model with `save_model(tuned, "models/best_ml_model")` in `notebooks/03_ml_pycaret.ipynb`

**Checkpoint**: US1 complete — baseline ML model saved, metrics reported, MLflow experiment `crash-severity-ml` populated. Verify macro F1 > 0.55 (SC-001) and minority class recall > 0.40 (SC-003).

---

## Phase 4: User Story 2 — Model Comparison & Selection (Priority: P2)

**Goal**: Train a shallow PyTorch MLP, compare it against the ML baseline on the same test split, and identify the best overall model.

**Independent Test**: Run notebook 04 end-to-end; confirm `models/mlp_model.pth` exists and epoch-level learning curves (ein_loss, eout_loss) are logged in MLflow experiment `crash-severity-dl`. Run notebook 05 and confirm a comparison table is produced.

### Implementation for User Story 2 — PyTorch MLP

- [ ] T021 [US2] Create `notebooks/04_dl_pytorch.ipynb` with markdown header documenting inputs (pipeline joblib, numpy arrays) and outputs (`models/mlp_model.pth`, MLflow experiment `crash-severity-dl`)
- [ ] T022 [US2] Define `ShallowMLP` class in `notebooks/04_dl_pytorch.ipynb`: Input(d) → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(64) → BatchNorm1d → ReLU → Dropout(0.3) → Linear(1); d = number of features from preprocessing
- [ ] T023 [US2] Convert numpy arrays to PyTorch tensors and create `TensorDataset` + `DataLoader` for train (batch_size=256, shuffle=True), val (20% of train, batch_size=512), and test sets in `notebooks/04_dl_pytorch.ipynb`
- [ ] T024 [US2] Instantiate loss: `criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.74]))` and `eval_criterion = nn.BCEWithLogitsLoss()` in `notebooks/04_dl_pytorch.ipynb`
- [ ] T025 [US2] Implement `train_one_epoch(model, optimizer, loader, criterion)` function in `notebooks/04_dl_pytorch.ipynb`
- [ ] T026 [US2] Implement `evaluate(model, X, y, criterion)` returning `(loss, macro_f1)` using threshold 0.5 on sigmoid output in `notebooks/04_dl_pytorch.ipynb`
- [ ] T027 [US2] Implement `EarlyStopping` class with `patience=10`, monitoring validation loss in `notebooks/04_dl_pytorch.ipynb`
- [ ] T028 [US2] Set up MLflow run: `mlflow.set_experiment("crash-severity-dl")`, log all hyperparameters (architecture, dropout, lr, weight_decay, pos_weight, loss_fn, early_stopping_patience) in `notebooks/04_dl_pytorch.ipynb`
- [ ] T029 [US2] Write training loop: for each epoch call train_one_epoch → evaluate ein → evaluate eout → log `{ein_loss, ein_f1, eout_loss, eout_f1, gap_f1}` with `step=epoch` → check early stopping in `notebooks/04_dl_pytorch.ipynb`
- [ ] T030 [US2] After training: evaluate on `X_test` / `y_test`, log `test_macro_f1` and `test_recall_class1`; save `mlp_model.pth` with `torch.save(model.state_dict(), "models/mlp_model.pth")` in `notebooks/04_dl_pytorch.ipynb`
- [ ] T031 [US2] Log model to MLflow registry: `mlflow.pytorch.log_model(model, "mlp_model", registered_model_name="crash-severity-mlp")` in `notebooks/04_dl_pytorch.ipynb`

### Implementation for User Story 2 — Evaluation & Comparison

- [ ] T032 [US2] Create `notebooks/05_evaluation.ipynb`; use `MlflowClient` to pull best run from `crash-severity-ml` (by `eout_macro_f1`) and best run from `crash-severity-dl` (by `test_macro_f1`) in `notebooks/05_evaluation.ipynb`
- [ ] T033 [US2] Build side-by-side comparison table: model name, macro F1, weighted F1, AUC-ROC, minority class recall, generalisation gap in `notebooks/05_evaluation.ipynb`
- [ ] T034 [US2] Load best overall model artifact and run enrichment analysis: compute `mean(NUMOFINJ | ŷ=0)` vs `mean(NUMOFINJ | ŷ=1)` using post-crash columns for validation only (SC-007, FR-011) in `notebooks/05_evaluation.ipynb`

**Checkpoint**: US2 complete — ML vs DL comparison table produced, best model identified, enrichment analysis confirms SC-007.

---

## Phase 5: User Story 3 — Feature Importance & Interpretability (Priority: P3)

**Goal**: Extract and rank the top 20 predictive features from the best model; cross-validate against EDA findings.

**Independent Test**: Confirm `docs/feature_importance.csv` exists with 20 rows and at least 5 of the top 10 features match high-risk factors in `docs/eda_findings.md` (SC-004).

### Implementation for User Story 3

- [ ] T035 [US3] In `notebooks/05_evaluation.ipynb`: load best ML model artifact from MLflow; extract feature importances (use `feature_importances_` for tree models, `coef_` for LR, permutation importance as fallback)
- [ ] T036 [US3] Rank top 20 features by importance score; create horizontal bar chart saved as `docs/feature_importance.png` and logged as `mlflow.log_artifact()` in `notebooks/05_evaluation.ipynb`
- [ ] T037 [US3] Save ranked feature list to `docs/feature_importance.csv` with columns: `feature`, `importance_score`, `rank` in `notebooks/05_evaluation.ipynb`
- [ ] T038 [US3] Cross-reference top 10 features against `docs/eda_findings.md` high-risk factors; print alignment count (must be ≥ 5 to pass SC-004) in `notebooks/05_evaluation.ipynb`
- [ ] T039 [P] [US3] Generate and save confusion matrix heatmap for best model as `docs/confusion_matrix.png` in `notebooks/05_evaluation.ipynb`
- [ ] T040 [P] [US3] Generate and save AUC-ROC curve for best model as `docs/roc_curve.png` in `notebooks/05_evaluation.ipynb`

**Checkpoint**: All user stories complete. Feature importance documented, SC-004 verified.

---

## Phase 6: Polish & Success Criteria Validation

**Purpose**: Final threshold checks and documentation to confirm all spec success criteria are met.

- [ ] T041 Assert macro F1 > 0.55 on held-out test set for best model (SC-001); print PASS/FAIL
- [ ] T042 Assert minority class (injury+fatal) recall > 0.40 for best model (SC-003); print PASS/FAIL
- [ ] T043 Assert mean NUMOFINJ for predicted class 1 > mean NUMOFINJ for predicted class 0 (SC-007); print ratio and PASS/FAIL
- [ ] T044 [P] Update `docs/eda_findings.md` with a final "Model Results" section summarising best model, key metrics, and top 5 features
- [ ] T045 [P] Confirm all experiment runs are visible in MLflow UI (`uv run mlflow ui`); verify `crash-severity-ml` and `crash-severity-dl` experiments both have runs with Ein/Eout metrics (SC-005)

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
    └── Phase 2 (Preprocessing) — BLOCKS all story phases
            ├── Phase 3 (US1: ML Baseline)      ← start here for MVP
            ├── Phase 4 (US2: DL + Comparison)  ← depends on Phase 3 for 05_evaluation
            └── Phase 5 (US3: Interpretability) ← depends on Phase 4 (best model identified)
                        └── Phase 6 (Polish & Validation)
```

### Notebook Execution Order

| Order | Notebook | Depends On | Outputs |
|---|---|---|---|
| 1 | `02_preprocessing.ipynb` | raw CSV | `preprocessing_pipeline.joblib`, arrays |
| 2 | `03_ml_pycaret.ipynb` | notebook 02 | `best_ml_model.pkl`, MLflow `crash-severity-ml` |
| 3 | `04_dl_pytorch.ipynb` | notebook 02 | `mlp_model.pth`, MLflow `crash-severity-dl` |
| 4 | `05_evaluation.ipynb` | notebooks 03 + 04 | comparison table, feature importance, enrichment |

### Parallel Opportunities

- T002 and T003 (Setup) can run in parallel
- T039 and T040 (confusion matrix + ROC curve) can run in parallel
- T044 and T045 (Polish) can run in parallel
- Notebooks 03 and 04 can run in parallel once notebook 02 is complete

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup)
2. Complete Phase 2 (Preprocessing) — critical blocker
3. Complete Phase 3 (US1: ML Baseline)
4. **STOP and VALIDATE**: confirm `models/best_ml_model.pkl` exists, macro F1 reported
5. This alone satisfies FR-001 through FR-004 and SC-006

### Incremental Delivery

1. Phase 1 + 2 → preprocessing pipeline ready
2. Phase 3 → baseline ML model, MLflow experiment populated (MVP)
3. Phase 4 → DL model + comparison table (SC-002 met)
4. Phase 5 → feature importance report (SC-004 met)
5. Phase 6 → all success criteria verified

---

## Notes

- Never fit the preprocessing pipeline on test data (FR-001)
- `pos_weight=2.74` in BCEWithLogitsLoss is the minority class weight, not the class ratio
- PyCaret's `compare_models()` will take several minutes — expected behaviour
- Early stopping patience=10 means training halts if val loss doesn't improve for 10 consecutive epochs
- MLflow tracking store auto-creates `mlruns/` at repo root on first run
