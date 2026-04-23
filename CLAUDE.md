# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data analysis interview challenge. The task is defined in `docs/Data_analysis_interview_challange.docx` and the primary dataset is `data/CGR_Crash_Data.csv` (~108MB crash data).

## Commands

```bash
# Run a script
uv run python <script.py>

# Add a dependency
uv add <package>

# Run notebooks (execute in order: 02 → 03 → 04 → 05)
uv run jupyter notebook notebooks/02_preprocessing.ipynb
uv run jupyter notebook notebooks/03_ml_pycaret.ipynb
uv run jupyter notebook notebooks/04_dl_pytorch.ipynb
uv run jupyter notebook notebooks/05_evaluation.ipynb

# MLflow experiment tracking UI
uv run mlflow ui   # open http://localhost:5000

# Data scientist skill scripts (hypothesis testing only — feature selection and experiment tracking use sklearn/MLflow directly)
python .agents/skills/data-scientist/scripts/hypothesis_tester.py --help

# Airflow workflow orchestration (learning environment)
cd airflow
.\setup.ps1                    # one-time setup
uv run airflow standalone      # start Airflow (http://localhost:8080)
```

## Architecture

- `data/CGR_Crash_Data.csv` — primary dataset (74,309 rows, 142 cols)
- `notebooks/eda.ipynb` — EDA complete (do not re-run)
- `notebooks/02_preprocessing.ipynb` — feature selection, encoding, pipeline serialisation
- `notebooks/03_ml_pycaret.ipynb` — PyCaret compare_models + tune + MLflow logging
- `notebooks/04_dl_pytorch.ipynb` — shallow MLP (128→64→1) with SLT controls
- `notebooks/05_evaluation.ipynb` — ML vs DL comparison, feature importance, enrichment
- `models/` — saved artifacts: `preprocessing_pipeline.joblib`, `best_ml_model.pkl`, `mlp_model.pth`
- `docs/eda_findings.md` — EDA findings reference
- `specs/001-crash-severity-model/` — spec, plan, tasks (T001–T045)
- `.specify/memory/constitution.md` — v1.0.0, 7 non-negotiable project principles
- `.agents/skills/data-scientist/scripts/hypothesis_tester.py` — t-tests, chi-square tests
- `airflow/` — Airflow learning environment with tutorial DAGs and ML pipeline orchestration

## Data Scientist Skill

The `data-scientist` skill is installed and active. Use it via the `Task` tool or invoke with `/data-scientist`. It follows a structured workflow: Define → Collect → Engineer → Train → Evaluate → Communicate.

Key conventions when doing ML work:
- Track all experiments with MLflow — mandatory metrics: `ein_macro_f1`, `eout_macro_f1`, `generalisation_gap`
- Always report effect size and CIs alongside p-values
- Start with logistic/linear regression before adding complexity
- Evaluate with ≥3 metrics; primary selection metric is macro F1 on held-out test set

## Apache Airflow (Workflow Orchestration)

Airflow is set up for learning workflow orchestration. Complete tutorial DAGs walk through:
- **01_hello_airflow.py** — basics: DAGs, operators, dependencies, task execution
- **02_crash_ml_pipeline.py** — real ML pipeline: notebooks, branching, parallel training, MLflow
- **03_advanced_concepts.py** — sensors, TaskGroups, XCom, dynamic tasks, advanced patterns

Quick start: `cd airflow`, `.\setup.ps1`, then `uv run airflow standalone` (UI: http://localhost:8080)

See `airflow/README.md` for comprehensive tutorial and `airflow/CHEATSHEET.md` for quick reference.

## Speckit Planning Workflow

Slash commands are available for structured feature development:
- `/speckit.specify` — write a feature spec
- `/speckit.plan` — generate a design/implementation plan
- `/speckit.tasks` — generate a task list from the plan
- `/speckit.implement` — execute tasks from `tasks.md`
