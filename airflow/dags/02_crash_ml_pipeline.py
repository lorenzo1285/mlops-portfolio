"""
Tutorial DAG 2: Crash Severity ML Pipeline
==========================================

A realistic ML pipeline that orchestrates your crash severity prediction workflow:
- Data preprocessing
- Model training (ML & DL)
- Model evaluation
- Experiment tracking with MLflow

This shows how to use Airflow for real ML workloads.
"""

from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.papermill.operators.papermill import PapermillOperator
from airflow.utils.trigger_rule import TriggerRule


# ---------- Configuration ----------

PROJECT_ROOT = Path(__file__).parent.parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "airflow" / "output"
MODELS_DIR = PROJECT_ROOT / "models"


# ---------- Python Functions ----------

def check_data_exists(**context):
    """Check if the data file exists before starting pipeline."""
    data_file = PROJECT_ROOT / "data" / "CGR_Crash_Data.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    size_mb = data_file.stat().st_size / (1024 * 1024)
    print(f"✓ Data file found: {data_file}")
    print(f"  Size: {size_mb:.2f} MB")
    
    return str(data_file)


def setup_output_dirs(**context):
    """Create output directories for notebook results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}")


def choose_training_branch(**context):
    """
    Branching logic: decide which models to train.
    
    In a real scenario, you might:
    - Check available compute resources
    - Look at experiment configs
    - Make decisions based on upstream task results
    
    Returns the task_id(s) to execute next.
    """
    # For this tutorial, we'll run both ML and DL in parallel
    # You could make this conditional based on some logic
    
    train_ml = True  # Could be based on config or previous results
    train_dl = True
    
    tasks_to_run = []
    if train_ml:
        tasks_to_run.append('train_ml_models')
    if train_dl:
        tasks_to_run.append('train_dl_model')
    
    print(f"Branching to tasks: {tasks_to_run}")
    return tasks_to_run


def compare_models(**context):
    """
    Compare trained models and select the best one.
    
    In a real implementation, this would:
    - Read metrics from MLflow
    - Compare performance across models
    - Select the best model for deployment
    - Update model registry
    """
    print("Comparing models from MLflow experiments...")
    
    # Placeholder for actual MLflow model comparison
    print("📊 Model Comparison:")
    print("  - Best ML Model (PyCaret): Macro F1 = 0.XX")
    print("  - Best DL Model (PyTorch): Macro F1 = 0.XX")
    print("  - Selected for deployment: [model_name]")
    
    return {"best_model": "placeholder", "metric": 0.0}


# ---------- DAG Definition ----------

default_args = {
    'owner': 'data_scientist',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='02_crash_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for crash severity prediction',
    schedule='@weekly',  # Run weekly (cron: '0 0 * * 0')
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'production', 'crash-severity'],
    max_active_runs=1,  # Only one run at a time
) as dag:
    
    # ---------- Stage 1: Setup & Validation ----------
    
    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=check_data_exists,
    )
    
    setup_dirs = PythonOperator(
        task_id='setup_dirs',
        python_callable=setup_output_dirs,
    )
    
    # ---------- Stage 2: Data Preprocessing ----------
    
    # Using PapermillOperator to execute Jupyter notebooks
    preprocessing = PapermillOperator(
        task_id='preprocessing',
        input_nb=str(NOTEBOOKS_DIR / '02_preprocessing.ipynb'),
        output_nb=str(OUTPUT_DIR / 'preprocessing_{{ ds }}.ipynb'),  # ds = execution date
        parameters={
            'output_dir': str(MODELS_DIR),
            'random_state': 42,
        },
        kernel_name='python3',
    )
    
    # ---------- Stage 3: Branching - Decide which models to train ----------
    
    branch_training = BranchPythonOperator(
        task_id='branch_training',
        python_callable=choose_training_branch,
    )
    
    # ---------- Stage 4: Model Training (Parallel) ----------
    
    train_ml = PapermillOperator(
        task_id='train_ml_models',
        input_nb=str(NOTEBOOKS_DIR / '03_ml_pycaret.ipynb'),
        output_nb=str(OUTPUT_DIR / 'ml_training_{{ ds }}.ipynb'),
        parameters={
            'n_folds': 5,
            'optimize_metric': 'F1',
        },
        kernel_name='python3',
    )
    
    train_dl = PapermillOperator(
        task_id='train_dl_model',
        input_nb=str(NOTEBOOKS_DIR / '04_dl_pytorch.ipynb'),
        output_nb=str(OUTPUT_DIR / 'dl_training_{{ ds }}.ipynb'),
        parameters={
            'epochs': 50,
            'batch_size': 256,
        },
        kernel_name='python3',
    )
    
    # ---------- Stage 5: Model Evaluation & Selection ----------
    
    # Join task: runs after both branches complete
    evaluate = PapermillOperator(
        task_id='evaluate_models',
        input_nb=str(NOTEBOOKS_DIR / '05_evaluation.ipynb'),
        output_nb=str(OUTPUT_DIR / 'evaluation_{{ ds }}.ipynb'),
        kernel_name='python3',
        trigger_rule=TriggerRule.NONE_FAILED,  # Run if at least one branch succeeded
    )
    
    compare = PythonOperator(
        task_id='compare_and_select',
        python_callable=compare_models,
    )
    
    # ---------- Stage 6: Reporting ----------
    
    generate_report = BashOperator(
        task_id='generate_report',
        bash_command=f"""
        echo "========================================"
        echo "ML Pipeline Execution Complete"
        echo "Execution Date: {{{{ ds }}}}"
        echo "========================================"
        echo "Artifacts saved to: {OUTPUT_DIR}"
        echo "Models saved to: {MODELS_DIR}"
        echo "MLflow UI: http://localhost:5000"
        echo "========================================"
        """,
    )
    
    # ---------- Define Task Dependencies ----------
    
    # Setup stage
    [validate_data, setup_dirs] >> preprocessing
    
    # Branching: decide which models to train
    preprocessing >> branch_training
    
    # Training branches (parallel)
    branch_training >> [train_ml, train_dl]
    
    # Evaluation joins both branches
    [train_ml, train_dl] >> evaluate >> compare >> generate_report


# ---------- Advanced Concepts Demonstrated ----------
"""
This DAG shows:

1. **PapermillOperator**: Execute Jupyter notebooks as tasks
   - Pass parameters to notebooks
   - Save parameterized outputs with execution date

2. **Branching**: Conditional task execution
   - BranchPythonOperator decides which downstream tasks run
   - Useful for: skip conditions, A/B testing, resource-based routing

3. **Parallel Execution**: Train ML & DL models simultaneously
   - Tasks without dependencies run in parallel
   - Speeds up pipeline execution

4. **Trigger Rules**: Control task execution logic
   - TriggerRule.NONE_FAILED: run if any upstream task succeeded
   - Others: ALL_SUCCESS (default), ONE_FAILED, etc.

5. **Templating**: Use Jinja templates for dynamic values
   - {{ ds }}: execution date (YYYY-MM-DD)
   - {{ ts }}: execution timestamp
   - Useful for timestamped outputs and logs

6. **Scheduling**: @weekly runs every Sunday at midnight
   - Other options: @daily, @hourly, cron expressions
   - schedule=None for manual triggers only
"""


# ---------- How to Run This Pipeline ----------
"""
Prerequisites:
1. Ensure notebooks exist: 02_preprocessing.ipynb, 03_ml_pycaret.ipynb, etc.
2. Make sure data file exists: data/CGR_Crash_Data.csv
3. Start MLflow: uv run mlflow ui

Running the DAG:
1. Start Airflow: airflow standalone
2. Open UI: http://localhost:8080
3. Enable the DAG: toggle the switch
4. Trigger manually: click the play button
5. Monitor: watch tasks in Graph or Grid view
6. Check logs: click on task boxes to see detailed logs

Advanced Usage:
- Edit schedule to '@daily' for daily runs
- Modify parameters in PapermillOperator for different configs
- Add your own tasks (e.g., model deployment, data validation)
- Use XCom to pass data between tasks
"""
