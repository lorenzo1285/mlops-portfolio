"""
Tutorial DAG 1: Hello Airflow
=============================

This is your first Airflow DAG! It demonstrates the basic concepts:
- DAG definition
- Task dependencies
- Different operator types
- Task execution

Run this to learn the fundamentals of Airflow.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


# ---------- Python Functions (used by PythonOperator) ----------

def print_hello():
    """Simple Python function that prints a greeting."""
    print("Hello from Airflow! 🚀")
    print("This task is running a Python function.")
    return "Hello task completed"


def process_data(**context):
    """
    More advanced function showing context access.
    
    The **context parameter gives you access to Airflow metadata like:
    - execution_date: when this DAG run started
    - task_instance: the current task instance object
    - dag_run: the current DAG run object
    """
    print(f"Processing data...")
    print(f"Execution date: {context['execution_date']}")
    print(f"Task instance: {context['task_instance'].task_id}")
    
    # Simulate some data processing
    result = {"records_processed": 100, "status": "success"}
    print(f"Result: {result}")
    
    # Return value can be accessed by downstream tasks using XCom
    return result


def print_completion(**context):
    """Demonstrate pulling data from previous task using XCom."""
    # Pull the return value from the previous task
    ti = context['task_instance']
    result = ti.xcom_pull(task_ids='process_data')
    
    print("=" * 50)
    print("Pipeline completed successfully!")
    print(f"Previous task result: {result}")
    print("=" * 50)


# ---------- DAG Definition ----------

# Default arguments applied to all tasks in this DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,  # Don't wait for previous runs
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,  # Retry failed tasks once
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
with DAG(
    dag_id='01_hello_airflow',
    default_args=default_args,
    description='Tutorial DAG to learn Airflow basics',
    schedule=None,  # Manual trigger only (use cron for scheduling, e.g., '@daily')
    start_date=datetime(2024, 1, 1),
    catchup=False,  # Don't backfill past dates
    tags=['tutorial', 'basics'],
) as dag:
    
    # ---------- Tasks ----------
    
    # Task 1: Bash command
    start = BashOperator(
        task_id='start',
        bash_command='echo "Starting the pipeline..." && date',
    )
    
    # Task 2: Python function
    hello = PythonOperator(
        task_id='hello',
        python_callable=print_hello,
    )
    
    # Task 3: Another Python function with context
    process = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        provide_context=True,
    )
    
    # Task 4: Bash command for cleanup
    cleanup = BashOperator(
        task_id='cleanup',
        bash_command='echo "Cleaning up temporary files..." && echo "Done!"',
    )
    
    # Task 5: Final completion message
    complete = PythonOperator(
        task_id='complete',
        python_callable=print_completion,
        provide_context=True,
    )
    
    # ---------- Define Dependencies (Task Flow) ----------
    
    # Linear flow: start → hello → process → cleanup → complete
    start >> hello >> process >> cleanup >> complete
    
    # Alternative syntax (same result):
    # start.set_downstream(hello)
    # hello.set_downstream(process)
    # process.set_downstream(cleanup)
    # cleanup.set_downstream(complete)
    
    # For parallel tasks, you can do:
    # start >> [task1, task2, task3] >> end


# ---------- How to Run This DAG ----------
"""
1. Start Airflow:
   airflow standalone
   
2. Open web UI:
   http://localhost:8080
   
3. Find this DAG: '01_hello_airflow'

4. Click the play button to trigger it

5. Watch the tasks execute in the Graph view

6. Click on tasks to see logs and details
"""
