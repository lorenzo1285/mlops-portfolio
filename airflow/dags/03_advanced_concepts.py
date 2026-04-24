"""
Tutorial DAG 3: Advanced Concepts
==================================

This DAG demonstrates intermediate/advanced Airflow concepts:
- TaskGroups for organizing tasks
- Dynamic task generation
- Sensors for waiting on conditions
- XCom for task communication
- Custom operators (example)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable


# ---------- Python Functions ----------

def check_if_ready(**context):
    """
    Sensor function: returns True when condition is met.
    
    In real scenarios, this might check:
    - If a file exists
    - If an API endpoint is available
    - If a database table is ready
    - If another service has completed
    """
    import random
    
    # Simulate a 70% chance of being ready
    is_ready = random.random() > 0.3
    
    if is_ready:
        print("✓ Condition met! Proceeding...")
    else:
        print("⏳ Not ready yet, will retry...")
    
    return is_ready


def extract_data(**context):
    """Extract data and push to XCom."""
    data = {
        "records": [
            {"id": 1, "value": 100},
            {"id": 2, "value": 200},
            {"id": 3, "value": 300},
        ],
        "timestamp": str(datetime.now()),
    }
    
    print(f"Extracted {len(data['records'])} records")
    
    # Push to XCom (other tasks can pull this)
    context['task_instance'].xcom_push(key='extracted_data', value=data)
    
    return data


def transform_data(**context):
    """Pull data from XCom and transform it."""
    ti = context['task_instance']
    
    # Pull from previous task
    data = ti.xcom_pull(task_ids='extract', key='extracted_data')
    
    # Transform: multiply all values by 2
    transformed = {
        "records": [
            {"id": r["id"], "value": r["value"] * 2}
            for r in data["records"]
        ],
        "timestamp": data["timestamp"],
        "transformed_at": str(datetime.now()),
    }
    
    print(f"Transformed {len(transformed['records'])} records")
    
    # Push transformed data
    ti.xcom_push(key='transformed_data', value=transformed)
    
    return transformed


def load_data(**context):
    """Load transformed data (final step)."""
    ti = context['task_instance']
    data = ti.xcom_pull(task_ids='transform', key='transformed_data')
    
    print("Loading data to destination...")
    for record in data["records"]:
        print(f"  Loaded: {record}")
    
    print(f"✓ Loaded {len(data['records'])} records successfully")


def generate_training_tasks(**context):
    """
    Example of dynamic task generation.
    Returns task_ids that should be created dynamically.
    """
    # In a real scenario, this might read from a config file,
    # database, or API to determine what tasks to create
    
    models = ["logistic_regression", "random_forest", "xgboost"]
    return models


def train_model(model_name: str, **context):
    """Simulate training a specific model."""
    print(f"Training {model_name}...")
    import time
    time.sleep(2)  # Simulate training time
    
    # Fake metrics
    import random
    accuracy = round(random.uniform(0.75, 0.95), 4)
    
    print(f"✓ {model_name} trained - Accuracy: {accuracy}")
    
    # Push metrics to XCom
    context['task_instance'].xcom_push(
        key=f'{model_name}_metrics',
        value={'model': model_name, 'accuracy': accuracy}
    )


# ---------- DAG Definition ----------

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='03_advanced_concepts',
    default_args=default_args,
    description='Advanced Airflow concepts tutorial',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['tutorial', 'advanced'],
) as dag:
    
    # ---------- Sensor: Wait for Condition ----------
    
    wait_for_ready = PythonSensor(
        task_id='wait_for_ready',
        python_callable=check_if_ready,
        poke_interval=30,  # Check every 30 seconds
        timeout=300,  # Give up after 5 minutes
        mode='poke',  # 'poke' (blocking) or 'reschedule' (non-blocking)
    )
    
    # ---------- TaskGroup 1: ETL Pipeline ----------
    
    with TaskGroup(group_id='etl_pipeline') as etl:
        extract = PythonOperator(
            task_id='extract',
            python_callable=extract_data,
        )
        
        transform = PythonOperator(
            task_id='transform',
            python_callable=transform_data,
        )
        
        load = PythonOperator(
            task_id='load',
            python_callable=load_data,
        )
        
        # TaskGroup internal dependencies
        extract >> transform >> load
    
    # ---------- TaskGroup 2: Model Training (Dynamic Tasks) ----------
    
    with TaskGroup(group_id='model_training') as training:
        # Dynamically create training tasks
        models = ["logistic_regression", "random_forest", "xgboost", "neural_net"]
        
        training_tasks = []
        for model in models:
            task = PythonOperator(
                task_id=f'train_{model}',
                python_callable=train_model,
                op_kwargs={'model_name': model},
            )
            training_tasks.append(task)
        
        # All training tasks can run in parallel
        # (no dependencies between them)
    
    # ---------- Aggregation ----------
    
    def aggregate_results(**context):
        """Collect results from all training tasks."""
        ti = context['task_instance']
        
        models = ["logistic_regression", "random_forest", "xgboost", "neural_net"]
        
        print("=" * 60)
        print("MODEL TRAINING RESULTS")
        print("=" * 60)
        
        all_metrics = []
        for model in models:
            metrics = ti.xcom_pull(
                task_ids=f'model_training.train_{model}',
                key=f'{model}_metrics'
            )
            if metrics:
                all_metrics.append(metrics)
                print(f"{metrics['model']:<25} Accuracy: {metrics['accuracy']:.4f}")
        
        # Find best model
        if all_metrics:
            best = max(all_metrics, key=lambda x: x['accuracy'])
            print("=" * 60)
            print(f"🏆 BEST MODEL: {best['model']} (Accuracy: {best['accuracy']:.4f})")
            print("=" * 60)
    
    aggregate = PythonOperator(
        task_id='aggregate_results',
        python_callable=aggregate_results,
    )
    
    # ---------- Final Summary ----------
    
    summary = BashOperator(
        task_id='summary',
        bash_command="""
        echo "========================================"
        echo "Pipeline Execution Complete!"
        echo "========================================"
        echo "Key concepts demonstrated:"
        echo "  ✓ PythonSensor for waiting on conditions"
        echo "  ✓ TaskGroups for organizing tasks"
        echo "  ✓ XCom for inter-task communication"
        echo "  ✓ Dynamic task generation"
        echo "  ✓ Parallel task execution"
        echo "========================================"
        """,
    )
    
    # ---------- Define Dependencies ----------
    
    # Wait until ready, then run ETL and training in parallel
    wait_for_ready >> [etl, training]
    
    # After both groups complete, aggregate and summarize
    [etl, training] >> aggregate >> summary


# ---------- Concepts Explained ----------
"""
1. **PythonSensor**
   - Waits for a condition to be true before proceeding
   - Useful for: file arrivals, API availability, external dependencies
   - poke_interval: how often to check (seconds)
   - timeout: how long to wait before failing

2. **TaskGroups**
   - Organize related tasks into logical groups
   - Improves DAG visualization and maintainability
   - Groups appear collapsed in the UI
   - Can nest groups within groups

3. **XCom (Cross-Communication)**
   - Share data between tasks
   - xcom_push(): send data from a task
   - xcom_pull(): retrieve data in another task
   - Stored in Airflow's metadata database
   - Limitation: works best for small data (< 1MB)

4. **Dynamic Task Generation**
   - Create tasks programmatically in a loop
   - Useful for: multiple models, data partitions, parallel processing
   - Tasks are created at DAG parse time, not runtime

5. **Parallel Execution**
   - Tasks without dependencies run concurrently
   - Controlled by executor configuration
   - Speeds up pipeline execution

6. **Task Dependencies**
   - Multiple upstreams: [task1, task2] >> task3
   - Multiple downstreams: task1 >> [task2, task3]
   - Join point: [task1, task2] >> task3 (waits for both)

7. **Airflow Variables** (not shown, but useful)
   - Global configuration values
   - Set via UI or CLI: airflow variables set key value
   - Access: Variable.get('key')
"""


# ---------- Exercises for Learning ----------
"""
Try modifying this DAG:

1. Add another TaskGroup for data validation
2. Create a sensor that waits for a specific file to exist
3. Add error handling with on_failure_callback
4. Implement a custom operator class
5. Use Variables to make the model list configurable
6. Add email notifications on success/failure
7. Create a SubDAG (deprecated but educational)
8. Implement task retries with exponential backoff
"""
