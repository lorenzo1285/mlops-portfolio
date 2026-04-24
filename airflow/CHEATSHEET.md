# Airflow Quick Reference Cheat Sheet

## 🚀 Starting Airflow

```powershell
# Set Airflow home
$env:AIRFLOW_HOME = "$PWD\airflow"

# Start Airflow (includes webserver + scheduler)
uv run airflow standalone

# Access UI
# http://localhost:8080 (admin/admin)
```

## 📋 Common CLI Commands

### DAG Management
```powershell
# List all DAGs
uv run airflow dags list

# Trigger a DAG run
uv run airflow dags trigger <dag_id>

# Pause/Unpause
uv run airflow dags pause <dag_id>
uv run airflow dags unpause <dag_id>

# Show DAG structure
uv run airflow dags show <dag_id>
```

### Task Operations
```powershell
# Test a task (dry run, no state save)
uv run airflow tasks test <dag_id> <task_id> 2024-01-01

# List tasks in a DAG
uv run airflow tasks list <dag_id>

# Clear task state (force rerun)
uv run airflow tasks clear <dag_id> -t <task_id>
```

### Variables
```powershell
# Set variable
uv run airflow variables set <key> <value>

# Get variable
uv run airflow variables get <key>

# List all
uv run airflow variables list

# Import from JSON
uv run airflow variables import variables.json
```

## 🔧 DAG Syntax

### Basic DAG Structure
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_function():
    print("Hello from Airflow!")

with DAG(
    dag_id='my_dag',
    start_date=datetime(2024, 1, 1),
    schedule='@daily',
    catchup=False,
    tags=['example'],
) as dag:
    
    task = PythonOperator(
        task_id='my_task',
        python_callable=my_function,
    )
```

### Task Dependencies
```python
# Sequential
t1 >> t2 >> t3

# Parallel fan-out
t1 >> [t2, t3, t4]

# Parallel fan-in
[t2, t3] >> t4

# Alternative syntax
t1.set_downstream(t2)
t2.set_upstream(t1)
```

### Common Operators
```python
# Bash command
from airflow.operators.bash import BashOperator
task = BashOperator(task_id='bash', bash_command='echo hi')

# Python function
from airflow.operators.python import PythonOperator
task = PythonOperator(task_id='py', python_callable=func)

# Execute notebook
from airflow.providers.papermill.operators.papermill import PapermillOperator
task = PapermillOperator(
    task_id='notebook',
    input_nb='input.ipynb',
    output_nb='output.ipynb',
)

# Branching (conditional)
from airflow.operators.python import BranchPythonOperator
task = BranchPythonOperator(
    task_id='branch',
    python_callable=choose_branch,  # returns task_id
)

# Sensor (wait for condition)
from airflow.sensors.python import PythonSensor
task = PythonSensor(
    task_id='wait',
    python_callable=check_condition,  # returns True/False
    poke_interval=60,
)
```

## 📅 Scheduling

| Schedule | Cron | Description |
|----------|------|-------------|
| `None` | - | Manual only |
| `@once` | - | Run once |
| `@hourly` | `0 * * * *` | Every hour |
| `@daily` | `0 0 * * *` | Midnight daily |
| `@weekly` | `0 0 * * 0` | Sunday midnight |
| `@monthly` | `0 0 1 * *` | 1st of month |
| `@yearly` | `0 0 1 1 *` | Jan 1st |

### Custom Cron
```python
# Every 15 minutes
schedule='*/15 * * * *'

# 9 AM weekdays
schedule='0 9 * * 1-5'

# Every 4 hours
schedule='0 */4 * * *'
```

## 🔄 XCom (Cross-Communication)

### Push data
```python
def push_data(**context):
    context['task_instance'].xcom_push(key='my_key', value='my_value')
    # Or simply return
    return 'my_value'  # auto-pushed with key='return_value'
```

### Pull data
```python
def pull_data(**context):
    ti = context['task_instance']
    value = ti.xcom_pull(task_ids='previous_task', key='my_key')
    # Or get return value
    value = ti.xcom_pull(task_ids='previous_task')  # gets 'return_value'
```

## 🎯 Task Groups

```python
from airflow.utils.task_group import TaskGroup

with TaskGroup(group_id='my_group') as group:
    t1 = PythonOperator(task_id='task1', python_callable=func1)
    t2 = PythonOperator(task_id='task2', python_callable=func2)
    t1 >> t2

# Reference: my_group.task1, my_group.task2
```

## 🎨 Jinja Templates

### Available variables
```python
# In bash_command or other templated fields:
bash_command="""
    echo "Execution date: {{ ds }}"              # 2024-01-01
    echo "Timestamp: {{ ts }}"                   # 2024-01-01T00:00:00+00:00
    echo "DAG run ID: {{ run_id }}"
    echo "Task ID: {{ task.task_id }}"
    echo "DAG ID: {{ dag.dag_id }}"
"""

# Date formats:
# {{ ds }}              2024-01-01
# {{ ds_nodash }}       20240101
# {{ ts }}              2024-01-01T00:00:00+00:00
# {{ ts_nodash }}       20240101T000000
```

## ⚙️ Trigger Rules

```python
from airflow.utils.trigger_rule import TriggerRule

task = PythonOperator(
    task_id='my_task',
    python_callable=func,
    trigger_rule=TriggerRule.ALL_SUCCESS,  # default
)
```

| Rule | Behavior |
|------|----------|
| `ALL_SUCCESS` | All parents succeeded (default) |
| `ALL_FAILED` | All parents failed |
| `ALL_DONE` | All parents finished (any state) |
| `ONE_SUCCESS` | At least one parent succeeded |
| `ONE_FAILED` | At least one parent failed |
| `NONE_FAILED` | No parent failed (some may be skipped) |
| `NONE_SKIPPED` | No parent was skipped |

## 🔍 Task States

| State | Meaning | Color (UI) |
|-------|---------|------------|
| `none` | Not scheduled yet | Gray |
| `scheduled` | Queued for execution | Tan |
| `queued` | In executor queue | Gray |
| `running` | Currently executing | Yellow |
| `success` | Completed successfully | Green |
| `failed` | Task failed | Red |
| `up_for_retry` | Failed, will retry | Purple |
| `skipped` | Skipped (branch not taken) | Pink |
| `upstream_failed` | Parent task failed | Orange |

## 🛠️ Debugging

### View logs
```powershell
# UI: Click task → Logs

# CLI: View logs for a task instance
uv run airflow tasks logs <dag_id> <task_id> 2024-01-01
```

### Test task locally
```powershell
# Run task without saving state
uv run airflow tasks test <dag_id> <task_id> 2024-01-01

# Useful for debugging
```

### Check DAG for errors
```powershell
# Parse DAG file
python airflow/dags/my_dag.py

# List import errors
uv run airflow dags list-import-errors
```

## 📊 Monitoring

### Task duration
```python
# In DAG definition
default_args = {
    'sla': timedelta(hours=2),  # Alert if task takes > 2 hours
}
```

### Email alerts
```python
default_args = {
    'email': ['your@email.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}
```

### Callbacks
```python
def on_failure(context):
    print(f"Task {context['task_instance'].task_id} failed!")

task = PythonOperator(
    task_id='my_task',
    python_callable=func,
    on_failure_callback=on_failure,
)
```

## 🎓 Best Practices

### ✅ DO
- Keep DAGs simple and readable
- Use meaningful task_ids
- Make tasks idempotent (can run multiple times safely)
- Use XCom for small data only (< 1 MB)
- Tag DAGs for organization
- Set proper retry policies
- Use sensors for external dependencies
- Test tasks with `airflow tasks test`

### ❌ DON'T
- Don't use top-level code in DAG files (runs on every scheduler parse)
- Don't make tasks dependent on execution order without explicit dependencies
- Don't store large data in XCom (use files/databases)
- Don't use SubDAGs (deprecated - use TaskGroups)
- Don't use `schedule_interval` (deprecated - use `schedule`)
- Don't set `start_date` dynamically (use fixed date)

## 📦 Project Structure

```
airflow/
├── dags/           # DAG files (auto-discovered)
├── logs/           # Task logs
├── plugins/        # Custom operators, hooks, sensors
├── airflow.cfg     # Configuration
└── airflow.db      # Metadata database (SQLite)
```

## 🔐 Security Note

Default credentials (standalone mode):
- Username: `admin`
- Password: `admin` (or see console output)

Change in production!

---

**Quick Start**:
```powershell
$env:AIRFLOW_HOME = "$PWD\airflow"
uv run airflow standalone
# Open http://localhost:8080
```

**Learn**: Work through tutorial DAGs in `airflow/dags/` directory!
