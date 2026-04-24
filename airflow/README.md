# Apache Airflow Learning Guide

Welcome to your hands-on Airflow learning environment! This guide will help you master workflow orchestration for data pipelines and ML workflows.

## 📚 What is Airflow?

Apache Airflow is an open-source platform for:
- **Orchestrating** complex workflows
- **Scheduling** recurring jobs
- **Monitoring** pipeline execution
- **Managing dependencies** between tasks

Think of it as a "cron on steroids" with a web UI, Python-based configuration, and powerful features for data engineering and MLOps.

---

## 🚀 Quick Start

### 1. Initialize Airflow

First, set up Airflow's database and create an admin user:

```powershell
# Set Airflow home directory (where config and database live)
$env:AIRFLOW_HOME = "$PWD\airflow"

# Initialize the database
uv run airflow db init

# Create an admin user
uv run airflow users create `
    --username admin `
    --firstname Admin `
    --lastname User `
    --role Admin `
    --email admin@example.com `
    --password admin
```

### 2. Start Airflow

Run Airflow in standalone mode (includes webserver + scheduler):

```powershell
uv run airflow standalone
```

This will:
- Start the web server on http://localhost:8080
- Start the scheduler (executes DAGs)
- Display login credentials in the console

### 3. Access the Web UI

1. Open http://localhost:8080 in your browser
2. Login with username: `admin`, password: `admin` (or what you set above)
3. You'll see the DAG list page

---

## 📖 Tutorial Path

Work through the DAGs in order to build your Airflow skills:

### **Tutorial 1: Hello Airflow** (`01_hello_airflow.py`)
**Goal**: Learn the fundamentals

**Concepts**:
- DAG definition and structure
- BashOperator and PythonOperator
- Task dependencies (`>>` syntax)
- Task execution and logging

**How to run**:
1. Go to the Airflow UI
2. Find `01_hello_airflow` in the DAG list
3. Toggle it ON (if paused)
4. Click the ▶️ (play) button to trigger
5. Click on the DAG name to see the Graph view
6. Watch tasks turn from gray → yellow → green
7. Click task boxes to view logs

**Things to try**:
- Modify the `print_hello()` function
- Add a new task between existing ones
- Change task dependencies to run tasks in parallel
- Make a task fail on purpose (raise an exception)
- Check the logs to see what happened

---

### **Tutorial 2: Crash ML Pipeline** (`02_crash_ml_pipeline.py`)
**Goal**: Orchestrate a real ML workflow

**Concepts**:
- PapermillOperator (execute Jupyter notebooks)
- BranchPythonOperator (conditional execution)
- Trigger rules (control when tasks run)
- Templating with Jinja (dynamic values)
- Parallel task execution

**Prerequisites**:
- Make sure your notebooks exist:
  - `notebooks/02_preprocessing.ipynb`
  - `notebooks/03_ml_pycaret.ipynb`
  - `notebooks/04_dl_pytorch.ipynb`
  - `notebooks/05_evaluation.ipynb`
- Data file: `data/CGR_Crash_Data.csv`

**How to run**:
1. Enable the DAG in the UI
2. Trigger it manually
3. Watch the Graph view:
   - Preprocessing runs first
   - Branch decides which training tasks run
   - ML and DL training run in parallel
   - Evaluation joins both branches
   - Report generates at the end

**Things to try**:
- Modify the `choose_training_branch()` logic to only train ML
- Add parameters to notebook tasks
- Schedule it to run daily (`schedule='@daily'`)
- Add a data validation task before preprocessing
- Create a deployment task after evaluation

**Real-world usage**:
This is how you'd orchestrate ML workflows in production:
- Daily/weekly retraining on new data
- Experiment tracking with MLflow
- Model comparison and selection
- Automated deployment pipelines

---

### **Tutorial 3: Advanced Concepts** (`03_advanced_concepts.py`)
**Goal**: Master intermediate/advanced features

**Concepts**:
- **PythonSensor**: Wait for conditions (files, APIs, etc.)
- **TaskGroups**: Organize tasks into logical units
- **XCom**: Share data between tasks
- **Dynamic task generation**: Create tasks programmatically
- **Parallel execution**: Run multiple tasks simultaneously

**How to run**:
1. Trigger `03_advanced_concepts` in the UI
2. Watch the sensor wait for the condition
3. Observe TaskGroups in the Graph view (collapsible)
4. Click tasks to see XCom values being passed
5. Check logs for aggregated results

**Things to try**:
- Modify the sensor condition to always succeed/fail
- Add another TaskGroup for data quality checks
- Create a sensor that waits for a file to exist
- Pass larger datasets through XCom (see limits)
- Add more models to the dynamic training loop

---

## 🎯 Key Concepts

### DAG Structure
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    dag_id='my_dag',
    start_date=datetime(2024, 1, 1),
    schedule='@daily',  # or cron: '0 0 * * *'
    catchup=False,
) as dag:
    
    task1 = PythonOperator(
        task_id='task1',
        python_callable=my_function,
    )
    
    task2 = PythonOperator(
        task_id='task2',
        python_callable=another_function,
    )
    
    # Define dependencies
    task1 >> task2
```

### Common Operators

| Operator | Use Case | Example |
|----------|----------|---------|
| `BashOperator` | Run shell commands | `bash_command='echo hello'` |
| `PythonOperator` | Run Python functions | `python_callable=my_func` |
| `PapermillOperator` | Execute Jupyter notebooks | `input_nb='notebook.ipynb'` |
| `BranchPythonOperator` | Conditional routing | Returns task_id to execute |
| `PythonSensor` | Wait for condition | Returns True when ready |

### Task Dependencies

```python
# Sequential
task1 >> task2 >> task3

# Parallel fan-out
task1 >> [task2, task3, task4]

# Parallel fan-in (join)
[task2, task3, task4] >> task5

# Mixed
task1 >> [task2, task3] >> task4
```

### Scheduling

| Expression | Meaning |
|------------|---------|
| `None` | Manual trigger only |
| `'@once'` | Run once, then never again |
| `'@hourly'` | Every hour (0 * * * *) |
| `'@daily'` | Every midnight (0 0 * * *) |
| `'@weekly'` | Every Sunday midnight (0 0 * * 0) |
| `'@monthly'` | First of month (0 0 1 * *) |
| `'0 */4 * * *'` | Every 4 hours |
| `'30 8 * * 1-5'` | 8:30 AM weekdays |

---

## 🛠️ Common Commands

### Airflow CLI

```powershell
# Start Airflow
uv run airflow standalone

# List DAGs
uv run airflow dags list

# Test a specific task (doesn't save state)
uv run airflow tasks test <dag_id> <task_id> 2024-01-01

# Trigger a DAG manually
uv run airflow dags trigger <dag_id>

# Backfill (run for historical dates)
uv run airflow dags backfill <dag_id> --start-date 2024-01-01 --end-date 2024-01-31

# Pause/unpause a DAG
uv run airflow dags pause <dag_id>
uv run airflow dags unpause <dag_id>

# Clear task state (rerun)
uv run airflow tasks clear <dag_id> --task-regex <task_id>
```

### Managing Variables

```powershell
# Set a variable
uv run airflow variables set my_key my_value

# Get a variable
uv run airflow variables get my_key

# List all variables
uv run airflow variables list

# Use in DAG
from airflow.models import Variable
value = Variable.get('my_key')
```

---

## 📁 Project Structure

```
airflow/
├── dags/                    # Your DAG files (automatically discovered)
│   ├── 01_hello_airflow.py
│   ├── 02_crash_ml_pipeline.py
│   └── 03_advanced_concepts.py
├── logs/                    # Task execution logs
├── plugins/                 # Custom operators, hooks, sensors
├── output/                  # Notebook outputs (from Papermill)
└── airflow.cfg             # Configuration file
```

**Important**: Airflow scans `dags/` directory for Python files containing DAG objects.

---

## 🎨 Web UI Guide

### Main Views

1. **DAGs List**
   - See all DAGs and their status
   - Toggle ON/OFF
   - Trigger runs
   - View recent runs

2. **Graph View**
   - Visual representation of task dependencies
   - Color-coded status (gray=not started, yellow=running, green=success, red=failed)
   - Click tasks to see details/logs

3. **Tree View**
   - Historical runs in a timeline
   - See which tasks succeeded/failed across runs
   - Useful for debugging patterns

4. **Gantt Chart**
   - Task execution timeline
   - Identify bottlenecks and parallelization opportunities

5. **Task Logs**
   - Click any task box → Logs
   - See stdout/stderr from task execution
   - Essential for debugging

### Task States

| Color | State | Meaning |
|-------|-------|---------|
| ⚪ Gray | `none` | Not yet scheduled |
| 🟡 Yellow | `running` | Currently executing |
| 🟢 Green | `success` | Completed successfully |
| 🔴 Red | `failed` | Task failed |
| 🟠 Orange | `upstream_failed` | Skipped due to upstream failure |
| 🟣 Purple | `up_for_retry` | Failed, will retry |

---

## 🚧 Common Issues & Solutions

### DAG not appearing in UI
**Cause**: Python syntax error or import error

**Solution**:
```powershell
# Check for errors
uv run python airflow/dags/my_dag.py

# View DAG import errors in UI
# Go to Browse → DAG Import Errors
```

### Task stuck in "running"
**Cause**: Task is waiting for resources or truly running

**Solution**:
- Check task logs in UI
- Verify executor has capacity (default: SequentialExecutor = 1 task at a time)
- Kill hung tasks: Mark Failed in UI

### XCom data too large
**Cause**: XCom stores data in metadata DB (SQLite by default)

**Solution**:
- Use XCom for small data (<1 MB)
- For large data: save to file/S3/database, pass path via XCom
- Or use Custom XCom Backend

### Notebooks not executing
**Cause**: Kernel not found or path incorrect

**Solution**:
```powershell
# List available kernels
uv run jupyter kernelspec list

# Ensure notebooks exist
ls notebooks/
```

---

## 🎓 Learning Exercises

### Beginner
1. ✅ Create a DAG with 5 tasks in a sequence
2. ✅ Add a task that fails, observe retries
3. ✅ Create parallel tasks (3 tasks fan-out from 1 task)
4. ✅ Use XCom to pass data between 2 tasks
5. ✅ Schedule a DAG to run daily

### Intermediate
1. ✅ Create a TaskGroup with 5 tasks inside
2. ✅ Implement a branching workflow (if-else logic)
3. ✅ Use a sensor to wait for a file to exist
4. ✅ Execute a Jupyter notebook with PapermillOperator
5. ✅ Create dynamic tasks in a loop

### Advanced
1. ✅ Build a complete ML pipeline with:
   - Data validation sensor
   - Preprocessing task
   - Parallel model training
   - Model comparison
   - Deployment decision
2. ✅ Implement custom operator class
3. ✅ Use Airflow with MLflow for experiment tracking
4. ✅ Set up email/Slack notifications
5. ✅ Deploy Airflow with Docker Compose + PostgreSQL

---

## 📚 Resources

### Official Documentation
- [Airflow Docs](https://airflow.apache.org/docs/)
- [Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/)
- [Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

### Key Topics to Explore
- **Executors**: SequentialExecutor, LocalExecutor, CeleryExecutor, KubernetesExecutor
- **Connections**: Configure external systems (databases, cloud, APIs)
- **Pools**: Limit concurrent tasks
- **SLAs**: Service level agreement monitoring
- **Callbacks**: on_success_callback, on_failure_callback
- **Jinja Templating**: Dynamic task configuration

### Production Considerations
- Use PostgreSQL/MySQL instead of SQLite
- Deploy with Docker or Kubernetes
- Set up proper authentication and RBAC
- Monitor with Prometheus/Grafana
- Use remote logging (S3, GCS)
- Implement data quality checks

---

## 🎯 Next Steps

1. **Complete all 3 tutorial DAGs** - understand the basics
2. **Modify the ML pipeline** - adapt it to your needs
3. **Create your own DAG** - orchestrate a real workflow
4. **Explore operators** - try BashOperator, EmailOperator, etc.
5. **Learn about executors** - understand parallelization
6. **Study production patterns** - idempotency, data validation, error handling
7. **Deploy to production** - Docker Compose or Kubernetes

---

## 🆘 Getting Help

- Check task logs in the UI (most issues show up here)
- Read Airflow documentation
- Search [Stack Overflow](https://stackoverflow.com/questions/tagged/airflow)
- Review [GitHub Issues](https://github.com/apache/airflow/issues)

---

**Happy Orchestrating! 🎵**

Start with `01_hello_airflow.py` and work your way up. Airflow is powerful - take it step by step, and you'll be orchestrating complex ML pipelines in no time!
