::page{title="Iris Classification Model with KNN using Apache Airflow"}

##### **Estimated time needed:** 45 minutes
In this guided project, you will delve into the world of machine learning by building an Iris classification model with the k-nearest neighbors (KNN) algorithm, and automating the entire workflow using Apache Airflow. You will start by exploring the Iris dataset, which contains measurements of Iris flowers, and learn to preprocess the data by handling missing values, encoding categorical variables, and splitting the data into training and testing sets. You will then implement the KNN algorithm, configure its parameters, and use the training data to train the model. You will evaluate the performance of the model using various metrics, such as accuracy, precision, recall, and F1-score, and fine-tune the model for optimal results.

Next, you will leverage the power of Apache Airflow to automate the entire workflow of building and deploying the Iris classification model. You will define and manage the workflow tasks in Apache Airflow using Directed Acyclic Graphs (DAGs), set up dependencies, and schedule the tasks. You will monitor the progress of the tasks and troubleshoot any issues that may arise. Finally, you will deploy the trained model for production use, set up a REST API for model inference, and configure model monitoring and alerting to ensure its performance in the production environment.


![flower](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/iris_flower_img.png "flower")
# Apache Airflow
Apache Airflow is an open-source platform for programmatically creating, scheduling, and monitoring workflows. It allows users to define their workflows as code, making it easy to schedule and automate complex tasks. Airflow provides a web interface for monitoring and managing workflows, as well as a robust set of plugins and integrations with other data technologies. This guided project will walk you through the basics of setting up and using Airflow for your own data pipeline workflows.


## Learning Objectives
In this guided project, you will learn the following:
- **Workflow automation:** Gain a thorough understanding of Apache Airflow's features and capabilities for workflow automation. Learn how to define Directed Acyclic Graphs (DAGs), set up dependencies between tasks, and schedule tasks to run at specific intervals.
- **Model training and evaluation:** Implement the KNN algorithm for classification tasks, configure its parameters, and use the Iris dataset to train the model.
- **Monitoring and alerting:** Learn how to monitor the performance and health of Apache Airflow workflows. Gain insights into Apache Airflow's built-in monitoring features.



## Prerequisites
- Python
- Machine Learning

------------












::page{title="Set Up Your Workspace"}

First things first, let's familiarize ourselves with the tools and materials we'll need for the project, including the programming environment and files for your web app.


## Cloud IDE

Cloud IDE, offered by Skills Network labs, is a lab environment that allows you to put your programming skills to the test in a real-world setting. There are many benefits to using Cloud IDE, whether you're developing an application or working with databases such as Airflow, MySQL, MongoDB, PostgreSQL, Cassandra, or Datasette. To make it easier for developers to adapt to the new environment, Cloud IDE is based on the familiar Visual Studio Code interface, reducing the learning curve.

Here's a breakdown of the components:

1. **Left Panel**: You can access and follow the step-by-step instructions shown in this panel, just like in the this guided project, as shown in the screenshot below.

2. **Right Panel**: This is where you'll get hands-on practice with your programming.

3. **File Explorer**: Like VS Code, you can create and access your files using the file explorer and make any edits you need in the Cloud IDE environment. You can find the file explorer in the screenshot below.

4. **Terminal**: This is where you can put the command lines.

![panel](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/panel.png "panel")

> Just a heads up, Cloud IDE doesn't save any of your files at the moment. To avoid any issues, we strongly suggest you finish this lab in one go.




-----------

::page{title="Open Airflow Server"}

Click on the below button to start apache airflow.

::openBigDataTool{tool="Apache Airflow" start="false"}







## What is Apache Airflow.?
Apache Airflow is an open-source platform used to programmatically author, schedule, and monitor workflows. It was originally created by Airbnb and is now maintained by the Apache Software Foundation.

Here are some basics of Apache Airflow:

DAGs (Directed Acyclic Graphs): Airflow uses DAGs to represent workflows. A DAG is a collection of tasks with dependencies between them. The tasks in a DAG are executed in a specific order.

Operators: Operators are the building blocks of tasks in Airflow. An operator is a Python class that defines a single task in a DAG. Airflow provides many built-in operators such as BashOperator, PythonOperator, and more.

Sensors: Sensors are special types of operators that wait for some external event or condition to occur before they execute their task.

Executors: Executors are responsible for executing tasks. Airflow supports several executors such as LocalExecutor, SequentialExecutor, and CeleryExecutor.

Connections: Connections are used to store the information required to connect to external systems such as databases, APIs, and more.

Variables: Variables are used to store key-value pairs that can be accessed from within a DAG.

Web UI: Airflow provides a web-based user interface that allows you to view and manage your DAGs, tasks, and logs.

CLI: Airflow provides a command-line interface (CLI) that allows you to interact with Airflow from the terminal.

Overall, Airflow is a powerful tool for creating, scheduling, and monitoring workflows. It provides a flexible and scalable platform for building data pipelines and automating complex workflows.

After clicking the 'Open Apache Airflow in IDE' button, You will able to see the below screen

![Airflow_start](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/step2_airflow_start.png "Airflow_start")



-------------

::page{title="Connect with Airflow Server Using Id Password"}

After following the steps from the previous page, you will see the following.

![Airflow_server_open](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/s3_click_airflow_webserver_copy_id_pass.png "Airflow_server_open")

You will be able to see status in blue button. In above image you can see it's stop status.
After getting starting status of apache airflow click on the below button to start the application for log in into airflow.


::startApplication{port="8080" display="internal" name="airflow webserver" route="/"}

Log in by copying the user ID and password from the last window

Just login and After logging in, start your journey to learn airflow.

![login_button](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/s4_login_id_pass.png "login_button")



-------------

::page{title="Create Folder structure"}

![terminal_command](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/terminal_command.png "terminal_command")

Run the following commands in your terminal to create a folder for your Airflow DAGs (i.e. your project).

```bash
mkdir airflow
```
```bash
cd airflow
```
```bash
mkdir dags
```
-------------

::page{title="Create Python file"}

Create a new Python file named `iris_classification.py` and save it into your `dags` folder by typing the following commands
```
cd dags
touch iris_classification.py
```
Click on the below button to create a Python file.

::openFile{path="airflow/dags/iris\_classification.py"}


##### Note: Click on the ```create the file``` button.
![create_file_permission](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/create_file.png "create_file_permission")



You can also create a new file using the file menu.
###### Note: Make sure you create file ```iris_classification.py ```  in ```dags folder```.


![new_file](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/create_new_file.png "new_file")



##### Note: IIf you are having trouble creating a Python file, you can also use the code below:

```
sudo chown -R 100999 /home/project/airflow/dags
sudo chmod -R g+rw /home/project/airflow/dags
```

-------------

::page{title="Writing Python Code"}

In this section,we will show you how to use Python code with Airflow, a powerful workflow management platform. You'll learn how to create your own Dags (Directed Acyclic Graphs) and perform task monitoring and other operations. By the end, you'll have the knowledge to effectively utilize Airflow to streamline your workflows and boost productivity.

Write a below code into that file ```iris_classification.py```

Importing required libraries
```python
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import math
import numpy as np
```
Create a function for reading csv file using pandas
###### Note: The xcom_push function is used in Apache Airflow to push a value to XCom, which is a feature that allows data sharing between tasks in a workflow. The xcom_push function is called on the task instance (context['ti']) to store a value with a specified key in the XCom storage.

```python
def train_iris_model_1(**context):
    # Your code for training and evaluating the Iris model with KNN algorithm (DAG 1)
    iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
    iris_data['class'] = iris_data['class'].astype('category').cat.codes
    # convert into Dictionary for passing into another function using xcom_push method
    df_dict = iris_data.to_dict(orient='records')
    # Push this df_dict as an iris using xcom_push method to access in train_iris_model2 function
    context['ti'].xcom_push(key='iris', value=df_dict)
    pass
```
Create a function for splitting a dataframe into traning  data and test data

```python
def train_iris_model_2(**context):
    # Your code for training and evaluating the Iris model with KNN algorithm (DAG 2)
    # xcom_pull will help you to get df_dict variable from train_iris_model_1 function
    iris_dict = context['ti'].xcom_pull(key='iris')
    iris_data= pd.DataFrame(iris_dict)
    # Split train test data
    X = iris_data.iloc[:, :-1].values
    y = iris_data.iloc[:, -1].values
    split = int(0.8 * len(iris_data))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    # Convert into list
    X_train=X_train.tolist()
    y_train=y_train.tolist()
    X_test=X_test.tolist()
    y_test=y_test.tolist()
    # convert into Dictionary for passing into another function using xcom_push method
    X_train={'X_train': X_train}
    y_train={'y_train': y_train}
    X_test={'X_test': X_test}
    y_test={'y_test': y_test}
    # Push variables using xcom_push method to access in train_iris_model_3 function
    context['ti'].xcom_push(key='X_train', value=X_train)
    context['ti'].xcom_push(key='y_train', value=y_train)
    context['ti'].xcom_push(key='X_test', value=X_test)
    context['ti'].xcom_push(key='y_test', value=y_test)
    pass
```
Build a KNN classification model and Calculate accuracy
```python

def train_iris_model_3(**context):
    # Your code for training and evaluating the Iris model with KNN algorithm (DAG 3)
    X_train = context['ti'].xcom_pull(key='X_train')
    y_train = context['ti'].xcom_pull(key='y_train')
    X_test = context['ti'].xcom_pull(key='X_test')
    y_test = context['ti'].xcom_pull(key='y_test')
    # Convert Dict into numpy array to apply KNN Algorithm
    X_train = np.array(X_train['X_train'])
    y_train = np.array(y_train['y_train'])
    X_test = np.array(X_test['X_test'])
    y_test = np.array(y_test['y_test'])
    predictions = []
    # KNN Algorithm
    for i in range(len(X_test)):
        distances = []
        for j in range(len(X_train)):
            dist = math.sqrt(sum([(a - b)**2 for a, b in zip(X_test[i], X_train[j])]))
            distances.append((dist, y_train[j]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:3]
        classes = [neighbor[1] for neighbor in neighbors]
        # Predict Classes
        prediction = max(set(classes), key=classes.count)
        predictions.append(prediction)
    # Calculate the Accuracy
    accuracy = sum([1 for i in range(len(y_test)) if y_test[i] == predictions[i]]) / float(len(y_test))
    print(f"Accuracy: {accuracy}")
```
Define the Dags
```python

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 5, 22),
}
# Here we have created 1 main tag iris_classification_models and 3 tasks
with DAG('iris_classification_models', default_args=default_args, schedule_interval=None) as dag:
    iris_classification_model_1 = PythonOperator(
        task_id='train_iris_classification_model_1',
        python_callable=train_iris_model_1,
    )
    iris_classification_model_2 = PythonOperator(
        task_id='train_iris_classification_model_2',
        python_callable=train_iris_model_2,
    )
    iris_classification_model_3 = PythonOperator(
        task_id='train_iris_classification_model_3',
        python_callable=train_iris_model_3,
    )
    iris_classification_model_1 >> iris_classification_model_2 >> iris_classification_model_3
```

Now go to Airflow Webserver and serach for ```iris_classification_models``` in dag search It will Automatically sync to the DAGs which are created by Python file.
![find_dag](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/find_dag.png "find_dag")



-------------

::page{title="What Are DAGs (Directed Acyclic Graphs), and Why Do We Generate Them?"}

DAGs (Directed Acyclic Graphs) are used in Apache Airflow to represent workflows. DAGs are a collection of tasks with dependencies between them, and each task represents a specific action that needs to be executed as part of the workflow.

There are several reasons why we generate DAGs in Airflow:

Workflow automation: DAGs provide a way to automate complex workflows by defining the sequence of tasks that need to be executed and the dependencies between them.

Improved visibility: DAGs make it easier to understand the structure of a workflow, as all the tasks and their dependencies are clearly defined.

Increased efficiency: DAGs help in automating repetitive tasks, which can reduce errors and save time.

Task dependencies: DAGs allow us to specify dependencies between tasks, ensuring that tasks are executed in the correct order.

Task parallelism: DAGs allow us to define parallelism between tasks, allowing multiple tasks to be executed at the same time if they don't have any dependencies on each other.

Task resiliency: DAGs allow us to define task retries, so if a task fails due to any reason, Airflow can automatically retry that task.

Version control: DAGs can be version controlled like any other code, which allows us to track changes made to workflows over time and easily roll back changes if needed.

Overall, DAGs are an essential component of Apache Airflow, and they provide a powerful way to manage complex workflows and automate repetitive tasks.


![dags](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/dag_image.png "dags")
------------

::page{title="View your DAGs and Jobs"}


**Congratulations!** Finally, you can run the Airflow program, and you can check the DAGs in the graph tab. You can also trigger a job to run manually using the play button. :tw-25b6:


You can also try below code to schedule a job at a particular time.

Change the date in the below code.
```default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 4, 25),
}
```

![dags_run_graph](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/dags_run_success.png "dags_run_graph")

**After successful running your DAG the status of DAG to successfull as you can see in above image.
You can also check particular dag's information.**
1. Status: The status of a task indicates its current state. When you hover over a task node, you may see the task's current status, such as "success," "failed," "running," or "skipped." This information helps monitor the progress and outcomes of task executions.

2. Task ID: The task ID is a unique identifier for each task instance. When you hover over a task node, you may see the task ID associated with that task instance. This ID is useful for tracking and troubleshooting specific task executions.

3. Run: It represents the specific date and time when a task instance was executed. When you hover over a task node, you may see the execution date associated with that task instance. This information is helpful for identifying when a task was executed and tracking the history of task runs.

4. Run ID: The default run ID is generated automatically for each DAG run. The run ID is a unique identifier that helps track and manage individual DAG runs. The default run ID follows the format {DAG_ID}_{EXECUTION_DATE}, where:

	{DAG_ID} is the ID of the DAG.
	{EXECUTION_DATE} is the date and time when the DAG run started.


5. Python Operator: The PythonOperator is used to execute arbitrary Python functions as tasks within a DAG. It allows you to define a Python function that will be executed when the task runs



![dag_info](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/1dag_info.png "dag_info")

**You can also check the particular dags's variables values which we are passing uing xcom function.
In below image you can see we are passing values of X_train, y_train , X_test, y_test from train_iris_model_2 to train_iris_model_3**
The push_value function is defined as the python_callable for the PythonOperator. Inside the function, we set a value of X_train, y_train , X_test, y_test and use xcom_push to store it in XCom with the key.

![x_com](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/xcom_info.png "x_com")

**You can also check the list of jobs from Browse button jobs tab.You can also check the list of jobs from Browse button jobs tab.**
![job_list_dags](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/job_list_dags.png "job_list_dags")

::page{title="Task Monitoring Using Airflow"}



Apache Airflow provides built-in features for monitoring workflows and tasks, which can help teams identify and resolve issues quickly. Some of the key monitoring features provided by Airflow include:

Task logs: Airflow logs all task runs and stores them in a database or an object store. Teams can use these logs to troubleshoot issues and monitor task progress.

Task statistics: Airflow provides task statistics, such as the number of tasks completed, the number of tasks failed, and the duration of tasks. Teams can use these statistics to monitor the performance of workflows and tasks.

Alerting: Airflow can send alerts to teams when tasks fail or when specific events occur. Teams can configure alerting rules based on their needs, such as sending an email or a Slack message.

SLA monitoring: Airflow can monitor SLAs (Service Level Agreements) for workflows and tasks, ensuring that tasks are completed within a specified timeframe.

Visualization: Airflow provides a web-based UI that allows teams to visualize workflows and task dependencies. Teams can use this UI to monitor the progress of workflows and identify issues quickly.

Metrics collection: Airflow can collect metrics such as CPU usage, memory usage, and disk usage for workflows and tasks. Teams can use these metrics to monitor resource usage and optimize workflows for better performance.

By leveraging these monitoring features provided by Airflow, teams can ensure that their workflows are running smoothly and that any issues are quickly identified and resolved. This can help improve the reliability of workflows and reduce downtime, resulting in better productivity and cost savings.

You can also monitor task instance.



![task_instance_informations](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IND-GPXX0DNQEN/images/task_inst_info.png "task_instance_informations")

In Apache Airflow, a task instance represents a specific instance of a task that is executed as part of a workflow. A task instance is created when a task is scheduled to run and contains information such as the task ID, the execution date, the task state (e.g. running, success, or failure), and any parameters or inputs required for the task.

When a task is executed, Airflow creates a new task instance for that execution. Each task instance is associated with a unique execution date, allowing multiple instances of the same task to be run at different times.

Task instances also provide visibility into the status of each task and allow teams to monitor the progress of workflows. Teams can use the Airflow web UI to view the state of each task instance, check for errors, and troubleshoot issues.

Overall, task instances are a critical component of Airflow's workflow management system, providing the information and tracking necessary to ensure that tasks are executed correctly and efficiently.













