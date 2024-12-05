from airflow import DAG
from airflow.operators import PythonOperator
from datetime import datetime
import os

def preprocess_data():
    os.system("python preprocess.py")  # Preprocessing script

def train_model():
    os.system("python train_model.py")  # Training script

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 12, 5),
    'retries': 1,
}

dag = DAG('retail_pipeline', default_args=default_args, schedule_interval='@daily')

preprocess = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)
train = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)

preprocess >> train