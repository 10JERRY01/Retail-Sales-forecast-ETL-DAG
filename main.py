import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import pickle

# Load dataset
df = pd.read_csv('retail_sales_dataset.csv')

# Data Cleaning
df['Date'] = pd.to_datetime(df['Date'])
df['Age'] = df['Age'].fillna(df['Age'].mean())  # Replace missing ages with mean
df = pd.get_dummies(df, columns=['Gender', 'Product Category'], drop_first=True)

# Feature Engineering
df['Total Amount'] = df['Quantity'] * df['Price per Unit']

# Save to local
df.to_csv('cleaned_retail_sales.csv', index=False)

'''
# Save to Redshift
engine = create_engine('postgresql+psycopg2://username:password@redshift-endpoint/dbname')
df.to_sql('retail_sales', engine, if_exists='replace', index=False)
print("Data uploaded to Redshift!")
'''

import matplotlib.pyplot as plt

# Sales trends over time
df.groupby('Date')['Total Amount'].sum().plot(title='Sales Over Time', figsize=(10, 6))
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# Average spending by age
df.groupby('Age')['Total Amount'].mean().plot(kind='bar', title='Average Spending by Age', figsize=(10, 6))
plt.xlabel('Age')
plt.ylabel('Avg Spending')
plt.show()

# Product category distribution
df['Product Category_Electronics'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Product Category Share', figsize=(8, 8))
plt.show()


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('cleaned_retail_sales.csv')

# Selected features and target variable
features = ['Age', 'Quantity', 'Gender_Male', 'Product Category_Clothing', 'Product Category_Electronics']
target = 'Total Amount'

# Splitting the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Confirm shapes
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))

# Save the model
joblib.dump(model, "total_sales_forecasting_model.pkl")
print("Model saved as total_sales_forecasting_model.pkl.")

from sklearn.cluster import KMeans
import seaborn as sns

# Selecting features for clustering
features = df[['Age', 'Total Amount']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# Visualize clusters
sns.scatterplot(data=df, x='Age', y='Total Amount', hue='Cluster', palette='viridis')
plt.title('Customer Segmentation')
plt.show()


'''
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import os

def preprocess_data():
    os.system("python preprocess.py")  # Preprocessing script

def train_model():
    os.system("python train_model.py")  # Training script

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

dag = DAG('retail_pipeline', default_args=default_args, schedule_interval='@daily')

preprocess = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)
train = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)

preprocess >> train
'''