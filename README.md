# Retail-Sales-forecast-ETL-DAG
# Project Overview
This project focuses on processing and analyzing retail sales data to derive actionable insights and build predictive models.
The project involves data preprocessing, exploratory data analysis (EDA), machine learning, and automation using industry-standard tools and techniques. 
The primary goals include:

# ETL Pipeline: Extract, Transform, and Load data into a cloud database.
EDA: Identify trends, customer behavior, and product performance.
Predictive Analytics:
Forecast sales using regression models.
Segment customers using clustering.
Automation: Automate data pipeline and model retraining with Airflow.
Key technologies include PySpark, Python (scikit-learn, pandas, matplotlib), Redshift, and Airflow.


# Configure Cloud Storage

Create an S3 bucket or Azure Blob Storage container for raw data.

# Run ETL Pipeline

python preprocess.py
This script cleans the raw data and loads it into the configured database.

# Run Exploratory Data Analysis

python eda.py
Generates visualizations and summary statistics saved as PNGs in the output/ directory.

# Train Models

python train_model.py
Trains the regression and clustering models and saves it.

# Set Up Airflow for Automation

Place the DAG (retail_pipeline.py) in your Airflow DAGs folder.
Start Airflow:

airflow webserver
airflow scheduler
# Usage
Access model predictions via the CLI or integrate the pipeline into a web app using Flask/Django.
View insights and visualizations in the output/ folder or a Tableau dashboard.
Insights from EDA
1. Sales Trends
Insight: Sales peak during weekends and holiday seasons, with a notable increase in electronics and clothing categories.
2. Customer Behavior
Males tend to purchase electronics, while females show higher spending on beauty and clothing products.
3. Product Performance
Insight: Electronics contribute 34.2% of revenue but have a lower frequency of purchase compared to clothing.
Model Evaluation Metrics
1. Sales Forecasting
Model: Gradient Boosting Regressor
Performance:
Root Mean Squared Error (RMSE): 527.46
2. Customer Segmentation
Model: K-Means Clustering
Clusters Identified:
Cluster 0: Budget Shoppers (low spending, frequent visits)
Cluster 1: High-Value Customers (high spending, electronics-focused)
Cluster 2: Seasonal Shoppers (sporadic but high-value transactions)
3. Recommendations
Use customer segments to tailor marketing campaigns.
Optimize inventory for top-selling categories during peak seasons.

# Future Work
Feature Expansion: Include external data like weather or holidays.
Real-Time Analysis: Integrate Apache Kafka for streaming data.
Cloud Deployment: Deploy the project on AWS Lambda with API Gateway for serving models.
