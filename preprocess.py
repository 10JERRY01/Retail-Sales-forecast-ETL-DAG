import pandas as pd
import numpy as np
from sqlalchemy import create_engine

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
