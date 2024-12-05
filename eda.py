import matplotlib.pyplot as plt
import pandas as pd

# Load cleaned dataset
df = pd.read_csv('cleaned_retail_sales.csv')
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
