from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import joblib

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


