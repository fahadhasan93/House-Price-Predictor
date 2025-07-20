import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import json

# Load the dataset
print("Loading dataset...")
# For demo purposes, we'll create a sample dataset structure
# In real implementation, you would load from the kaggle dataset

# Create sample data structure based on common housing features
np.random.seed(42)
n_samples = 1000

data = {
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'sqft_living': np.random.randint(500, 5000, n_samples),
    'sqft_lot': np.random.randint(1000, 20000, n_samples),
    'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
    'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'view': np.random.randint(0, 5, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'grade': np.random.randint(3, 13, n_samples),
    'yr_built': np.random.randint(1900, 2023, n_samples),
    'yr_renovated': np.random.choice([0] + list(range(1950, 2023)), n_samples, p=[0.7] + [0.3/73]*73),
    'zipcode': np.random.choice([98001, 98002, 98003, 98004, 98005], n_samples),
    'lat': np.random.uniform(47.1, 47.8, n_samples),
    'long': np.random.uniform(-122.5, -121.3, n_samples),
}

# Create price based on features (realistic relationship)
price = (
    data['bedrooms'] * 50000 +
    data['bathrooms'] * 30000 +
    data['sqft_living'] * 150 +
    data['sqft_lot'] * 5 +
    data['floors'] * 20000 +
    data['waterfront'] * 200000 +
    data['view'] * 25000 +
    data['condition'] * 15000 +
    data['grade'] * 40000 +
    (2023 - data['yr_built']) * -1000 +
    np.where(data['yr_renovated'] > 0, 50000, 0) +
    np.random.normal(0, 50000, n_samples)  # Add noise
)

data['price'] = np.maximum(price, 50000)  # Minimum price of $50k

df = pd.DataFrame(data)

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())

print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

# Data cleaning
print("\nChecking for missing values:")
print(df.isnull().sum())

# Feature engineering
df['age'] = 2023 - df['yr_built']
df['renovated'] = (df['yr_renovated'] > 0).astype(int)
df['price_per_sqft'] = df['price'] / df['sqft_living']

# Prepare features for modeling
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
           'waterfront', 'view', 'condition', 'grade', 'age', 'renovated']

X = df[features]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\nTraining Linear Regression model...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
lr_pred = lr_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

# Evaluate models
print("\n=== MODEL EVALUATION ===")
print("\nLinear Regression:")
print(f"R² Score: {r2_score(y_test, lr_pred):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, lr_pred)):,.2f}")
print(f"MAE: ${mean_absolute_error(y_test, lr_pred):,.2f}")

print("\nRandom Forest:")
print(f"R² Score: {r2_score(y_test, rf_pred):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, rf_pred)):,.2f}")
print(f"MAE: ${mean_absolute_error(y_test, rf_pred):,.2f}")

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(feature_importance)

# Save models and scaler
print("\nSaving models...")
joblib.dump(rf_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Save feature names and model info
model_info = {
    'features': features,
    'model_type': 'RandomForestRegressor',
    'r2_score': float(r2_score(y_test, rf_pred)),
    'rmse': float(np.sqrt(mean_squared_error(y_test, rf_pred))),
    'mae': float(mean_absolute_error(y_test, rf_pred))
}

with open('model_info.json', 'w') as f:
    json.dump(model_info, f)

print("Models saved successfully!")
print("Analysis complete!")
