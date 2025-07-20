import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import json

def create_enhanced_dataset(n_samples=2000):
    """Create a more realistic housing dataset"""
    np.random.seed(42)
    
    # Generate correlated features for more realistic data
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.35, 0.30, 0.12, 0.03])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], n_samples, 
                                p=[0.10, 0.05, 0.25, 0.15, 0.25, 0.15, 0.05])
    
    # Living area correlated with bedrooms
    sqft_living = np.random.normal(
        800 + bedrooms * 400, 
        200 + bedrooms * 50
    ).astype(int)
    sqft_living = np.clip(sqft_living, 500, 8000)
    
    # Lot size with some correlation to living area
    sqft_lot = np.random.normal(
        5000 + sqft_living * 0.5,
        2000
    ).astype(int)
    sqft_lot = np.clip(sqft_lot, 1000, 50000)
    
    floors = np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.4, 0.1, 0.35, 0.1, 0.05])
    waterfront = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    view = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.6, 0.2, 0.1, 0.07, 0.03])
    condition = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.50, 0.25, 0.05])
    grade = np.random.choice(range(3, 13), n_samples)
    
    # Age with realistic distribution
    yr_built = np.random.choice(range(1900, 2024), n_samples)
    age = 2023 - yr_built
    
    # Renovation probability increases with age
    renovated = np.random.binomial(1, np.clip(age / 100, 0.1, 0.4), n_samples)
    
    # Create realistic price based on features
    price_base = (
        bedrooms * 35000 +
        bathrooms * 25000 +
        sqft_living * 120 +
        sqft_lot * 3 +
        floors * 15000 +
        waterfront * 150000 +
        view * 20000 +
        condition * 10000 +
        grade * 35000 +
        age * -500 +
        renovated * 40000 +
        np.random.normal(0, 30000, n_samples)  # Market variation
    )
    
    # Add location-based price multipliers
    location_multiplier = np.random.choice([0.8, 0.9, 1.0, 1.1, 1.2, 1.4], n_samples, 
                                         p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])
    
    price = price_base * location_multiplier
    price = np.maximum(price, 50000)  # Minimum price
    
    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'age': age,
        'renovated': renovated,
        'price': price
    }
    
    return pd.DataFrame(data)

def train_and_evaluate_model():
    """Train and evaluate the house price prediction model"""
    
    print("Creating enhanced dataset...")
    df = create_enhanced_dataset(2000)
    
    print(f"Dataset created with {len(df)} samples")
    print("\nDataset statistics:")
    print(df.describe())
    
    # Define features
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
               'waterfront', 'view', 'condition', 'grade', 'age', 'renovated']
    
    X = df[features]
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluate model
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n=== MODEL PERFORMANCE ===")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"Mean Price: ${y_test.mean():,.2f}")
    print(f"RMSE as % of mean price: {(rmse/y_test.mean()*100):.1f}%")
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== FEATURE IMPORTANCE ===")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']:<15}: {row['importance']:.4f}")
    
    # Save model and metadata
    print("\nSaving model...")
    joblib.dump(rf_model, 'house_price_model.pkl')
    
    # Save model metadata
    model_metadata = {
        'model_type': 'RandomForestRegressor',
        'features': features,
        'performance': {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'cv_mean_r2': float(cv_scores.mean()),
            'cv_std_r2': float(cv_scores.std())
        },
        'feature_importance': {
            row['feature']: float(row['importance']) 
            for _, row in feature_importance.iterrows()
        },
        'training_info': {
            'n_samples': len(df),
            'n_features': len(features),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print("Model and metadata saved successfully!")
    
    # Sample predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        features_sample = X_test.iloc[idx]
        
        print(f"\nSample {i+1}:")
        print(f"  Actual: ${actual:,.0f}")
        print(f"  Predicted: ${predicted:,.0f}")
        print(f"  Error: ${abs(actual - predicted):,.0f} ({abs(actual - predicted)/actual*100:.1f}%)")
        print(f"  Features: {dict(features_sample)}")
    
    return rf_model, model_metadata

if __name__ == "__main__":
    model, metadata = train_and_evaluate_model()
    print("\nModel training completed successfully!")
