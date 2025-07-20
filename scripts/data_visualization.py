import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample housing data for visualization"""
    np.random.seed(42)
    n_samples = 1000
    
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.2, 0.4, 0.25, 0.05])
    bathrooms = np.random.choice([1, 1.5, 2, 2.5, 3, 4], n_samples, p=[0.15, 0.1, 0.3, 0.2, 0.2, 0.05])
    sqft_living = np.random.normal(1800 + bedrooms * 300, 500).astype(int)
    sqft_living = np.clip(sqft_living, 500, 5000)
    
    age = np.random.randint(0, 100, n_samples)
    condition = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.15, 0.5, 0.25, 0.05])
    grade = np.random.choice(range(3, 13), n_samples)
    waterfront = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # Create realistic price
    price = (
        bedrooms * 40000 +
        bathrooms * 30000 +
        sqft_living * 130 +
        (100 - age) * 1000 +
        condition * 15000 +
        grade * 35000 +
        waterfront * 200000 +
        np.random.normal(0, 50000, n_samples)
    )
    price = np.maximum(price, 50000)
    
    return pd.DataFrame({
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'age': age,
        'condition': condition,
        'grade': grade,
        'waterfront': waterfront,
        'price': price
    })

def create_visualizations():
    """Create comprehensive visualizations for the housing data"""
    
    print("Creating sample dataset...")
    df = create_sample_data()
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Price distribution
    plt.subplot(4, 3, 1)
    plt.hist(df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of House Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Price ($)')
    plt.ylabel('Frequency')
    plt.ticklabel_format(style='plain', axis='x')
    
    # 2. Price vs Living Area
    plt.subplot(4, 3, 2)
    plt.scatter(df['sqft_living'], df['price'], alpha=0.6, color='coral')
    plt.title('Price vs Living Area', fontsize=14, fontweight='bold')
    plt.xlabel('Living Area (sq ft)')
    plt.ylabel('Price ($)')
    plt.ticklabel_format(style='plain', axis='y')
    
    # Add trend line
    z = np.polyfit(df['sqft_living'], df['price'], 1)
    p = np.poly1d(z)
    plt.plot(df['sqft_living'], p(df['sqft_living']), "r--", alpha=0.8)
    
    # 3. Price by Bedrooms
    plt.subplot(4, 3, 3)
    bedroom_prices = df.groupby('bedrooms')['price'].mean()
    plt.bar(bedroom_prices.index, bedroom_prices.values, color='lightgreen', alpha=0.8)
    plt.title('Average Price by Number of Bedrooms', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Average Price ($)')
    plt.ticklabel_format(style='plain', axis='y')
    
    # 4. Price by Condition
    plt.subplot(4, 3, 4)
    condition_prices = df.groupby('condition')['price'].mean()
    plt.bar(condition_prices.index, condition_prices.values, color='gold', alpha=0.8)
    plt.title('Average Price by Condition Rating', fontsize=14, fontweight='bold')
    plt.xlabel('Condition (1=Poor, 5=Excellent)')
    plt.ylabel('Average Price ($)')
    plt.ticklabel_format(style='plain', axis='y')
    
    # 5. Age vs Price
    plt.subplot(4, 3, 5)
    plt.scatter(df['age'], df['price'], alpha=0.6, color='mediumpurple')
    plt.title('House Age vs Price', fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)')
    plt.ylabel('Price ($)')
    plt.ticklabel_format(style='plain', axis='y')
    
    # Add trend line
    z = np.polyfit(df['age'], df['price'], 1)
    p = np.poly1d(z)
    plt.plot(df['age'], p(df['age']), "r--", alpha=0.8)
    
    # 6. Waterfront vs Non-waterfront
    plt.subplot(4, 3, 6)
    waterfront_prices = df.groupby('waterfront')['price'].mean()
    labels = ['No Waterfront', 'Waterfront']
    plt.bar(labels, waterfront_prices.values, color=['lightcoral', 'lightblue'], alpha=0.8)
    plt.title('Average Price: Waterfront vs Non-waterfront', fontsize=14, fontweight='bold')
    plt.ylabel('Average Price ($)')
    plt.ticklabel_format(style='plain', axis='y')
    
    # 7. Correlation Heatmap
    plt.subplot(4, 3, 7)
    correlation_matrix = df[['bedrooms', 'bathrooms', 'sqft_living', 'age', 'condition', 'grade', 'price']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 8. Price by Grade
    plt.subplot(4, 3, 8)
    grade_prices = df.groupby('grade')['price'].mean()
    plt.plot(grade_prices.index, grade_prices.values, marker='o', linewidth=2, markersize=6, color='darkgreen')
    plt.title('Average Price by Grade', fontsize=14, fontweight='bold')
    plt.xlabel('Grade (3=Low, 12=High)')
    plt.ylabel('Average Price ($)')
    plt.grid(True, alpha=0.3)
    plt.ticklabel_format(style='plain', axis='y')
    
    # 9. Feature Importance (from a quick model)
    plt.subplot(4, 3, 9)
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'age', 'condition', 'grade', 'waterfront']
    X = df[features]
    y = df['price']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.barh(importance_df['feature'], importance_df['importance'], color='orange', alpha=0.8)
    plt.title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    
    # 10. Price Range Distribution
    plt.subplot(4, 3, 10)
    price_ranges = pd.cut(df['price'], bins=[0, 200000, 400000, 600000, 800000, float('inf')], 
                         labels=['<$200K', '$200K-$400K', '$400K-$600K', '$600K-$800K', '>$800K'])
    price_range_counts = price_ranges.value_counts()
    plt.pie(price_range_counts.values, labels=price_range_counts.index, autopct='%1.1f%%', 
            colors=['lightcoral', 'lightskyblue', 'lightgreen', 'gold', 'plum'])
    plt.title('Distribution of Price Ranges', fontsize=14, fontweight='bold')
    
    # 11. Bathrooms vs Price
    plt.subplot(4, 3, 11)
    bathroom_prices = df.groupby('bathrooms')['price'].mean()
    plt.bar(bathroom_prices.index.astype(str), bathroom_prices.values, color='lightpink', alpha=0.8)
    plt.title('Average Price by Number of Bathrooms', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Bathrooms')
    plt.ylabel('Average Price ($)')
    plt.xticks(rotation=45)
    plt.ticklabel_format(style='plain', axis='y')
    
    # 12. Summary Statistics
    plt.subplot(4, 3, 12)
    plt.axis('off')
    
    stats_text = f"""
    DATASET SUMMARY
    
    Total Houses: {len(df):,}
    
    Price Statistics:
    • Mean: ${df['price'].mean():,.0f}
    • Median: ${df['price'].median():,.0f}
    • Min: ${df['price'].min():,.0f}
    • Max: ${df['price'].max():,.0f}
    
    House Characteristics:
    • Avg Bedrooms: {df['bedrooms'].mean():.1f}
    • Avg Bathrooms: {df['bathrooms'].mean():.1f}
    • Avg Living Area: {df['sqft_living'].mean():,.0f} sq ft
    • Avg Age: {df['age'].mean():.0f} years
    
    Special Features:
    • Waterfront: {(df['waterfront'].sum()/len(df)*100):.1f}%
    """
    
    plt.text(0.1, 0.9, stats_text, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(pad=3.0)
    plt.savefig('housing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations created and saved as 'housing_analysis.png'")
    
    # Print some insights
    print("\n=== KEY INSIGHTS ===")
    print(f"1. Average house price: ${df['price'].mean():,.0f}")
    print(f"2. Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
    print(f"3. Strongest price correlation: {correlation_matrix['price'].abs().sort_values(ascending=False).index[1]} ({correlation_matrix['price'].abs().sort_values(ascending=False).iloc[1]:.3f})")
    print(f"4. Waterfront premium: ${df[df['waterfront']==1]['price'].mean() - df[df['waterfront']==0]['price'].mean():,.0f}")
    print(f"5. Most important feature: {importance_df.iloc[-1]['feature']} ({importance_df.iloc[-1]['importance']:.3f})")

if __name__ == "__main__":
    create_visualizations()
