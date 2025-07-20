# 🏠 House Price Predictor

A full-stack machine learning application that predicts house prices using advanced algorithms and provides an intuitive web interface for real estate analysis.

![House Price Predictor](https://img.shields.io/badge/ML-House%20Price%20Prediction-blue)
![Next.js](https://img.shields.io/badge/Next.js-13+-black)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue)

## 🌟 Features

### 🎯 **Core Functionality**
- **Real-time Price Predictions** - Get instant house price estimates
- **Interactive Web Interface** - User-friendly form with 11+ input parameters
- **Confidence Intervals** - Price ranges with statistical confidence
- **Feature Importance Analysis** - Understand what drives house prices
- **Responsive Design** - Works seamlessly on desktop and mobile

### 📊 **Advanced Analytics**
- **Model Performance Dashboard** - View R² scores, RMSE, and accuracy metrics
- **Feature Correlation Analysis** - Understand relationships between variables
- **Price Range Distribution** - Market segment analysis
- **Comparative Analysis** - Compare different property configurations

### 🔧 **Technical Features**
- **Random Forest ML Model** - 85%+ accuracy with cross-validation
- **RESTful API** - Clean API endpoints for predictions
- **Data Visualization** - Comprehensive charts and graphs
- **Error Handling** - Robust input validation and error management

## 🛠️ Tech Stack

### **Frontend**
- **Next.js 13+** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Modern UI components
- **Lucide React** - Beautiful icons

### **Backend**
- **Next.js API Routes** - Serverless API endpoints
- **Python** - Data science and ML model training
- **scikit-learn** - Machine learning algorithms
- **pandas & numpy** - Data manipulation and analysis

### **Data Science**
- **Random Forest Regressor** - Primary ML model
- **matplotlib & seaborn** - Data visualization
- **joblib** - Model serialization
- **Cross-validation** - Model validation techniques

## 🚀 Getting Started

### Prerequisites

```bash
node >= 18.0.0
python >= 3.8
npm or yarn
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/house-price-predictor.git
cd house-price-predictor
```

2. **Install dependencies**
```bash
npm install
# or
yarn install
```

3. **Install Python dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

4. **Run the development server**
```bash
npm run dev
# or
yarn dev
```

5. **Open your browser**
Navigate to [http://localhost:3000](http://localhost:3000)

## 📖 Usage Guide

### 🏡 **Making Predictions**

1. **Navigate to the Prediction Tab**
2. **Enter House Details:**
   - Number of bedrooms and bathrooms
   - Living area and lot size (sq ft)
   - Number of floors
   - Property age and condition
   - Special features (waterfront, view, renovated)

3. **Click "Predict House Price"**
4. **View Results:**
   - Estimated market value
   - Price confidence interval
   - Price per square foot
   - Key property highlights

### 📊 **Model Analysis**

1. **Switch to the Analysis Tab**
2. **Explore Model Performance:**
   - Accuracy metrics (R² Score, RMSE, MAE)
   - Feature importance rankings
   - Training dataset information



## 🔌 API Documentation

### **POST /api/predict**

Predicts house price based on input features.

**Request Body:**
```json
{
  "bedrooms": 3,
  "bathrooms": 2,
  "sqft_living": 2000,
  "sqft_lot": 5000,
  "floors": 1,
  "waterfront": 0,
  "view": 0,
  "condition": 3,
  "grade": 7,
  "age": 20,
  "renovated": 0
}
```

**Response:**
```json
{
  "predicted_price": 450000,
  "confidence_interval": [382500, 517500],
  "features_used": { ... }
}
```

## 🤖 Machine Learning Model

### **Model Details**
- **Algorithm:** Random Forest Regressor
- **Features:** 11 key house characteristics
- **Accuracy:** 85%+ R² Score
- **Validation:** 5-fold cross-validation
- **Training Data:** 2000+ synthetic house records

### **Key Features (by importance)**
1. **Living Area (sq ft)** - 35%
2. **Grade/Quality** - 18%
3. **Number of Bathrooms** - 12%
4. **Property Age** - 10%
5. **Number of Bedrooms** - 8%
6. **Condition Rating** - 7%
7. **View Quality** - 5%
8. **Waterfront Access** - 5%

### **Model Performance**
- **R² Score:** 0.85+ (85%+ variance explained)
- **RMSE:** ~$45,000 (Root Mean Square Error)
- **MAE:** ~$32,000 (Mean Absolute Error)

## 📊 Data Science Scripts

### **Running Analysis Scripts**

1. **Data Analysis & Model Training:**
```bash
python scripts/data_analysis.py
```

2. **Enhanced Model Training:**
```bash
python scripts/model_training.py
```

3. **Generate Visualizations:**
```bash
python scripts/data_visualization.py
```

## 🎨 Screenshots
<img width="1049" height="655" alt="Screenshot From 2025-07-19 23-54-14" src="https://github.com/user-attachments/assets/831ae5dd-a8f9-4c2e-adc5-cda790e00488" />
<img width="1049" height="655" alt="Screenshot From 2025-07-19 23-57-20" src="https://github.com/user-attachments/assets/eccda832-377a-4192-bc8f-b0e2ee1b5ee0" />



### Main Prediction Interface
- Clean, intuitive form for entering house details
- Real-time validation and user feedback
- Responsive design for all screen sizes

### Results Dashboard
- Prominent price display with confidence intervals
- Key property highlights and price per sq ft
- Professional styling with clear information hierarchy

### Model Analysis
- Performance metrics and feature importance
- Interactive charts and visualizations
- Technical details for data science enthusiasts

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**  
- [GitHub](https://github.com/fahadhasan93/)  
- [LinkedIn](https://www.linkedin.com/in/md-fahad-hasan-61720a350/)



## 🙏 Acknowledgments

- **scikit-learn** team for excellent ML libraries
- **Next.js** team for the amazing React framework
- **shadcn/ui** for beautiful UI components



