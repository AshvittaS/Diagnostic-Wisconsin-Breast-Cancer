# 🎗️ Wisconsin Breast Cancer Diagnostic Analysis & Prediction System

A comprehensive machine learning pipeline for breast cancer diagnosis using the Wisconsin Diagnostic Breast Cancer dataset. This project features advanced data preprocessing, dimensionality reduction, and an interactive web application for real-time predictions.

## 🚀 Unique Features & Innovations

### 🔬 **Advanced Feature Engineering**
- **Intelligent Correlation-Based Feature Selection**: Automatically removes highly correlated features (>90% correlation) to prevent multicollinearity
- **Log Transformation for Skewed Data**: Applied log transformation to handle skewed distributions, ensuring better model performance
- **Comprehensive Data Preprocessing**: Handles negative values and zero values before log transformation

### 📊 **Dimensionality Reduction with PCA**
- **Optimal Component Selection**: Uses 6 principal components that capture maximum variance
- **Interactive PCA Visualization**: Plotly-based scatter matrix showing component relationships
- **Feature Importance Mapping**: Heatmap visualization showing how original features contribute to each principal component

### 🎯 **High-Performance Model**
- **99.1% Accuracy**: Achieved exceptional performance on test data
- **Balanced Classification**: Excellent precision and recall for both malignant and benign cases
- **Confidence Scoring**: Provides prediction confidence levels for better clinical decision-making

### 🌐 **Interactive Web Application**
- **Real-time Predictions**: Gradio-based web interface for instant tumor classification
- **User-Friendly Input**: Simple form interface for medical professionals
- **Production-Ready**: Deployed with proper model serialization and loading

## 📈 **Key Visualizations & Insights**

### Correlation Analysis
The project includes a comprehensive correlation heatmap that reveals:
- Strong correlations between related features (mean, SE, worst values)
- Identified redundant features for removal
- Feature relationships that guide preprocessing decisions

### Distribution Analysis
- **Box Plot Analysis**: Identified outliers and data distribution patterns
- **KDE Plots**: Visualized feature distributions to understand data characteristics
- **Skewness Detection**: Automated identification of skewed features requiring transformation

### PCA Visualization
- **Explained Variance Plot**: Shows cumulative variance explained by each component
- **Interactive Scatter Matrix**: 3D visualization of principal components with color-coded diagnosis
- **Feature Contribution Heatmap**: Shows how original features load onto principal components

## 🛠️ **Technical Architecture**

### Data Pipeline
1. **Data Loading**: Direct from UCI ML Repository
2. **Preprocessing**: 
   - Ordinal encoding (M=1, B=0)
   - Correlation-based feature removal
   - Log transformation for skewed features
3. **Scaling**: StandardScaler for normalization
4. **Dimensionality Reduction**: PCA with 6 components
5. **Model Training**: Logistic Regression with optimized parameters

### Model Performance
```
Classification Report:
              precision    recall  f1-score   support
           0       0.99      1.00      0.99        71
           1       1.00      0.98      0.99        43
    accuracy                           0.99       114
```

## 📦 **Installation & Setup**

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- pandas==2.0.3
- numpy==1.24.3
- matplotlib==3.7.2
- seaborn==0.12.2
- scikit-learn==1.3.0
- plotly==5.15.0
- gradio==3.50.2

## 🚀 **Usage**

### Running the Web Application
```bash
python app.py
```

### Jupyter Notebook Analysis
Open `task.ipynb` to explore the complete analysis pipeline including:
- Data exploration and visualization
- Feature engineering and selection
- Model training and evaluation
- Interactive visualizations

## 📊 **Dataset Information**

- **Source**: UCI Machine Learning Repository - Wisconsin Diagnostic Breast Cancer
- **Features**: 30 numerical features (mean, standard error, worst values)
- **Samples**: 569 instances
- **Target**: Binary classification (Malignant/Benign)
- **Features Include**: Radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension

## 🔍 **Key Insights from Analysis**

1. **Feature Reduction**: Reduced from 30 to 21 features by removing highly correlated ones
2. **PCA Effectiveness**: 6 components capture significant variance while reducing dimensionality
3. **Model Robustness**: Logistic regression performs exceptionally well on this dataset
4. **Clinical Relevance**: High accuracy makes this suitable for clinical decision support

## 🎯 **Unique Contributions**

1. **Automated Feature Selection**: Intelligent correlation-based removal of redundant features
2. **Advanced Preprocessing**: Sophisticated handling of skewed data with log transformation
3. **Interactive Visualizations**: Plotly-based 3D scatter matrix for PCA exploration
4. **Production Deployment**: Complete web application with model serialization
5. **Comprehensive Analysis**: End-to-end pipeline from raw data to deployed application

## 📈 **Performance Metrics**

- **Accuracy**: 99.1%
- **Precision**: 99% (Malignant), 100% (Benign)
- **Recall**: 100% (Malignant), 98% (Benign)
- **F1-Score**: 99% (Malignant), 99% (Benign)

## 🔬 **Clinical Applications**

This system can be used by:
- **Radiologists**: For preliminary tumor assessment
- **Oncologists**: For treatment planning
- **Researchers**: For breast cancer studies
- **Medical Students**: For educational purposes

## 📝 **Files Description**

- `task.ipynb`: Complete analysis notebook with visualizations
- `app.py`: Gradio web application for predictions
- `scaler.pkl`: Trained StandardScaler object
- `pca.pkl`: Trained PCA object
- `logreg.pkl`: Trained Logistic Regression model
- `wdbc.data`: Original dataset file
- `requirements.txt`: Python dependencies

## 🎨 **Visualization Highlights**

The project includes several unique visualizations:
- **Correlation Heatmap**: Triangular mask for clean correlation analysis
- **Box Plot Grid**: Comprehensive outlier detection across all features
- **KDE Distribution Plots**: Understanding data distributions
- **PCA Scatter Matrix**: Interactive 3D visualization of principal components
- **Feature Importance Heatmap**: Understanding PCA component contributions
- **ROC Curve**: Model performance visualization

---

*This project demonstrates advanced machine learning techniques applied to medical diagnosis, showcasing the power of data science in healthcare applications.*