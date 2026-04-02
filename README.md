# 🎗️ Wisconsin Breast Cancer Diagnostic Analysis & Prediction System

A comprehensive machine learning pipeline for breast cancer diagnosis using the Wisconsin Diagnostic Breast Cancer dataset. This project features advanced data preprocessing, dimensionality reduction, and an interactive web application for real-time predictions.

## 🚀 Features

- **Advanced Feature Engineering**: Intelligent correlation-based feature selection, log transformation for skewed data, and comprehensive preprocessing.
- **Dimensionality Reduction**: PCA with optimal component selection and interactive visualizations.
- **High-Performance Model**: 99.1% accuracy with balanced classification and confidence scoring.
- **Interactive Web App**: Gradio-based interface for real-time tumor classification.

## 🐳 How to Execute Using Docker

### Prerequisites
- Docker installed on your system

### Build the Docker Image
Run the following command in the project directory to build the Docker image:

```bash
docker build -t breast_cancer .
```

### Run the Application
After building the image, start the container:

```bash
docker run -p 7860:7860 breast_cancer
```

The application will be available at `http://localhost:7860` in your web browser.

### Stopping the Container
To stop the running container, use `Ctrl+C` in the terminal or run:

```bash
docker stop <container_id>
```

## 📊 Project Insights

- **Correlation Analysis**: Heatmap revealing feature relationships and redundant features.
- **Distribution Analysis**: Box plots, KDE plots, and skewness detection.
- **PCA Visualization**: Explained variance plots, interactive scatter matrices, and feature contribution heatmaps.

## 🛠️ Technical Stack

- **Python Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, plotly, gradio
- **Machine Learning**: PCA, classification models
- **Web Framework**: Gradio for interactive UI
- **Containerization**: Docker for easy deployment
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
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- plotly
- gradio

## 🚀 **Usage**

### 🌐 **Live Demo**
**Try the application online**: [Breast Cancer Predictor](https://ashvitta07-breast-cancer-classification.hf.space/?logs=container&__theme=system&deep_link=Lwbhu7kdJSM)

The deployed application allows you to:
- Enter tumor features through an intuitive interface
- Get instant predictions (Malignant/Benign)
- View confidence scores for each prediction
- Share results via direct links

### Running Locally
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
5. **Cloud Deployment**: Successfully deployed on Hugging Face Spaces for global access
6. **Comprehensive Analysis**: End-to-end pipeline from raw data to deployed application

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