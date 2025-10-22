import pickle
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# -------------------------
# 1️⃣ Load saved objects
# -------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("logreg.pkl", "rb") as f:
    lr = pickle.load(f)

# -------------------------
# 2️⃣ Column order
# -------------------------
columns = [
    'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'texture_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'texture_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# -------------------------
# 3️⃣ Prediction function
# -------------------------
def predict_tumor(*inputs):
    # Convert inputs to dataframe
    x = pd.DataFrame([inputs], columns=columns)
    
    # Apply log transform to same columns (avoid negative/zero issues)
    for col in columns:
        x[col] = np.log1p(x[col])
    
    # Scale
    x_scaled = scaler.transform(x)
    
    # PCA
    x_pca = pca.transform(x_scaled)
    
    # Predict
    pred = lr.predict(x_pca)[0]
    prob = lr.predict_proba(x_pca)[0][pred]
    
    return "Malignant" if pred == 1 else "Benign", f"Confidence: {prob*100:.2f}%"

# -------------------------
# 4️⃣ Gradio interface
# -------------------------
inputs = [gr.Number(label=col) for col in columns]
outputs = [gr.Textbox(label="Prediction"), gr.Textbox(label="Confidence")]

app = gr.Interface(fn=predict_tumor, inputs=inputs, outputs=outputs, title="Breast Cancer Predictor",
                   description="Enter tumor features to predict if it is malignant or benign.")

# -------------------------
# 5️⃣ Launch app
# -------------------------
if __name__ == "__main__":
    app.launch()
