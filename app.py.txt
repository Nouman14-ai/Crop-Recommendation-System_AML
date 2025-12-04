import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("Crop_recommendation.csv")

# Label Encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Features & Target
X = df.drop('label', axis=1)
y = df['label']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
model = GaussianNB()
model.fit(X_scaled, y)

# -------------------------------
# STREAMLIT DASHBOARD
# -------------------------------
st.set_page_config(page_title="Crop Recommendation System", layout="wide")

st.title("🌱 Crop Recommendation Dashboard")
st.markdown("This dashboard predicts the most suitable **crop** based on soil & weather conditions.")

# Sidebar info
st.sidebar.header("📌 Input Parameters")

# Sidebar Input Fields
N = st.sidebar.slider("Nitrogen (N)", 0, 150, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 150, 50)
K = st.sidebar.slider("Potassium (K)", 0, 150, 50)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

if st.sidebar.button("Predict Crop"):
    
    # Prepare input
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled = scaler.transform(data)
    
    # Prediction
    prediction = model.predict(scaled)[0]
    crop_name = le.inverse_transform([prediction])[0]

    st.subheader("🌾 Recommended Crop")
    st.success(f"👉 **{crop_name}**")

    # Probabilities
    st.subheader("📊 Model Confidence (Probability %)")
    probs = model.predict_proba(scaled)[0]
    prob_df = pd.DataFrame({
        "Crop Name": le.classes_,
        "Probability (%)": np.round(probs * 100, 2)
    }).sort_values(by="Probability (%)", ascending=False)

    st.dataframe(prob_df)

else:
    st.warning("🔽 Enter values in the sidebar and click **Predict Crop**")

# Display dataset preview
with st.expander("📘 View Dataset"):
    st.dataframe(df.head())
