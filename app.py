import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --------------------------
# PAGE SETTINGS
# --------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# --------------------------
# CUSTOM CSS (Professional Look)
# --------------------------
st.markdown("""
<style>
.main { background-color: #F5FAF3; }
.card {
    padding: 28px;
    background: black;
    border-radius: 14px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    text-align: center;
}
.title-text { font-size: 34px; font-weight:700; color:#2E7D32; margin-bottom:6px; }
.small { color:#555; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# HEADER
# --------------------------
st.markdown('<div class="title-text">🌾 Crop Recommendation Dashboard</div>', unsafe_allow_html=True)
st.write("Enter soil & weather parameters on the left and click **Recommend Crop**.")

# --------------------------
# SIDEBAR INPUT
# --------------------------
st.sidebar.header("🔧 Input Soil Parameters")
N = st.sidebar.number_input("Nitrogen (N)", 0, 200, 50)
P = st.sidebar.number_input("Phosphorus (P)", 0, 200, 50)
K = st.sidebar.number_input("Potassium (K)", 0, 200, 50)
ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 60.0)
predict_btn = st.sidebar.button("🌱 Recommend Crop")

# --------------------------
# LOAD model.pkl (pipeline + label encoder)
# --------------------------
@st.cache_data
def load_artifact():
    try:
        with open("model.pkl", "rb") as f:
            artifact = pickle.load(f)
        return artifact
    except Exception as e:
        return None

artifact = load_artifact()

# --------------------------
# TABS
# --------------------------
tab1, tab2 = st.tabs(["📌 Recommendation", "📁 Dataset"])

with tab1:
    st.subheader("🌱 Recommended Crop")
    if predict_btn:
        if artifact is None:
            st.error("Model artifact not found. Make sure `model.pkl` exists in the repo.")
        else:
            pipeline = artifact["pipeline"]
            le = artifact["label_encoder"]

            input_arr = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            pred_numeric = pipeline.predict(input_arr)[0]
            crop_name = le.inverse_transform([pred_numeric])[0]

            # Single-crop stylish card
            st.markdown(f"""
            <div class="card">
                <h2 style="margin-bottom:6px;">🌟 Recommended Crop</h2>
                <h1 style="color:#145A32; margin-top:0; margin-bottom:6px;">{crop_name}</h1>
                <p class="small">Predicted using a trained Naive Bayes model</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Change parameters on the left and click **Recommend Crop** to predict.")

with tab2:
    st.subheader("📂 Dataset Preview")
    try:
        df = pd.read_csv("Crop_recommendation.csv")
        st.dataframe(df.head(200), use_container_width=True)
    except Exception:
        st.warning("Crop_recommendation.csv not found in the app folder.")


