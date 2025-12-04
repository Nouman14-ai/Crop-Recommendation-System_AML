import streamlit as st
import pandas as pd
import pickle

# --------------------------
#  PAGE SETTINGS
# --------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# --------------------------
#  CUSTOM CSS (Professional Look)
# --------------------------
st.markdown("""
<style>
/* Main background */
.main {
    background-color: #F5FAF3;
}

/* Card style */
.card {
    padding: 25px;
    background: white;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}

/* Title */
.title-text {
    font-size: 40px;
    font-weight: 700;
    color: #2E7D32;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
#  HEADER
# --------------------------
st.markdown('<p class="title-text">🌾 Crop Recommendation Dashboard</p>', unsafe_allow_html=True)
st.write("A smart machine learning app that recommends the **best crop** for your soil conditions.")

# --------------------------
#  SIDEBAR INPUT
# --------------------------
st.sidebar.header("🔧 Input Soil Parameters")

N = st.sidebar.number_input("Nitrogen (N)", 0, 200, 50)
P = st.sidebar.number_input("Phosphorus (P)", 0, 200, 50)
K = st.sidebar.number_input("Potassium (K)", 0, 200, 50)

ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0, 500, 100)
temperature = st.sidebar.number_input("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.number_input("Humidity (%)", 0, 100, 60)

predict_btn = st.sidebar.button("🌱 Recommend Crop")

# --------------------------
#  LOAD MODEL
# --------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
except:
    model = None

# --------------------------
#  TABS SECTION
# --------------------------
tab1, tab2 = st.tabs(["📌 Crop Recommendation", "📁 Dataset"])

# --------------------------
#  TAB 1: RECOMMENDATION
# --------------------------
with tab1:

    st.subheader("🌱 Recommended Crop")

    if predict_btn:
        if model is None:
            st.error("Model file not found! Upload model.pkl.")
        else:
            input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
            prediction = model.predict(input_data)[0]

            # Modern Card Output
            st.markdown(f"""
            <div class="card">
                <h2>🌟 Best Crop to Grow: <br><span style="color:#1B5E20">{prediction}</span></h2>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Click **Recommend Crop** to get the best crop.")

# --------------------------
#  TAB 2: DATASET
# --------------------------
with tab2:
    st.subheader("📂 Uploaded Dataset Preview")

    try:
        df = pd.read_csv("Crop_recommendation.csv")
        st.dataframe(df, use_container_width=True)
    except:
        st.warning("Dataset not found! Upload Crop_recommendation.csv")
