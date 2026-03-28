import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from utils import preprocess_image   # ✅ use only preprocessing from utils

# --- Page Config ---
st.set_page_config(
    page_title="Fake ID Card Detector",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        .stApp {
            background-color: #FFFFFF;
            color: #2C2C2C;
        }

        h1, h2, h3, h4 {
            color: #8B0000 !important;
            font-family: 'Arial Black', sans-serif;
        }

        div.stButton > button {
            background-color: #FF7F50;
            color: white;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 24px;
            border: none;
        }

        div.stButton > button:hover {
            background-color: #E67348;
        }

        .stSuccess, .stError {
            font-size: 20px !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("🪪 Fake ID Card Detector")
st.markdown("### Upload an ID card image to check if it's **Real or Fake**")

# --- Load Model ---
@st.cache_resource
def load_my_model():
    try:
        return load_model("model.h5")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

model = load_my_model()

# --- Prediction Function ---
def predict(model, img_pil):
    img_array = preprocess_image(img_pil)  # from utils (150x150)
    prediction = model.predict(img_array)[0][0]
    return prediction

# --- Main App ---
if model:
    uploaded_file = st.file_uploader("📂 Choose an ID Card Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded ID", use_column_width=True)

        img_pil = Image.open(uploaded_file).convert('RGB')

        if st.button("🔍 Predict"):
            with st.spinner("🔎 Analyzing image..."):

                cnn_score = predict(model, img_pil)

                st.markdown("---")
                st.subheader("🧠 CNN Analysis")

                # Score display
                st.markdown(f"**Prediction Score:** `{cnn_score:.4f}`")

                # Confidence
                confidence = cnn_score if cnn_score > 0.5 else (1 - cnn_score)

                st.progress(int(confidence * 100))
                st.markdown(f"📊 **Confidence:** {confidence * 100:.2f}%")

                # Result
                if cnn_score > 0.5:
                    st.success("✅ Likely a **Real ID**")
                else:
                    st.error("❌ Likely a **Fake ID**")