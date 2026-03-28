import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Page Config ---
st.set_page_config(
    page_title="Fake ID Card Detector",
    layout="centered"
)

# --- Custom CSS for Light Peach Background ---
st.markdown("""
    <style>
        /* App background */
        .stApp {
            background-color: #FFFFFF; /* White */
            color: #2C2C2C;
        }

        /* Title and headers */
        h1, h2, h3, h4, h5, h6 {
            color: #8B0000 !important; /* Dark Red */
            font-family: 'Arial Black', sans-serif;
        }

        /* Buttons */
        div.stButton > button {
            background-color: #FF7F50; /* Coral */
            color: white;
            border-radius: 12px;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 24px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #E67348;
            color: #fff;
        }

        /* Prediction boxes */
        .stSuccess {
            background-color: #DFF2BF !important;
            color: #4F8A10 !important;
            font-size: 20px !important;
            font-weight: bold;
            border-radius: 10px;
        }
        .stError {
            background-color: #FFBABA !important;
            color: #D8000C !important;
            font-size: 20px !important;
            font-weight: bold;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- UI Title ---
st.title(" Fake ID Card Detector")
st.markdown("### Upload an ID card image to check if it's **Real or Fake**")

# --- Model Loading ---
@st.cache_resource
def load_my_model():
    try:
        model = load_model("model.h5")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model. Ensure 'model.h5' exists. Details: {e}")
        return None

# --- Prediction Function ---
def predict_cnn(img, model):
    img_resized = img.resize((150, 150))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return prediction

# --- Main Logic ---
cnn_model = load_my_model()

if cnn_model:
    uploaded_file = st.file_uploader("📂 Choose an ID Card Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded ID", use_column_width=True)
        
        img_pil = Image.open(uploaded_file).convert('RGB')
        
        if st.button("🔍 Predict"):
            with st.spinner("🔎 Analyzing image..."):
                cnn_score = predict_cnn(img_pil, cnn_model)
                
                st.markdown("---")
                st.subheader("🧠 CNN Analysis")
                st.markdown(f"**Prediction Score:** `{cnn_score:.4f}`")

                # Probability Bar 
                # And Displaying prediction confidence score
                st.progress(int(cnn_score * 100))  
                st.markdown(f"**Confidence: {cnn_score*100:.2f}% Real**")

                if cnn_score > 0.5:
                    st.success("✅ Likely a **Real ID**")
                else:
                    st.error("❌ Likely a **Fake ID**")