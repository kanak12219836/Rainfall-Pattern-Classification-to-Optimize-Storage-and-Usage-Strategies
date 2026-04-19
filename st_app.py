import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Config
MODEL_PATH = "models_trained/xception_model.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['heavy', 'light', 'medium']

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Pre-Process
def preprocess(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    img = np.array(image) / 255.0
    return np.expand_dims(img, axis=0)

# UI
st.set_page_config(page_title="Rainfall Classifier", layout="wide")

st.title("🌧️ Rainfall Pattern Classification")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)

with col2:
    st.subheader("Prediction Result")

    if uploaded:
        x = preprocess(image)
        preds = model.predict(x)[0]

        idx = int(np.argmax(preds))
        conf = float(np.max(preds))

        st.markdown(
            f"""
            ### 🧠 Prediction
            <div style="
                padding:15px;
                border-radius:12px;
                background-color:#1f1f2e;
                color:white;
                font-size:20px;
                font-weight:bold;
                text-align:center;">
                {CLASS_NAMES[idx].upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        st.metric("Confidence", f"{conf:.2%}")

        st.write("### Probability Breakdown")
        for i, c in enumerate(CLASS_NAMES):
            st.progress(float(preds[i]), text=f"{c}: {preds[i]:.3f}")