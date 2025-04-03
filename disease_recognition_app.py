import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("Skin_Disease_Recognition.h5")

# Page settings
st.set_page_config(page_title="Skin Disease Detector", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .main {
            background-color: white;
            padding: 2rem;
            
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🧴 Skin Disease Detection")
st.write("Upload an image of your skin, and the model will predict the disease.")

# File uploader
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0

    if img_array.shape[-1] == 4:  # Remove alpha channel if present
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Class labels (update as per your model)
    labels = ['Acne', 'Rosacea']

    st.success(f"🩺 Prediction: {labels[predicted_index]}")
    st.info(f"📊 Confidence: {confidence * 100:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)
