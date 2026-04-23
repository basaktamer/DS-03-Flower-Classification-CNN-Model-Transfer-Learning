import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

st.set_page_config(page_title="BloomID Final", page_icon="🌸")
st.title("🌸 BloomID Flower Classifier")

@st.cache_resource
def load_final_model():
    # We use the .keras file you just uploaded
    model_path = "flower_transfer_model.keras"
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        return None
    
    try:
        # safe_mode=False is the key to bypassing the 'dense_2' metadata ghost
        model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        return model
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None

model = load_final_model()

def predict(image, model):
    size = (64, 64)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img = np.asarray(image).astype('float32') / 255.0
    img = img[:, :, ::-1] # BGR Flip
    img = np.expand_dims(img, axis=0)
    
    # Direct functional call
    preds = model(img, training=False)
    
    # If it returns the list of 2 tensors, grab the first one
    if isinstance(preds, list):
        preds = preds[0]
        
    return preds.numpy()

uploaded_file = st.file_uploader("Upload Flower Photo", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    with st.spinner('Identifying...'):
        try:
            results = predict(image, model)
            classes = ['dandelion', 'daisy', 'sunflower', 'tulip', 'rose']
            
            idx = np.argmax(results)
            confidence = np.max(results) * 100
            
            st.success(f"Prediction: **{classes[idx]}**")
            st.info(f"Confidence: {confidence:.2f}%")
        except Exception as e:
            st.error(f"Prediction Error: {e}")