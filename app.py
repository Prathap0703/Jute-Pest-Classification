import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
from class_labels import class_labels

# Google Drive file ID
FILE_ID = "1XqVPyn3pP4ihDORoW1dKJjIXlDmoTqAC"
MODEL_PATH = "xception_model_jute.keras"

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.resize((299, 299))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🌿 Jute Pest Classification - Multi Image Prediction")

uploaded_files = st.file_uploader(
    "Upload one or more jute pest images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    index = st.slider("Select image index", 0, len(uploaded_files) - 1) if len(uploaded_files) > 1 else 0
    image = Image.open(uploaded_files[index])
    st.image(image, caption=f"Image {index + 1} of {len(uploaded_files)}", use_container_width=True)

    if st.button("Predict this image"):
        with st.spinner("Predicting..."):
            img_array = preprocess_image(image)
            preds = model.predict(img_array)
            probs = tf.nn.softmax(preds).numpy()[0]
            pred_index = np.argmax(probs)
            confidence = probs[pred_index]

        st.success(f"**Predicted Class:** {class_labels[pred_index]}")
        st.info(f"**Confidence:** {confidence:.4f}")

        st.subheader("Prediction Probabilities for all Classes")
        prob_dict = {class_labels[i]: float(probs[i]) for i in range(len(class_labels))}
        st.bar_chart(prob_dict)
else:
    st.info("Please upload at least one image to get started.")

st.markdown("---")
st.markdown("Developed by Prathap V & Team | Jute Pest Classification Project")
