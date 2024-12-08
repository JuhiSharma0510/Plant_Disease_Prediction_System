import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown

# Google Drive file ID of the model
GDRIVE_FILE_ID = "1UX81o6Tc_JKjpzj8IClIkAmCtLa_CgJP"

# Paths for model and class indices
model_file = "plant_disease_prediction_model.h5"
class_indices_file = "class_indices.json"

# Download model and class indices from Google Drive
def download_model():
    if not os.path.exists(model_file):
        with st.spinner("Downloading model... This may take a while."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", model_file, quiet=False)
            st.success("Model downloaded successfully!")

def load_class_indices():
    if os.path.exists(class_indices_file):
        return json.load(open(class_indices_file))
    else:
        st.error("Class indices file not found!")
        return None

# Load the pre-trained model
download_model()
model = tf.keras.models.load_model(model_file)

# Load class indices
class_indices = load_class_indices()

# Function to load and preprocess the image
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(model, uploaded_image, class_indices):
    img = Image.open(uploaded_image)
    preprocessed_img = load_and_preprocess_image(img)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title("🌱 Plant Disease Classifier")
st.markdown("This application identifies diseases in plants based on uploaded images. Powered by Deep Learning.")

# Sidebar with plant information and emojis 🌱
st.sidebar.title("🌿 Plants in This Project")
st.sidebar.markdown("Explore the plants and their diseases included in this classifier:")

plants = {
    "Apple 🍎": "Common diseases include Apple Scab, Black Rot, and Cedar Apple Rust.",
    "Blueberry 🫐": "Generally healthy with occasional fungal diseases.",
    "Cherry 🍒": "Cherry trees are prone to Powdery Mildew and Leaf Spot diseases.",
    "Corn 🌽": "Susceptible to Gray Leaf Spot, Common Rust, and Northern Leaf Blight.",
    "Tomato 🍅": "Affected by diseases like Bacterial Spot, Early Blight, and Leaf Mold.",
}

for plant, description in plants.items():
    st.sidebar.subheader(plant)
    st.sidebar.write(description)
    st.sidebar.markdown("---")

# File uploader
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    
    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption="Uploaded Image")
        
    with col2:
        if st.button('Classify'):
            try:
                # Preprocess the uploaded image and predict the class
                prediction = predict_image_class(model, uploaded_image, class_indices)
                st.success(f'Prediction: {str(prediction)}')
            except Exception as e:
                st.error(f"An error occurred: {e}")
