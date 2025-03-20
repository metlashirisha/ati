import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Extract input shape (handle cases where batch size is None)
if isinstance(model.input_shape, list):
    input_shape = model.input_shape[0]  # Some models return a list of shapes
else:
    input_shape = model.input_shape  # Expected shape: (None, height, width, channels)

# Ensure input shape is valid
if len(input_shape) == 4:
    _, img_height, img_width, img_channels = input_shape  # Ignore batch dimension
else:
    st.error("Invalid model input shape. Please check the model architecture.")
    st.stop()

# Define image preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB if needed
    image = image.resize((img_width, img_height))  # Resize to model's expected size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32)

# Streamlit UI
st.title("Autism Facial Recognition Model")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        # Preprocess image
        processed_image = preprocess_image(image)

        # Check the shape before passing to the model
        st.write(f"Processed Image Shape: {processed_image.shape}")

        # Make prediction
        prediction = model.predict(processed_image)

        # Display the result
        st.write(f"Prediction: {prediction}")
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
