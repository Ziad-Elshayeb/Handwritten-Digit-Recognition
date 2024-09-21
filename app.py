import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Disable GPU if you want to avoid GPU issues
tf.config.set_visible_devices([], 'GPU')

# Load your trained model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to 28x28 as per the MNIST dataset
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    image = np.array(image)
    
    # Invert colors if necessary (some MNIST models need black background and white digits)
    image = np.invert(image)
    
    # Normalize the image to the range 0-1
    image = image / 255.0
    
    # Reshape for the model
    image = image.reshape(1, 28, 28, 1)
    
    return image

# Streamlit app design
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9) and the model will recognize it.")

# Image upload feature
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Digit', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the digit
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction, axis=1)[0]

    # Display the result
    st.write(f"The model predicts this digit is: **{predicted_digit}**")
    
    # Option to display confidence levels
    if st.checkbox("Show confidence levels", True):
        confidence_scores = prediction[0]
        st.bar_chart(confidence_scores)
