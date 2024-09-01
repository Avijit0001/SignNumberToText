import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import inflect
import pyttsx3
from rembg import remove

# Initialize the TTS engine
engine = pyttsx3.init()

# Initialize the inflect engine for converting numbers to words
p = inflect.engine()

# Load the Keras model
model = keras.models.load_model("sign")  # Adjust the path if necessary

# Streamlit app
st.title('Sign Number Decode')

# Image capture and display
image = st.camera_input("Capture an image")

if image:
    img = Image.open(image)

    # Remove the background
    img_no_bg = remove(img)

    # Convert to grayscale and resize
    img_no_bg = img_no_bg.convert('L')
    img_no_bg = img_no_bg.resize((64, 64))
    
    img_array = np.array(img_no_bg) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display results
    st.image(img_no_bg, caption='Captured Image with Background Removed', use_column_width=True)
    st.write(f'Predicted Class: {predicted_class}')

    # Trigger the text-to-speech on button click
    if st.button('Say Number'):
        words = p.number_to_words(predicted_class)
        engine.say(words)
        engine.runAndWait()
