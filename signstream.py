import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import inflect
import pyttsx3

engine = pyttsx3.init()
p = inflect.engine()


# Load the Keras model from the .pkl file
model = keras.models.load_model("sign")

# Streamlit app
st.title('Sign Number Decode')

# Image capture and display
image = st.camera_input("Capture an image")

if image:
    img = Image.open(image)
    img = img.convert('L') 
    img = img.resize((64, 64))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = np.expand_dims(img_array, axis=-1)  

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    st.image(img, caption='Captured Image', use_column_width=True)
    st.write(f'Predicted Class: {predicted_class[0]}')
    x = st.button("Say the Number") 
    if(x):
        number = predicted_class[0]
        words = p.number_to_words(number)
        engine.say(words)
        engine.runAndWait()
