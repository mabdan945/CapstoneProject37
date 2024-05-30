import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background

# Function to reset the app
def reset_app():
    st.session_state['file'] = None

# Initialize session state variables if not already initialized
if 'file' not in st.session_state:
    st.session_state['file'] = None

set_background('./bgrd/bg.jpg')

# Set title
st.title('Casting Quality Control')

# Set header
st.header('Please upload a Casting Product Image')

# Upload file
uploaded_file = st.file_uploader('', type=['jpeg', 'jpg', 'png'], key='file')

# Load classifier
model = load_model('./modelcast.h5')

# Load class names
with open('./model/label.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# Display image and classification results if a file is uploaded
if st.session_state['file'] is not None:
    image = Image.open(st.session_state['file']).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify image
    class_name, conf_score = classify(image, model, class_names)

    # Write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))

# Add a reset button
if st.button('Reset'):
    reset_app()
