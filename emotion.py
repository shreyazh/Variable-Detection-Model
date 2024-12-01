# dependencies
import streamlit as st
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# streamlit page setup
st.title("Facial Emotion Detection")
st.write("Upload an image to detect emotions.")

# image uploading section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# when an image is uploaded my friend
if uploaded_image is not None:
    # converting the uploaded image to OpenCV format
    img = Image.open(uploaded_image)
    img_np = np.array(img)
    

    # creating two columns for displaying the image and results side-by-side
    col1, col2 = st.columns([1, 1])

    # displaying the uploaded image with a reduced width in the first column
    with col1:
        st.image(img, caption="Uploaded Image", width=300)

    # running DeepFace analysis for emotion detection & displaying results in second column
    with col2:
        st.write("Emotion Analysis Result:")
        try:
            result = DeepFace.analyze(img_np, actions=['emotion'])

            # displaying each emotion with its probability
            for emotion, score in result[0]['emotion'].items():
                st.write(f"{emotion}: {score:.2f}%")              
                     
        except Exception as e:
            st.error("Error analyzing image. Please try a different image.")