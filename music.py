import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
from streamlit_webrtc import webrtc_streamer
import webbrowser
#####################################
try:
    gender = np.load("gender.npy")
except:
    gender = ""

#####################################
# Predicting the gender
model = tf.keras.models.load_model("mdl_wt.hdf5")
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'male',
            1: 'female'}
pred = ""
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        pred = map_dict[prediction]
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))
        np.save("gender.npy", np.array([pred]))
  
        