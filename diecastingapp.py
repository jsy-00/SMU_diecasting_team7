# import streamlit as st
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# # Define available models
# models = {
#     "baseline": './model/baseline.keras',
#     "baseline+adam": './model/baseline+adam.keras',
#     "baseline+rmsprop": './model/baseline+rmsprop.keras',
# }

# # Define class labels
# class_labels = ['NG', 'OK']

# def load_model(model_name):
#     return models[model_name](weights='imagenet')

# def preprocess_image(image):
#     image = image.convert('RGB')  # Convert RGBA to RGB
#     image = image.resize((224, 224))
#     image = np.array(image)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# def predict(image, model):
#     preprocessed_image = preprocess_image(image)
#     predictions = model.predict(preprocessed_image)
#     class_index = np.argmax(predictions)
#     class_name = class_labels[class_index]
#     confidence = predictions[0][class_index]
#     return class_name, confidence

# st.title('DieCasting Classifier')

# # Model selection
# model_name = st.selectbox("Choose a model", list(models.keys()))


# # Load the selected model
# model = tf.keras.models.load_model(models[model_name], compile=False, safe_mode=True)

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
#     st.write("")
#     st.write(f"Classifying...")
#     class_name, confidence = predict(image, model)
#     st.write(f"Prediction: {class_name}")
#     st.write(f"Confidence: {confidence:.2f}")

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import io

# Define the model
model_path = './model/baseline.keras'

# Define class labels
class_labels = ['NG', 'OK']

# Load the model
model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)

def preprocess_image(image):
    image = image.convert('RGB')  # Convert RGBA to RGB
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image, model):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    class_index = np.argmax(predictions)
    class_name = class_labels[class_index]
    confidence = predictions[0][class_index]
    return class_name, confidence

# Streamlit app title
st.title('DieCasting Classifier')

# Image or video upload
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # If it's an image, process it normally
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")
        class_name, confidence = predict(image, model)
        st.write(f"Prediction: {class_name}")
        st.write(f"Confidence: {confidence:.2f}")

    elif uploaded_file.type.startswith('video'):
        # If it's a video, process the first frame to predict
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

        # Convert the video bytes to an OpenCV-readable format
        nparr = np.frombuffer(video_bytes, np.uint8)
        cap = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if cap is not None:
            # Convert the frame to PIL Image and process
            pil_image = Image.fromarray(cv2.cvtColor(cap, cv2.COLOR_BGR2RGB))
            st.image(pil_image, caption='First Frame of Video', use_column_width=True)

            st.write("Classifying first frame...")
            class_name, confidence = predict(pil_image, model)
            st.write(f"Prediction: {class_name}")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.error("Error: Could not decode video frame")
