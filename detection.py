# Install necessary libraries
# !pip install streamlit torch torchvision Pillow matplotlib opencv-python-headless

import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Load YOLOv5 model
@st.cache_resource
def load_model():
    # Load a pre-trained YOLOv5 model (weights)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

# Function to predict objects using YOLOv5
def predict_image(image, model):
    # Convert to OpenCV format
    image_cv = np.array(image.convert('RGB'))
    results = model(image_cv)  # YOLOv5 model predictions
    
    # Get predictions data
    predictions = results.pandas().xyxy[0]  # bounding boxes with labels
    return predictions, image_cv

# Function to draw bounding boxes and labels on the image
def draw_boxes(predictions, image_cv):
    for idx, row in predictions.iterrows():
        # Extract coordinates and labels
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        
        # Draw rectangle
        cv2.rectangle(image_cv, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Add label and confidence
        label_text = f'{label} ({confidence:.2f})'
        cv2.putText(image_cv, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    return image_cv

# Streamlit app UI
def main():
    st.title("Object Detection and Classification Tool")
    st.write("Upload an image and the model will detect and classify objects in the image using YOLOv5.")

    # Load model
    model = load_model()

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Button to trigger object detection
        if st.button('Detect Objects'):
            with st.spinner('Detecting objects...'):
                predictions, image_cv = predict_image(image, model)
                result_image = draw_boxes(predictions, image_cv)
                
                # Display result
                st.image(result_image, caption='Object Detection Results', use_column_width=True)

            st.success("Detection complete!")
            st.write(predictions)  # Show the detection details in a table

if __name__ == "__main__":
    main()
