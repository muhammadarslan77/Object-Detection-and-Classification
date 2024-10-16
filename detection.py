import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Define COCO instance category names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load a pre-trained model for object detection
@st.cache_resource
def load_model():
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to get predictions from the model
def get_predictions(image, model, threshold=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard ImageNet normalization
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract the boxes and labels with scores above the threshold
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(predictions[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(predictions[0]['boxes'].detach().numpy())]
    pred_scores = list(predictions[0]['scores'].detach().numpy())

    filtered_boxes = []
    filtered_classes = []

    for i, score in enumerate(pred_scores):
        if score > threshold:
            filtered_boxes.append(pred_boxes[i])
            filtered_classes.append(pred_classes[i])

    return filtered_boxes, filtered_classes

# Draw boxes and labels on the image
def draw_boxes(boxes, classes, image):
    img = np.array(image)
    for box, label in zip(boxes, classes):
        cv2.rectangle(img, box[0], box[1], color=(255, 0, 0), thickness=2)
        cv2.putText(img, label, (int(box[0][0]), int(box[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return img

# Streamlit App Layout
st.title("Object Detection and Classification Tool")
st.write("Upload an image to perform object detection and classification.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure image is in RGB mode
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting objects...")
    model = load_model()  # Load the model

    # Perform object detection
    boxes, classes = get_predictions(image, model, threshold=0.5)

    # Draw boxes on the image
    result_img = draw_boxes(boxes, classes, image.copy())

    # Display the image with detected boxes
    st.image(result_img, caption="Detected Objects", use_column_width=True)

    # Display the detected classes
    st.write("Detected Classes:", classes)
