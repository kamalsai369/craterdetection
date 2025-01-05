import streamlit as st
from PIL import Image
import torch
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO(r"C:\Craters Detection\craterdetection-main\finetuned_model.pt")

# Streamlit UI for image upload
st.title("Crater Detection Web App")

uploaded_image = st.file_uploader("Choose an image...", type="jpg")

if uploaded_image:
    # Open the image using PIL
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Perform inference
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img_array)

    # Visualize the detected bounding boxes
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_array)

    for result in results[0].boxes:
        x_min, y_min, x_max, y_max = result.xyxy[0].cpu().numpy()
        conf = result.conf[0].cpu().item()
        rect = plt.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            edgecolor='red', facecolor='none', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(
            x_min, y_min, f'{conf:.2f}',
            bbox=dict(facecolor='yellow', alpha=0.5)
        )

    plt.title("Crater Detection")
    plt.axis('off')
    st.pyplot(fig)
