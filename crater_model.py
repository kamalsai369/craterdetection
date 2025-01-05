import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')


# Import wandb and log in
import wandb

# Log in to wandb with API key
wandb.login(key='fcc68e0c164b296e8d7aea49b4b03b796359a018')

train_img_path = "/content/drive/MyDrive/Crater Detection/craters/train/images"
train_lbl_path = "/content/drive/MyDrive/Crater Detection/craters/train/labels"
valid_img_path = "/content/drive/MyDrive/Crater Detection/craters/valid/images"
valid_lbl_path = "/content/drive/MyDrive/Crater Detection/craters/train/labels"
test_img_path = "/content/drive/MyDrive/Crater Detection/craters/test/images"
test_lbl_path = "/content/drive/MyDrive/Crater Detection/craters/train/labels"
model_path = "/content/drive/MyDrive/Crater Detection/best.pt"
data_yaml_path = "/content/drive/MyDrive/Crater Detection/data.yaml"

def load_labels(label_path):
    label_files = os.listdir(label_path)
    data = []
    classes = set()
    for file in label_files:
        with open(os.path.join(label_path, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                data.append([file, *parts])
                classes.add(int(parts[0]))
    df = pd.DataFrame(data, columns=['file', 'class', 'x_center', 'y_center', 'width', 'height'])
    return df, sorted(classes)

train_labels, train_classes = load_labels(train_lbl_path)
valid_labels, valid_classes = load_labels(valid_lbl_path)
test_labels, test_classes = load_labels(test_lbl_path)

all_classes = sorted(set(train_classes + valid_classes + test_classes))
class_names = [f'class_{i}' for i in all_classes]

print("Train Labels")
print(train_labels.head())
print("\nValidation Labels")
print(valid_labels.head())
print("\nTest Labels")
print(test_labels.head())

# Create data.yaml
data_yaml_content = f"""
train: {train_img_path}
val: {valid_img_path}
test: {test_img_path}

nc: {len(all_classes)}  # number of classes
names: {class_names}  # class names
"""

with open(data_yaml_path, 'w') as f:
    f.write(data_yaml_content)

# Plotting distribution of bounding box sizes
def plot_bounding_box_distribution(labels, title):
    labels['area'] = labels['width'] * labels['height']
    fig = px.histogram(labels, x='area', nbins=50, title=title)
    fig.show()

plot_bounding_box_distribution(train_labels, 'Train Bounding Box Area Distribution')
plot_bounding_box_distribution(valid_labels, 'Validation Bounding Box Area Distribution')
plot_bounding_box_distribution(test_labels, 'Test Bounding Box Area Distribution')

def visualize_sample_images(image_path, label_df, n_samples=5):
    image_files = os.listdir(image_path)[:n_samples]
    for img_file in image_files:
        img_path = os.path.join(image_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)

        labels = label_df[label_df['file'] == img_file]
        for _, label in labels.iterrows():
            x_center = label['x_center'] * img.shape[1]
            y_center = label['y_center'] * img.shape[0]
            width = label['width'] * img.shape[1]
            height = label['height'] * img.shape[0]
            x_min = x_center - width / 2
            y_min = y_center - height / 2

            rect = plt.Rectangle(
                (x_min, y_min), width, height,
                edgecolor='red', facecolor='none', linewidth=2
            )
            ax.add_patch(rect)

        plt.title(f'Sample Image: {img_file}')
        plt.axis('off')
        plt.show()

# Call the visualization function for train, validation, and test sets
visualize_sample_images(train_img_path, train_labels)
visualize_sample_images(valid_img_path, valid_labels)
visualize_sample_images(test_img_path, test_labels)

# YOLOv8 Model Training and Evaluation
!pip install -q ultralytics
from ultralytics import YOLO

import os
import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model and move it to the device
model = YOLO('yolov8n.pt').to(device)

# Train the model on the device
model.train(data=data_yaml_path, epochs=50, device=device)

# Evaluate the model
results = model.val()

# Save the trained model
model.save('/content/drive/MyDrive/Crater Detection/craters/best_model.pt')
def visualize_detections(model, image_path, n_samples=10):
    image_files = os.listdir(image_path)[:n_samples]
    for img_file in image_files:
        img_path = os.path.join(image_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform inference on the device
        results = model(img_path, device=device)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img)

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

        plt.title(f'Detection in: {img_file}')
        plt.axis('off')
        plt.show()

# Call the visualization function
visualize_detections(model, test_img_path)

print("Model training, evaluation, and sample visualization completed. The trained model is saved at '/content/drive/MyDrive/Crater Detection/craters/best_model.pt'")

# Load the trained YOLO model and fine-tune it
model = YOLO('/content/drive/MyDrive/Crater Detection/craters/best_model.pt')

# Fine-tune with more epochs and possibly a lower learning rate
model.train(data=data_yaml_path, epochs=20, lr0=0.0001, device=device)

# Evaluate the fine-tuned model
results = model.val()

# Save the fine-tuned model
model.save('/content/drive/MyDrive/Crater Detection/craters/finetuned_model.pt')


