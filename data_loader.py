import os
import numpy as np
from PIL import Image

def load_and_flatten_images(path, label):
    """Load images from directory, convert to grayscale, resize, and flatten"""
    images = []
    labels = []
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        try:
            img = Image.open(img_path)
            if img.mode != 'L':
                img = img.convert('L')  # Convert to grayscale
            img = img.resize((16, 16))  # Resize to 16x16 pixels
            img_array = np.array(img).flatten()  # Flatten to 1D array
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return images, labels
