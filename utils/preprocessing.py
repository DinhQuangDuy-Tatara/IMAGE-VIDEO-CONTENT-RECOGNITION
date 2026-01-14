import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def preprocess_image(image_path, size=(224, 224)):
    """
    Load, resize, normalize, and convert image to tensor.

    Args:
        image_path (str): Path to the image file.
        size (tuple): Target size for resizing (width, height).

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    # Load image using PIL for better compatibility
    image = Image.open(image_path).convert('RGB')

    # Define transforms: resize, to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Apply transforms
    image_tensor = transform(image)

    return image_tensor.unsqueeze(0)  # Add batch dimension