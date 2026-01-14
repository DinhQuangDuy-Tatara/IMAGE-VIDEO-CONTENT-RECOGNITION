import torch
import torch.nn as nn
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_type='resnet18', num_classes=4, weights_path=None):
    """
    Load the specified model.

    Args:
        model_type (str): 'simple' for SimpleCNN, 'resnet18' for ResNet18.
        num_classes (int): Number of output classes.
        weights_path (str): Path to saved weights (optional).

    Returns:
        nn.Module: Loaded model.
    """
    if model_type == 'simple':
        model = SimpleCNN(num_classes=num_classes)
    elif model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
        # Modify the final layer for num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unsupported model type")

    if weights_path:
        model.load_state_dict(torch.load(weights_path))
        model.eval()

    return model

def predict_image(model, image_tensor, device='cpu'):
    """
    Predict the label for an image tensor.

    Args:
        model (nn.Module): The classification model.
        image_tensor (torch.Tensor): Preprocessed image tensor.
        device (str): Device to run on.

    Returns:
        int: Predicted class index.
    """
    model.to(device)
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()