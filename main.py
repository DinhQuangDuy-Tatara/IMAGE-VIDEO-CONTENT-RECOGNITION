import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model import load_model
from evaluate import evaluate_model, plot_confusion_matrix
from video_processing import extract_frames, recognize_video_content, evaluate_video_recognition
import os

# Class names for the project
CLASS_NAMES = ['humans', 'vehicles', 'nature', 'indoor']

def train_model(model, train_loader, val_loader, epochs=10, device='cpu'):
    """
    Train the model.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        epochs (int): Number of epochs.
        device (str): Device.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Evaluate on validation
        metrics = evaluate_model(model, val_loader, device)
        print(f"Val Accuracy: {metrics['accuracy']}")

    # Save model
    torch.save(model.state_dict(), 'models/model_weights.pth')

def demo_image_classification():
    # Load model
    model = load_model('simple', num_classes=4)
    # Assume trained, load weights
    if os.path.exists('models/model_weights.pth'):
        model.load_state_dict(torch.load('models/model_weights.pth'))

    # Example image path (assume exists)
    image_path = 'data/images/example.jpg'  # User should provide
    if os.path.exists(image_path):
        from utils.preprocessing import preprocess_image
        from models.model import predict_image
        image_tensor = preprocess_image(image_path)
        pred = predict_image(model, image_tensor)
        print(f"Predicted class: {CLASS_NAMES[pred]}")
    else:
        print("Example image not found. Please add an image to data/images/")

def demo_video_processing():
    # Load model
    model = load_model('simple', num_classes=4)
    if os.path.exists('models/model_weights.pth'):
        model.load_state_dict(torch.load('models/model_weights.pth'))

    # Example video path
    video_path = 'data/videos/example.mp4'  # User should provide
    if os.path.exists(video_path):
        frame_paths = extract_frames(video_path, 'data/frames', sample_rate=30)
        pred = recognize_video_content(model, frame_paths)
        print(f"Predicted video class: {CLASS_NAMES[pred]}")
    else:
        print("Example video not found. Please add a video to data/videos/")

if __name__ == "__main__":
    # For demo, assume CIFAR10 as proxy dataset (not ideal, but for runnable code)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Map CIFAR classes to our classes (rough approximation)
    # 0: airplane -> vehicles, 1: automobile -> vehicles, 2: bird -> nature, etc.
    class_mapping = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2}  # Simplified

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Remap labels
    train_dataset.targets = [class_mapping[t] for t in train_dataset.targets]
    val_dataset.targets = [class_mapping[t] for t in val_dataset.targets]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = load_model('simple', num_classes=4)
    train_model(model, train_loader, val_loader, epochs=5)

    # Evaluate
    metrics = evaluate_model(model, val_loader)
    print(f"Final Accuracy: {metrics['accuracy']}")
    plot_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES)

    # Demo
    demo_image_classification()
    demo_video_processing()