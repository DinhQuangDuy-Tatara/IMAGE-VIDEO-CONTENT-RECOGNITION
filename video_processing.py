import cv2
import os
from utils.preprocessing import preprocess_image
from models.model import predict_image
from collections import Counter
import time
from sklearn.metrics import accuracy_score
import numpy as np

def extract_frames(video_path, output_dir, sample_rate=30):
    """
    Extract frames from video using uniform sampling.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory to save frames.
        sample_rate (int): Extract every N frames.

    Returns:
        list: List of frame paths.
    """
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        frame_count += 1

    cap.release()
    return frame_paths

def recognize_video_content(model, frame_paths, device='cpu'):
    """
    Recognize video content by classifying frames and using majority voting.

    Args:
        model (nn.Module): The classification model.
        frame_paths (list): List of frame image paths.
        device (str): Device to run on.

    Returns:
        int: Predicted video class.
    """
    predictions = []
    for frame_path in frame_paths:
        image_tensor = preprocess_image(frame_path)
        pred = predict_image(model, image_tensor, device)
        predictions.append(pred)

    # Majority voting
    if predictions:
        most_common = Counter(predictions).most_common(1)[0][0]
        return most_common
    else:
        return None

def evaluate_video_recognition(model, video_paths, true_labels, sample_rate=30, device='cpu'):
    """
    Evaluate video recognition accuracy and processing time.

    Args:
        model (nn.Module): The classification model.
        video_paths (list): List of video paths.
        true_labels (list): True labels for videos.
        sample_rate (int): Frame sampling rate.
        device (str): Device to run on.

    Returns:
        dict: Accuracy and average processing time.
    """
    predictions = []
    times = []

    for video_path in video_paths:
        start_time = time.time()
        frame_paths = extract_frames(video_path, 'data/frames', sample_rate)
        pred = recognize_video_content(model, frame_paths, device)
        end_time = time.time()
        predictions.append(pred)
        times.append(end_time - start_time)

    accuracy = accuracy_score(true_labels, predictions)
    avg_time = np.mean(times)

    return {'accuracy': accuracy, 'avg_processing_time': avg_time}