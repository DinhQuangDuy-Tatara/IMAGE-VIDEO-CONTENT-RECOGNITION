# Image & Video Content Recognition

This project implements an image and video content recognition system using CNN-based image classification combined with video frame sampling.

## Project Structure

- `data/`: Contains images, videos, and extracted frames
- `models/`: Model definitions and weights
- `utils/`: Utility functions for preprocessing
- `main.py`: Main script for training and demo
- `evaluate.py`: Evaluation functions
- `video_processing.py`: Video frame extraction and recognition
- `requirements.txt`: Python dependencies

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your images in `data/images/` and videos in `data/videos/`.

## Usage

Run the main script to train the model and see demos:
```bash
python main.py
```

For image classification:
- Use `demo_image_classification()` in main.py

For video recognition:
- Use `demo_video_processing()` in main.py

## Technologies

- Python 3
- PyTorch
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

## Evaluation

The system evaluates accuracy and confusion matrix for images, and accuracy with processing time for videos.

## Note

This is an academic project. For real-world use, train on appropriate datasets and fine-tune models.