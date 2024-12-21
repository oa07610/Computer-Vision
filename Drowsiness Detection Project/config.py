import os

# Roboflow API Key
ROBOFLOW_API_KEY = "dLeQDsrjKHZbEPzwDBBh"

# Workspace and Project Details
ROBOFLOW_WORKSPACE = "augmented-startups"
ROBOFLOW_PROJECT = "drowsiness-detection-cntmz"
ROBOFLOW_VERSION = 2

# Dataset Download Settings
DATASET_FORMAT_YOLOV8 = "yolov8"
DATASET_FORMAT_YOLOV11 = "yolov11"

# Base Directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_YAML_PATH = os.path.join(BASE_DIR, 'data.yaml')
VIDEO_PATH = os.path.join(BASE_DIR, 'sample_video.mp4')

# YOLO Model Configurations
YOLO_MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLOv10m": "yolov10m.pt",
    "YOLOv11n": "yolo11n.pt"
}

EPOCHS = 50
IMGSZ = 640
BATCH_SIZE = 16

# CNN Model Configuration
CNN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sleepiness_nn.pth')

# Paths for Data Processing
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, 'train', 'images')
TRAIN_LABEL_DIR = os.path.join(BASE_DIR, 'train', 'labels')
VAL_IMAGE_DIR = os.path.join(BASE_DIR, 'valid', 'images')
VAL_LABEL_DIR = os.path.join(BASE_DIR, 'valid', 'labels')

TRAIN_CSV_PATH = os.path.join(BASE_DIR, 'sleepiness_train.csv')
VAL_CSV_PATH = os.path.join(BASE_DIR, 'sleepiness_val.csv')

# Paths for Trained Models
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8_sleep_detection.pt')
YOLO_MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Detection Video Paths
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, 'WIN_20241215_20_50_47_Pro.mp4')
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, 'output_sleep_detection.mp4')

# Ensemble Detection Paths
ENSEMBLE_OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, 'output_sleep_detection_ensemble.mp4')

# Real-Time Detection Settings
REAL_TIME_OUTPUT = "Real-time Drowsiness Detection"

# Additional Configurations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'