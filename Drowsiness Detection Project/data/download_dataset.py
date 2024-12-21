from roboflow import Roboflow
from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    DATASET_FORMAT_YOLOV8,
    DATASET_FORMAT_YOLOV11
)

def download_dataset():
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    # Download in YOLOv8 format
    dataset_yolov8 = version.download(DATASET_FORMAT_YOLOV8)
    print(f"YOLOv8 dataset downloaded to {dataset_yolov8}")

    # Download in YOLOv11 format
    dataset_yolov11 = version.download(DATASET_FORMAT_YOLOV11)
    print(f"YOLOv11 dataset downloaded to {dataset_yolov11}")

if __name__ == "__main__":
    download_dataset()