import os
from ultralytics import YOLO
from config import (
    YOLO_MODELS,
    DATA_YAML_PATH,
    EPOCHS,
    IMGSZ,
    BATCH_SIZE,
    YOLO_MODELS_DIR
)

def train_yolo_models():
    os.makedirs(YOLO_MODELS_DIR, exist_ok=True)
    for model_name, model_path in YOLO_MODELS.items():
        print(f"Training {model_name} model...")
        model = YOLO(model_path)
        model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            name=f"{model_name}_detection"
        )
    print("YOLO model training completed.")

if __name__ == "__main__":
    train_yolo_models()
