import time
import torch
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from config import (
    YOLO_MODELS,
    DATA_YAML_PATH,
    VIDEO_PATH,
    YOLO_MODELS_DIR,
    DEVICE
)
from utils.plotting import plot_evaluation_metrics

def measure_inference_speed(model_path, video_path, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    frame_times = []
    model = YOLO(model_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        model.predict(source=frame, device=device, verbose=False)
        frame_times.append((time.time() - start_time) * 1000)
    cap.release()
    return np.mean(frame_times)

def evaluate_model(model_path, data_yaml_path):
    model = YOLO(model_path)
    results = model.val(data=data_yaml_path)
    return {
        'mAP@0.5': results.box.maps[0],
        'mAP@0.5:0.95': results.box.map,
        'Precision': results.box.p,
        'Recall': results.box.r
    }

def main():
    inference_times = {}
    device = DEVICE
    
    # Measure inference speed
    for model_name in YOLO_MODELS.keys():
        print(f"Measuring inference speed for {model_name}...")
        model_detection_path = os.path.join(YOLO_MODELS_DIR, f"{model_name}_detection", "weights", "best.pt")
        avg_time = measure_inference_speed(model_detection_path, VIDEO_PATH, device)
        inference_times[model_name] = avg_time
        print(f"{model_name}: {avg_time:.2f} ms")
    
    # Evaluate models
    accuracy_metrics = {}
    for model_name in YOLO_MODELS.keys():
        print(f"Evaluating {model_name}...")
        model_detection_path = os.path.join(YOLO_MODELS_DIR, f"{model_name}_detection", "weights", "best.pt")
        metrics = evaluate_model(model_detection_path, DATA_YAML_PATH)
        accuracy_metrics[model_name] = metrics
        print(f"{model_name} metrics:", metrics)
    
    # Compile results into DataFrame
    data = []
    for model, metrics in accuracy_metrics.items():
        avg_precision = np.mean(metrics['Precision'])
        avg_recall = np.mean(metrics['Recall'])
        inference_time = inference_times.get(model, np.nan)
        data.append({
            'Model': model,
            'mAP@0.5': metrics['mAP@0.5'],
            'mAP@0.5:0.95': metrics['mAP@0.5:0.95'],
            'Average Precision': avg_precision,
            'Average Recall': avg_recall,
            'Inference Time (ms)': inference_time
        })
    df = pd.DataFrame(data)
    print("Evaluation Metrics DataFrame:")
    print(df)
    
    # Plot Evaluation Metrics
    plot_evaluation_metrics(df)

if __name__ == "__main__":
    main()