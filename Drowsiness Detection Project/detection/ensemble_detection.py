import cv2
import torch
from ultralytics import YOLO
import numpy as np
from config import (
    YOLO_MODEL_PATH,
    CNN_MODEL_PATH,
    INPUT_VIDEO_PATH,
    ENSEMBLE_OUTPUT_VIDEO_PATH,
    DEVICE
)
from utils.feature_extraction import extract_features

def evaluate_yolo(model, data_yaml_path):
    results = model.val(data=data_yaml_path, imgsz=640, conf=0.001, iou=0.65, task='detect')
    yolov8_metrics = {
        'mAP@0.5': results.box.maps[0],
        'mAP@0.5:0.95': results.box.map,
        'Precision': results.box.p,
        'Recall': results.box.r
    }
    return yolov8_metrics

class SleepinessNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, num_classes=2):
        super(SleepinessNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def perform_ensemble_detection(yolo_model_path, cnn_model_path, input_video, output_video):
    # Load YOLOv8 Model
    yolo_model = YOLO(yolo_model_path)

    # Load CNN Model
    cnn_model = SleepinessNN()
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device('cpu')))
    cnn_model.eval()

    # Define Ensembling Weights
    YOLON_WEIGHT = 0.3
    CNN_WEIGHT = 0.7

    # Define Colors for Annotations
    COLORS = {
        'awake': (0, 255, 0),    # Green
        'drowsy': (0, 0, 255),   # Red
    }

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        print(f"Processing frame {current_frame}/{frame_count}", end='\r')

        # YOLOv8 Inference
        results_yolo = yolo_model(frame, verbose=False)
        yolo_preds = []
        yolo_confs = []

        for result in results_yolo:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls.cpu().numpy())
                conf = float(box.conf.cpu().numpy())
                label = yolo_model.names.get(cls, 'Unknown')
                yolo_preds.append(label)
                yolo_confs.append(conf)

        if yolo_preds:
            yolo_prediction = max(set(yolo_preds), key=yolo_preds.count)
            yolo_confidence = np.mean([yolo_confs[i] for i, lbl in enumerate(yolo_preds) if lbl == yolo_prediction])
        else:
            yolo_prediction = 'No Detection'
            yolo_confidence = 0.0

        # CNN Inference
        features = extract_features(frame)
        if features == (None, None, None, None, None):
            cnn_prediction = 'No Face Detected'
            cnn_confidence = 0.0
        else:
            avg_EAR, mar, yaw, pitch, roll = features
            input_features = torch.tensor([avg_EAR, mar, yaw, pitch, roll], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = cnn_model(input_features)
                probabilities = torch.softmax(output, dim=1)
                prob_drowsy = probabilities[0][1].item()
                prob_awake = probabilities[0][0].item()
                _, predicted = torch.max(output, 1)
                cnn_prediction = 'Drowsy' if predicted.item() == 1 else 'Awake'
                cnn_confidence = prob_drowsy if cnn_prediction == 'Drowsy' else prob_awake

        # Ensemble Prediction
        total_weight = YOLON_WEIGHT + CNN_WEIGHT
        weight_yolo = YOLON_WEIGHT / total_weight
        weight_cnn = CNN_WEIGHT / total_weight

        score_awake = 0.0
        score_drowsy = 0.0

        if yolo_prediction == 'awake':
            score_awake += yolo_confidence * weight_yolo
        elif yolo_prediction == 'drowsy':
            score_drowsy += yolo_confidence * weight_yolo

        score_awake += (1 - cnn_confidence) * weight_cnn
        score_drowsy += cnn_confidence * weight_cnn

        ensemble_prediction = 'Drowsy' if score_drowsy > score_awake else 'Awake'
        ensemble_confidence = score_drowsy if ensemble_prediction == 'Drowsy' else score_awake

        # Annotate Ensemble Prediction
        cv2.putText(frame, f"Ensemble: {ensemble_prediction} ({ensemble_confidence:.2f})",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.get(ensemble_prediction, (255, 255, 255)), 2)

        # Visual Alert for Drowsy Prediction
        if ensemble_prediction == 'Drowsy':
            cv2.putText(frame, 'ALERT: Drowsy Detected', (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)

    print("\nEnsemble video processing complete.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Annotated ensemble video saved as {output_video}.")

if __name__ == "__main__":
    perform_ensemble_detection(
        yolo_model_path=YOLO_MODEL_PATH,
        cnn_model_path=CNN_MODEL_PATH,
        input_video=INPUT_VIDEO_PATH,
        output_video=ENSEMBLE_OUTPUT_VIDEO_PATH
    )