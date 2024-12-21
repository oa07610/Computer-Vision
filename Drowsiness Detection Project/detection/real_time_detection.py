import cv2
import torch
from ultralytics import YOLO
import numpy as np
import winsound
from config import (
    YOLO_MODEL_PATH,
    CNN_MODEL_PATH,
    DEVICE
)
from utils.feature_extraction import extract_features

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

def real_time_detection(yolo_model_path, cnn_model_path):
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

    # Initialize Video Capture for Real-Time (webcam)
    cap = cv2.VideoCapture(0)  # Use webcam (0 for default)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Real-time video processing with Ensemble Model... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        # Trigger visual alert for Drowsy prediction
        if ensemble_prediction == 'Drowsy':
            cv2.putText(frame, 'ALERT: Drowsy Detected', (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Optional: Trigger a sound alert
            try:
                winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms
            except:
                pass  # Handle cases where winsound is not available

        # Display the frame in a window
        cv2.imshow("Real-time Drowsiness Detection", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    print("Real-time video processing finished.")

if __name__ == "__main__":
    real_time_detection(
        yolo_model_path=YOLO_MODEL_PATH,
        cnn_model_path=CNN_MODEL_PATH
    )