import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from config import (
    VAL_CSV_PATH,
    CNN_MODEL_PATH
)

class SleepinessDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx, 1:6].values.astype(np.float32)  # EAR, MAR, Yaw, Pitch, Roll
        label = self.data.iloc[idx, 6]
        label = 1 if label == 'drowsy' else 0
        return features, label

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

def evaluate_model(val_csv, model_path):
    # Load the validation dataset
    val_dataset = SleepinessDataset(val_csv)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model and load trained weights
    model = SleepinessNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability for 'drowsy'
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probabilities.tolist())

    # Compute Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, target_names=['Awake', 'Drowsy'])

    # Display Metrics
    print("=== Evaluation Metrics ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}\n")

    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Optionally, save metrics to a file or return them
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc,
        'Confusion Matrix': conf_matrix,
        'Classification Report': class_report
    }

    return metrics

if __name__ == "__main__":
    metrics = evaluate_model(VAL_CSV_PATH, CNN_MODEL_PATH)