import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from config import (
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    CNN_MODEL_PATH
)
from models.evaluate_cnn import SleepinessDataset, SleepinessNN

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

def train_model(train_csv, val_csv, model_path, num_epochs=50, learning_rate=0.001):
    # Create datasets
    train_dataset = SleepinessDataset(train_csv)
    val_dataset = SleepinessDataset(val_csv)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SleepinessNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {100 * correct / total:.2f}%')

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print('Training complete and model saved.')

if __name__ == "__main__":
    train_model(
        train_csv=TRAIN_CSV_PATH,
        val_csv=VAL_CSV_PATH,
        model_path=CNN_MODEL_PATH,
        num_epochs=50,
        learning_rate=0.001
    )