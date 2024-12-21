import os
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.feature_extraction import extract_features

def get_label(image_path, label_dir):
    base = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_dir, base + '.txt')
    if not os.path.exists(label_path):
        return None
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    class_id = int(lines[0].strip().split()[0])
    return class_id

def process_dataset(image_dir, label_dir, output_csv):
    data = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in tqdm(image_files, desc=f'Processing {image_dir}'):
        img_path = os.path.join(image_dir, img_file)
        label = get_label(img_path, label_dir)
        if label is None:
            print(f"Warning: No label for image {img_path}. Skipping.")
            continue
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue
        avg_EAR, mar, yaw, pitch, roll = extract_features(image)
        if avg_EAR is None:
            print(f"Warning: No face detected in image {img_path}. Assigning default values.")
            avg_EAR, mar, yaw, pitch, roll = np.nan, np.nan, np.nan, np.nan, np.nan
        data.append({
            'Image': img_file,
            'EAR': avg_EAR,
            'MAR': mar,
            'Yaw': yaw,
            'Pitch': pitch,
            'Roll': roll,
            'Label': label
        })
    df = pd.DataFrame(data)
    df = df.dropna()
    class_map = {0: 'awake', 1: 'drowsy'}
    df['Label'] = df['Label'].map(class_map)
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} with {len(df)} entries.")

if __name__ == "__main__":
    from config import (
        TRAIN_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        VAL_IMAGE_DIR,
        VAL_LABEL_DIR,
        TRAIN_CSV_PATH,
        VAL_CSV_PATH
    )
    # Process training dataset
    process_dataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, TRAIN_CSV_PATH)
    
    # Process validation dataset
    process_dataset(VAL_IMAGE_DIR, VAL_LABEL_DIR, VAL_CSV_PATH)