import cv2
import torch
from ultralytics import YOLO
import os
from config import (
    YOLO_MODEL_PATH,
    DATA_YAML_PATH,
    VIDEO_PATH,
    OUTPUT_VIDEO_PATH,
    DEVICE
)

def perform_detection(model_path, input_video, output_video, data_yaml):
    model = YOLO(model_path)

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

        results = model(frame, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_tensor = box.cls
                conf_tensor = box.conf

                cls = int(cls_tensor.cpu().numpy()) if torch.is_tensor(cls_tensor) else int(cls_tensor)
                conf = float(conf_tensor.cpu().numpy()) if torch.is_tensor(conf_tensor) else float(conf_tensor)

                label = model.names.get(cls, 'Unknown')

                if isinstance(box.xyxy, torch.Tensor):
                    xyxy = box.xyxy.squeeze(0).tolist() if box.xyxy.dim() == 2 else box.xyxy.tolist()
                else:
                    xyxy = box.xyxy

                if len(xyxy) != 4:
                    print(f"\nWarning: Unexpected number of coordinates {len(xyxy)}.")
                    continue

                x1, y1, x2, y2 = map(int, xyxy)

                color = (0, 255, 0) if label == 'awake' else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                print(f"Frame {current_frame}: {label} with confidence {conf:.2f}")

        out.write(frame)

    print("\nVideo processing complete.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Annotated video saved as {output_video}.")

if __name__ == "__main__":
    perform_detection(
        model_path=YOLO_MODEL_PATH,
        input_video=VIDEO_PATH,
        output_video=OUTPUT_VIDEO_PATH,
        data_yaml=DATA_YAML_PATH
    )