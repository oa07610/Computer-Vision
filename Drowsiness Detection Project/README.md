# Drowsiness Detection Project

## Overview

This project develops a real-time drowsiness detection system optimized for low-resource environments by integrating multiple YOLO (You Only Look Once) models with a Convolutional Neural Network (CNN). Utilizing facial features such as Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), Yaw, Pitch, and Roll, the system employs YOLOv8n, YOLOv8m, YOLOv10m, and YOLOv11n for efficient face detection. The extracted features are classified using a CNN, and an ensemble strategy combines the strengths of both models to achieve high accuracy and minimal latency. This integrated approach ensures reliable drowsiness monitoring suitable for deployment on devices with limited computational capabilities.

## Additional Notes

Configuration: All configuration variables such as paths and model names are centralized in config.py. Modify this file as needed to suit your environment.
Mediapipe Limitations: Ensure that the system has adequate resources for real-time processing, especially if running on CPU.
Sound Alerts: The winsound module is only available on Windows. If you're using a different OS, consider alternative methods for sound alerts or remove this feature.
Model Storage: Trained models are saved in the models/ directory. Ensure sufficient storage space.

## Troubleshooting

CUDA Issues: If you encounter CUDA-related errors, ensure that your system has the appropriate NVIDIA drivers and CUDA toolkit installed. Alternatively, switch to CPU by modifying the DEVICE variable in config.py.
Dataset Issues: Ensure that the dataset paths in config.py are correctly set and that the dataset has been downloaded and processed without errors.
Module Not Found Errors: Ensure that all scripts are run from the project's root directory or adjust the Python path accordingly.

## License

MIT License

## Acknowledgements

Ultralytics YOLO
Mediapipe
Roboflow
