import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define landmark indices for eyes and mouth based on Mediapipe
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]

def calculate_EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_MAR(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (2.0 * D)
    return mar

def get_head_pose(image, landmarks):
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points.
    image_points = np.array([
        landmarks[30],     # Nose tip
        landmarks[8],      # Chin
        landmarks[36],     # Left eye left corner
        landmarks[45],     # Right eye right corner
        landmarks[48],     # Left Mouth corner
        landmarks[54]      # Right mouth corner
    ], dtype="double")

    # Camera internals
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

    # Solve PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return None, None, None

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Combine rotation matrix and translation vector
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))

    # Decompose projection matrix to get Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

    yaw = euler_angles[1][0]
    pitch = euler_angles[0][0]
    roll = euler_angles[2][0]

    return yaw, pitch, roll

def extract_features(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None, None, None, None

    face_landmarks = results.multi_face_landmarks[0]
    ih, iw, _ = image.shape

    landmarks = []
    for lm in face_landmarks.landmark:
        x, y = int(lm.x * iw), int(lm.y * ih)
        landmarks.append((x, y))

    # Extract eyes and mouth
    left_eye = np.array([landmarks[i] for i in LEFT_EYE_IDX])
    right_eye = np.array([landmarks[i] for i in RIGHT_EYE_IDX])
    mouth = np.array([landmarks[i] for i in MOUTH_IDX])

    # Calculate EAR and MAR
    left_EAR = calculate_EAR(left_eye)
    right_EAR = calculate_EAR(right_eye)
    avg_EAR = (left_EAR + right_EAR) / 2.0
    mar = calculate_MAR(mouth)

    # Calculate head pose
    yaw, pitch, roll = get_head_pose(image, landmarks)

    return avg_EAR, mar, yaw, pitch, roll