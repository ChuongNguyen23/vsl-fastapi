# server/predictor.py
import os
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "Models/checkpoints/final_model.keras")
MODEL_URL = os.environ.get("MODEL_URL")  # optional: public URL to download model if not in repo

def download_model_if_needed():
    """
    Nếu model không có trong repo, bạn có thể đặt MODEL_URL env var (ví dụ link Google Drive hoặc S3).
    Hàm này sẽ tải model về MODEL_LOCAL_PATH khi cần.
    """
    if os.path.exists(MODEL_LOCAL_PATH):
        return MODEL_LOCAL_PATH

    url = MODEL_URL
    if not url:
        raise FileNotFoundError(f"Model not found at {MODEL_LOCAL_PATH} and MODEL_URL not set")

    # Tải file (giản lược). Bạn có thể dùng requests or gdown for GoogleDrive.
    import requests
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    print(f"Downloading model from {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(MODEL_LOCAL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded.")
    return MODEL_LOCAL_PATH

# Load model once
_model = None
_label_map = None
_inv_label_map = None

def load_model_and_labels(label_map_path="Logs/label_map.json"):
    global _model, _label_map, _inv_label_map
    if _model is not None:
        return
    # ensure model file exists (or download)
    if not os.path.exists(MODEL_LOCAL_PATH):
        download_model_if_needed()
    # Load TF model
    _model = tf.keras.models.load_model(MODEL_LOCAL_PATH)
    # load label map
    import json
    with open(label_map_path, 'r', encoding='utf-8') as f:
        _label_map = json.load(f)
    _inv_label_map = {v: k for k, v in _label_map.items()}
    print("Model and label map loaded.")

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    if results and results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]
    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    keypoints = np.concatenate([pose_kps,left_hand_kps, right_hand_kps])
    return keypoints.flatten()

def interpolate_keypoints(keypoints_sequence, target_len = 60):
    if keypoints_sequence is None or len(keypoints_sequence) == 0:
        return None
    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)
    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))
    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]
        interpolator = interp1d(original_times, feature_values, kind='cubic', bounds_error=False, fill_value="extrapolate")
        interpolated_sequence[:, feature_idx] = interpolator(target_times)
    return interpolated_sequence

def sequence_frames(video_path, holistic):
    sequence_frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 100)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue
        try:
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            if keypoints is not None:
                sequence_frames.append(keypoints)
        except Exception:
            continue
    cap.release()
    return sequence_frames

def predict_from_video(video_path):
    """
    Input: video_path (local path)
    Output: dict e.g. {"label":"...", "confidence":0.95}
    """
    global _model, _inv_label_map
    if _model is None:
        load_model_and_labels()

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    seq = sequence_frames(video_path, holistic)
    holistic.close()
    if not seq:
        return {"error": "No keypoints extracted from video"}

    kp = interpolate_keypoints(seq)
    if kp is None:
        return {"error": "Interpolation failed"}
    # model expects shape (1, seq_len, features)
    pred = _model.predict(np.expand_dims(kp, axis=0))
    pred_idx = int(np.argmax(pred, axis=1)[0])
    confidence = float(np.max(pred))
    label = _inv_label_map.get(pred_idx, str(pred_idx))
    return {"label": label, "confidence": confidence}
