# server/predictor.py

import os
import json
import requests
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d

# =========================
# CONFIG & GLOBAL VARIABLES
# =========================
mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS * 2

MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "Models/checkpoints/final_model.h5")
# Set your direct-download link here by ENV var or fallback
MODEL_URL = os.environ.get("MODEL_URL",
    "https://github.com/ChuongNguyen23/vsl-fastapi/releases/tag/final_model.h5"
)
LABEL_MAP_PATH = os.environ.get("LABEL_MAP_PATH", "Logs/label_map.json")

_model = None
_label_map = None
_inv_label_map = None

# =========================
# UTILS
# =========================
def ensure_direct_download_link(url: str) -> str:
    if not url:
        return url
    if "drive.google.com" in url and "uc?export=download" not in url:
        import re
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
        if match:
            file_id = match.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

def download_model_if_needed():
    if os.path.exists(MODEL_LOCAL_PATH):
        print(f"✅ Model found locally at {MODEL_LOCAL_PATH}")
        return MODEL_LOCAL_PATH

    url = ensure_direct_download_link(MODEL_URL)
    if not url:
        raise FileNotFoundError(f"Model not found at {MODEL_LOCAL_PATH} and MODEL_URL not set")

    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    print(f"⬇️ Downloading model from: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_LOCAL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"✅ Model downloaded to {MODEL_LOCAL_PATH}")
    return MODEL_LOCAL_PATH

# =========================
# MODEL LOAD & LABEL MAP
# =========================
def load_model_and_labels():
    global _model, _label_map, _inv_label_map
    if _model is not None:
        return

    model_path = download_model_if_needed()
    _model = tf.keras.models.load_model(model_path)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        _label_map = json.load(f)
    _inv_label_map = {v: k for k, v in _label_map.items()}
    print("✅ Model and label map loaded successfully.")

# =========================
# MEDIAPIPE + PREPROCESSING
# =========================
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

    keypoints = np.concatenate([pose_kps, left_hand_kps, right_hand_kps])
    return keypoints.flatten()

def interpolate_keypoints(keypoints_sequence, target_len=60):
    if keypoints_sequence is None or len(keypoints_sequence) == 0:
        return None
    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)
    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))
    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]
        interpolator = interp1d(original_times, feature_values,
                                kind="cubic", bounds_error=False, fill_value="extrapolate")
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

# =========================
# PREDICTION
# =========================
def predict_from_video(video_path):
    """
    Input: video_path (local path)
    Output: dict e.g. {"label":"Xin chào", "confidence":0.95}
    """
    global _model, _inv_label_map
    if _model is None:
        load_model_and_labels()

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)
    seq = sequence_frames(video_path, holistic)
    holistic.close()

    if not seq:
        return {"error": "No keypoints extracted from video"}

    kp = interpolate_keypoints(seq)
    if kp is None:
        return {"error": "Interpolation failed"}

    pred = _model.predict(np.expand_dims(kp, axis=0))
    pred_idx = int(np.argmax(pred, axis=1)[0])
    confidence = float(np.max(pred))
    label = _inv_label_map.get(pred_idx, str(pred_idx))

    return {"label": label, "confidence": confidence}
