import os
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
import json
from scipy.interpolate import interp1d

# ✅ Hằng số và cấu hình
mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS * 2

MODEL_LOCAL_PATH = os.environ.get("MODEL_LOCAL_PATH", "Models/checkpoints/final_model.h5")
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://drive.google.com/file/d/1jIXbNFG4nl401WcNhv-FNMwNc3CvE4IR/view?usp=sharing"
)

def download_model_if_needed():
    """Tải model nếu chưa có (hỗ trợ link Google Drive)."""
    if os.path.exists(MODEL_LOCAL_PATH):
        print(f"✅ Model found locally at {MODEL_LOCAL_PATH}")
        return MODEL_LOCAL_PATH

    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    print(f"⬇️ Downloading model from: {MODEL_URL}")

    if "drive.google.com" in MODEL_URL:
        import gdown
        gdown.download(MODEL_URL, MODEL_LOCAL_PATH, fuzzy=True)
    else:
        import requests
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_LOCAL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    print(f"✅ Model downloaded to {MODEL_LOCAL_PATH}")
    return MODEL_LOCAL_PATH


# ✅ Load model và label map
_model = None
_label_map = None
_inv_label_map = None

def load_model_and_labels(label_map_path="Logs/label_map.json"):
    global _model, _label_map, _inv_label_map
    if _model is not None:
        return

    model_path = download_model_if_needed()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model still not found at {model_path}")

    print(f"📦 Loading model from {model_path} ...")
    _model = tf.keras.models.load_model(model_path)

    with open(label_map_path, "r", encoding="utf-8") as f:
        _label_map = json.load(f)
    _inv_label_map = {v: k for k, v in _label_map.items()}
    print("✅ Model and labels loaded successfully.")


# ✅ Các hàm hỗ trợ nhận diện
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results


def extract_keypoints(results):
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark[:N_UPPER_BODY_POSE_LANDMARKS]):
            pose_kps[i] = [lm.x, lm.y, lm.z]
    if results.left_hand_landmarks:
        left_hand_kps = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        right_hand_kps = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])

    return np.concatenate([pose_kps, left_hand_kps, right_hand_kps]).flatten()


def interpolate_keypoints(sequence, target_len=60):
    if not sequence:
        return None
    original_times = np.linspace(0, 1, len(sequence))
    target_times = np.linspace(0, 1, target_len)
    num_features = sequence[0].shape[0]
    interpolated = np.zeros((target_len, num_features))
    for i in range(num_features):
        vals = [f[i] for f in sequence]
        fn = interp1d(original_times, vals, kind="linear", fill_value="extrapolate")
        interpolated[:, i] = fn(target_times)
    return interpolated


def predict_from_video(video_path):
    if _model is None:
        load_model_and_labels()

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    seq, cap = [], cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // 100)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue
        _, results = mediapipe_detection(frame, holistic)
        seq.append(extract_keypoints(results))
    cap.release()
    holistic.close()

    if not seq:
        return {"error": "No keypoints extracted"}

    kp = interpolate_keypoints(seq)
    pred = _model.predict(np.expand_dims(kp, axis=0))
    idx = int(np.argmax(pred))
    return {"label": _inv_label_map.get(idx, str(idx)), "confidence": float(np.max(pred))}
