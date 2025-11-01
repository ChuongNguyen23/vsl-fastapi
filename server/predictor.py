import os
import json
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d
import gdown

# ======================
# ⚙️ Cấu hình
# ======================
mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS * 2

MODEL_LOCAL_PATH = "Models/checkpoints/final_model.keras"
LABEL_MAP_PATH = "Logs/label_map.json"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1jIXbNFG4nl401WcNhv-FNMwNc3CvE4IR"

# ======================
# 📥 Tải model nếu chưa có
# ======================
def download_model_if_needed():
    if os.path.exists(MODEL_LOCAL_PATH):
        print("✅ Model found locally, skipping download.")
        return MODEL_LOCAL_PATH

    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    print("⬇️ Downloading model from:", MODEL_URL)
    gdown.download(MODEL_URL, MODEL_LOCAL_PATH, quiet=False)
    print("✅ Model downloaded successfully.")
    return MODEL_LOCAL_PATH


# ======================
# 🧠 Load model và label map (1 lần duy nhất)
# ======================
_model = None
_inv_label_map = None

def load_model_and_labels():
    global _model, _inv_label_map
    if _model is not None:
        return

    model_path = download_model_if_needed()
    print(f"📦 Loading model from {model_path} ...")
    _model = tf.keras.models.load_model(model_path)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    _inv_label_map = {v: k for k, v in label_map.items()}

    print("✅ Model and label map loaded.")


# ======================
# 🧩 Hàm xử lý video
# ======================
def mediapipe_detection(image, holistic_model):
    """Chạy Mediapipe detection trên 1 frame"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results


def extract_keypoints(results):
    """Trích xuất keypoints từ kết quả Mediapipe"""
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
    """Nội suy dãy keypoints về độ dài cố định"""
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


# ======================
# 🔍 Hàm chính dự đoán
# ======================
def predict_from_video(video_path):
    """Xử lý video đầu vào và dự đoán ký hiệu"""
    print(f"🎬 Processing video: {video_path}")

    if _model is None:
        load_model_and_labels()

    seq = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video file.")
        return {"error": "Cannot open video file"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // 100)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
                continue
            try:
                _, results = mediapipe_detection(frame, holistic)
                seq.append(extract_keypoints(results))
            except Exception as e:
                print(f"⚠️ Frame processing error: {e}")
                continue

    cap.release()
    print(f"📹 Extracted {len(seq)} frames.")

    if not seq:
        return {"error": "No keypoints extracted from video"}

    kp = interpolate_keypoints(seq)
    pred = _model.predict(np.expand_dims(kp, axis=0))
    idx = int(np.argmax(pred))
    confidence = float(np.max(pred))
    label = _inv_label_map.get(idx, f"class_{idx}")

    print(f"✅ Prediction done: {label} ({confidence:.2f})")
    return {"label": label, "confidence": confidence}
