# server/test_predict.py
import requests
import os
import sys
from pathlib import Path

# --- CẬP NHẬT ĐÚNG URL API của bạn ở đây ---
API_URL = "https://vsl-fastapi.onrender.com/predict"
# ----------------------------------------

VIDEO_PATH = r"C:\Users\nvchu\Desktop\a\Dataset\Videos\D0008.mp4"  # chỉnh đường dẫn nếu cần

def make_small_clone(src_path, max_seconds=3):
    """
    Nếu video quá lớn, bạn có thể tạo bản cắt ngắn để test (cần ffmpeg trên PATH).
    Nếu không có ffmpeg, function trả về src_path.
    """
    try:
        from subprocess import run, CalledProcessError
        tmp = Path(src_path).with_suffix(".short.mp4")
        cmd = ["ffmpeg", "-y", "-i", str(src_path), "-t", str(max_seconds), "-c", "copy", str(tmp)]
        run(cmd, check=True, capture_output=True)
        if tmp.exists():
            print(f"✅ Created short test video: {tmp}")
            return str(tmp)
    except Exception as e:
        print("ℹ️ ffmpeg not available or trimming failed:", e)
    return src_path

def send_video(path):
    print("🎬 Sending video to server:", path)
    # dùng timeout lớn (120s) — nhưng render có giới hạn, nếu server xử lý lâu vẫn sẽ bị kill
    timeout_seconds = 120

    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "video/mp4")}
        try:
            # stream=True để không tải toàn bộ response vào bộ nhớ trước khi in ra
            resp = requests.post(API_URL, files=files, timeout=timeout_seconds, stream=True)
        except requests.exceptions.RequestException as e:
            print("❌ Request failed:", repr(e))
            return None

    print("📡 Status Code:", resp.status_code)
    try:
        # đọc và in body an toàn (nếu server trả chunked lởm thì vẫn cố gắng in phần đã nhận)
        body = resp.content.decode(errors="replace")
        print("⚙️ Response body:")
        print(body)
    except Exception as e:
        print("⚠️ Could not decode response body:", e)
        # Thử in phần text (requests.text có thể gây lại lỗi chunked)
        try:
            print(resp.text)
        except Exception:
            print("<no body available>")

    return resp

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print("❗ File not found:", VIDEO_PATH)
        sys.exit(1)

    # thử tạo bản ngắn để test (bỏ comment nếu muốn cắt)
    test_path = make_small_clone(VIDEO_PATH, max_seconds=3)
    send_video(test_path)
