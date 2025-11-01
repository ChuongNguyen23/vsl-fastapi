import requests
import os

# ===========================
# ⚙️ Cấu hình
# ===========================
API_URL = "https://vsl-fastapi.onrender.com/predict"

# Đường dẫn video cần test — bạn đổi lại nếu muốn
VIDEO_PATH = r"C:\Users\nvchu\Desktop\a\Dataset\Videos\D0008.mp4"

# ===========================
# 🚀 Gửi video tới server
# ===========================
if not os.path.exists(VIDEO_PATH):
    print("❌ Không tìm thấy file video:", VIDEO_PATH)
    exit()

print("🎬 Sending video to server:", VIDEO_PATH)

try:
    with open(VIDEO_PATH, "rb") as f:
        files = {"file": (os.path.basename(VIDEO_PATH), f, "video/mp4")}
        r = requests.post(API_URL, files=files, timeout=180)

    print("📡 Status Code:", r.status_code)

    if r.status_code == 200:
        print("✅ Prediction result:")
        print(r.json())
    else:
        print("⚠️ Server returned error:")
        print(r.text)

except requests.exceptions.ConnectionError:
    print("❌ Không thể kết nối tới API. Kiểm tra lại URL:", API_URL)

except requests.exceptions.Timeout:
    print("⏱️ Quá thời gian chờ phản hồi từ server. Có thể model đang xử lý video dài.")

except Exception as e:
    print("❌ Lỗi không xác định:", str(e))
