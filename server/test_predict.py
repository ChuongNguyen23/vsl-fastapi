# server/test_predict.py
import requests

video_path = r"C:\Users\nvchu\Desktop\a\Dataset\Videos\D0008.mp4"
url = "https://vsl-fastapi.onrender.com/predict"

print(f"🎬 Sending video to server: {video_path}")
with open(video_path, "rb") as f:
    files = {"file": ("test.mp4", f, "video/mp4")}
    r = requests.post(url, files=files)

print(f"📡 Status Code: {r.status_code}")
print(f"⚙️ Response: {r.text}")
