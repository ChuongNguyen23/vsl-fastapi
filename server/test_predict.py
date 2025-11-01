import requests

API_URL = "https://vsl-fastapi.onrender.com/predict"
VIDEO_PATH = r"C:\Users\nvchu\Desktop\a\Dataset\Videos\D0008.mp4"
print("🎬 Sending video to server:", VIDEO_PATH)

with open(VIDEO_PATH, "rb") as f:
    files = {"file": (VIDEO_PATH, f, "video/mp4")}
    response = requests.post(API_URL, files=files)

print("📡 Status Code:", response.status_code)
try:
    print("🧩 Response JSON:", response.json())
except:
    print("⚠️ Raw Response:", response.text)
