import requests
import time
import os

API_URL = "https://vsl-fastapi.onrender.com"  # 🔁 đổi URL của bạn

video_path = r"C:\Users\nvchu\Desktop\a\Dataset\Videos\D0008.mp4"
print("🎬 Uploading video:", video_path)

with open(video_path, "rb") as f:
    r = requests.post(f"{API_URL}/upload", files={"file": f})

task = r.json()
print("✅ Task created:", task)

task_id = task["task_id"]

# Poll kết quả
while True:
    time.sleep(5)
    res = requests.get(f"{API_URL}/result/{task_id}")
    data = res.json()
    print("📡 Status:", data["status"])
    if data["status"] != "processing":
        print("🎯 Result:", data)
        break
