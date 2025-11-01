import requests

API_URL = "https://vsl-fastapi.onrender.com/predict"
video_path = r"C:\Users\nvchu\Desktop\a\Dataset\Videos\D0008.mp4"

print("🎬 Sending video to server:", video_path)
files = {"file": open(video_path, "rb")}
r = requests.post(API_URL, files=files)

print("📡 Status Code:", r.status_code)
print("⚙️ Raw Response:")
print(r.text)

if r.headers.get("content-type", "").startswith("application/json"):
    print("✅ JSON Response:", r.json())
else:
    print("❌ Server did not return JSON — maybe 502 or internal error.")
