import os
import shutil
import uuid
import threading
import time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from server.predictor import predict_from_video, load_model_and_labels

# ========================
# ⚙️ Khởi tạo FastAPI app
# ========================
app = FastAPI(title="Vietnamese Sign Language Recognition API")

# Cho phép Flutter gọi API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: *, production: ["https://ten-mien-cua-ban.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ========================
# 🧠 Khởi động server (load model 1 lần)
# ========================
@app.on_event("startup")
def startup_event():
    print("🔄 Loading model on startup...")
    try:
        load_model_and_labels()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model on startup: {e}")

    # 🔁 Keep-alive thread để Render không kill container
    def keep_alive():
        while True:
            print("💓 Server still alive...")
            time.sleep(30)
    threading.Thread(target=keep_alive, daemon=True).start()


# ========================
# 📡 Health check endpoint
# ========================
@app.get("/")
def health_check():
    return {
        "status": "✅ Server is alive",
        "message": "Vietnamese Sign Language FastAPI is running!",
    }


# ========================
# 🎥 API dự đoán video
# ========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"📩 File received: {file.filename}")

    # Kiểm tra định dạng file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        # Lưu file tạm
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"✅ File saved at: {file_path}")

        # Dự đoán
        print("🔮 Starting prediction...")
        result = predict_from_video(file_path)
        print(f"✅ Prediction completed: {result}")
        return result

    except Exception as e:
        print(f"❌ ERROR during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Xóa file tạm sau khi xử lý
        try:
            os.remove(file_path)
            print("🧹 Temporary file deleted.")
        except Exception as e:
            print(f"⚠️ File cleanup failed: {e}")
