import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from server.predictor import predict_from_video, load_model_and_labels

# ========================
# ⚙️ Khởi tạo FastAPI app
# ========================
app = FastAPI(title="VSL Prediction API")

# Cho phép Flutter gọi API
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
# 🚀 Khởi động server
# ========================
@app.on_event("startup")
def startup_event():
    print("🔄 Loading model on startup...")
    try:
        load_model_and_labels()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model on startup: {e}")


# ========================
# 📡 API health check
# ========================
@app.get("/")
def root():
    return {"status": "ok", "message": "VSL FastAPI is running!"}


# ========================
# 🎥 API dự đoán video
# ========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print("📩 File received:", file.filename)

    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    try:
        # Save file to server
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("✅ File saved at:", file_path)

        # Run prediction
        print("🔮 Starting prediction...")
        result = predict_from_video(file_path)
        print("✅ Prediction completed:", result)

        return result

    except Exception as e:
        print("❌ ERROR while predicting:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            os.remove(file_path)
            print("🧹 Temporary file deleted.")
        except Exception as e:
            print("⚠️ File cleanup failed:", e)

@app.get("/")
def home():
    return {"status": "ok", "message": "VSL FastAPI is running!"}


