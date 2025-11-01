import os
import shutil
import uuid
import threading
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from server.predictor import predict_from_video, load_model_and_labels

# ========================
# ⚙️ Cấu hình FastAPI
# ========================
app = FastAPI(title="VSL Background Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================
# 🧠 Biến lưu trạng thái task
# ========================
tasks = {}  # task_id -> {"status": "processing" | "done" | "error", "result": {...}}


# ========================
# 🔄 Load model 1 lần
# ========================
@app.on_event("startup")
def startup_event():
    print("🔄 Loading model on startup...")
    try:
        load_model_and_labels()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")


# ========================
# 📡 Health check
# ========================
@app.get("/")
def root():
    return {"status": "ok", "message": "VSL FastAPI running"}


# ========================
# 🧵 Hàm chạy nền xử lý video
# ========================
def process_video_in_background(task_id, file_path):
    try:
        print(f"🧠 Background task started: {task_id}")
        result = predict_from_video(file_path)
        tasks[task_id] = {"status": "done", "result": result}
        print(f"✅ Task {task_id} done: {result}")
    except Exception as e:
        print(f"❌ Background error: {e}")
        tasks[task_id] = {"status": "error", "result": {"error": str(e)}}
    finally:
        try:
            os.remove(file_path)
        except:
            pass


# ========================
# 📤 Upload video (tạo task)
# ========================
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    task_id = uuid.uuid4().hex
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Tạo record "processing"
    tasks[task_id] = {"status": "processing", "result": None}

    # Tạo luồng nền xử lý
    thread = threading.Thread(target=process_video_in_background, args=(task_id, file_path))
    thread.start()

    return {"task_id": task_id, "status": "processing"}


# ========================
# 📥 Lấy kết quả theo task_id
# ========================
@app.get("/result/{task_id}")
def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]
