import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from server.predictor import predict_from_video, load_model_and_labels

app = FastAPI(title="VSL Prediction API (Async)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Bộ nhớ tạm lưu kết quả xử lý
TASK_RESULTS = {}


@app.on_event("startup")
def startup_event():
    print("🔄 Loading model on startup...")
    try:
        load_model_and_labels()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model on startup: {e}")


@app.get("/")
def home():
    return {"status": "ok", "message": "VSL FastAPI is running!"}


# ========================
# 🚀 API upload video
# ========================
@app.post("/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """Nhận video và xử lý ngầm"""
    print("📩 File received:", file.filename)

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    task_id = uuid.uuid4().hex
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
    result_path = os.path.join(RESULT_DIR, f"{task_id}.json")

    # Lưu file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Đánh dấu trạng thái ban đầu
    TASK_RESULTS[task_id] = {"status": "processing", "result": None}

    # Xử lý ngầm
    background_tasks.add_task(run_prediction, file_path, result_path, task_id)

    return {"task_id": task_id, "status": "processing"}


def run_prediction(video_path: str, result_path: str, task_id: str):
    """Chạy nhận dạng ngầm"""
    try:
        print(f"🔮 [TASK {task_id}] Starting prediction...")
        result = predict_from_video(video_path)
        TASK_RESULTS[task_id] = {"status": "done", "result": result}

        # Lưu ra file (để kiểm tra lại)
        with open(result_path, "w", encoding="utf-8") as f:
            import json
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✅ [TASK {task_id}] Done: {result}")
    except Exception as e:
        TASK_RESULTS[task_id] = {"status": "error", "error": str(e)}
        print(f"❌ [TASK {task_id}] Error: {e}")
    finally:
        try:
            os.remove(video_path)
        except:
            pass


# ========================
# 📊 API lấy kết quả
# ========================
@app.get("/result/{task_id}")
def get_result(task_id: str):
    if task_id not in TASK_RESULTS:
        raise HTTPException(status_code=404, detail="Task ID not found")
    return TASK_RESULTS[task_id]
