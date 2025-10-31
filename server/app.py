# server/app.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil, uuid
from server.predictor import predict_from_video, load_model_and_labels

app = FastAPI(title="VSL Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: * ; production: list của domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pre-load model on startup to avoid loading per request
@app.on_event("startup")
def startup_event():
    load_model_and_labels()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)
    # save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_from_video(file_path)
    except Exception as e:
        result = {"error": str(e)}
    finally:
        try:
            os.remove(file_path)
        except:
            pass

    return result
