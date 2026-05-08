from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from starlette.concurrency import run_in_threadpool
import io
import os
import torch
from inference import EngineFaultDetector
import config

detector = None
model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
le_path = os.path.join(config.MODELS_DIR, "label_encoder.joblib")
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm", ".opus"}
ALLOWED_AUDIO_MIME_PREFIXES = ("audio/",)
MIN_AUDIO_SIZE_BYTES = 1024


def _load_detector():
    global detector
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print(f"⚠️ Model files not found at {model_path} or {le_path}")
        print("Please run: python main.py")
        detector = None
        return

    try:
        detector = EngineFaultDetector(model_path, le_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_detector()

    yield


app = FastAPI(title="Car Engine Fault Detection API", version="1.0.0", lifespan=lifespan)


def ensure_latest_detector():
    """Ensure the model is loaded. Does NOT reload on every request to prevent latency."""
    global detector
    if detector is None:
        print("🚀 Detector is None, attempting to load...")
        _load_detector()


def _is_probably_audio_upload(upload_file: UploadFile) -> bool:
    filename = upload_file.filename or ""
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    content_type = (upload_file.content_type or "").lower()

    if ext in ALLOWED_AUDIO_EXTENSIONS:
        return True

    return any(content_type.startswith(prefix) for prefix in ALLOWED_AUDIO_MIME_PREFIXES)

# Permissive CORS for production stability
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API Health check"""
    model_status = "loaded" if detector else "not loaded"
    return {
        "status": "running",
        "model": model_status,
        "message": "Car Engine Fault Detection API"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lazy load only if absolutely necessary
    if detector is None:
        ensure_latest_detector()

    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using: python main.py"
        )

    if not _is_probably_audio_upload(file):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a WAV, MP3, M4A, OGG, FLAC, AAC, WEBM, or OPUS audio file."
        )
    
    # Validate file size
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    contents = await file.read()
    if len(contents) < MIN_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Audio file is too small or empty. Please upload a valid recording."
        )
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: 5MB, Got: {len(contents)/1024/1024:.1f}MB"
        )
    
    try:
        # Use a temporary file for librosa compatibility and stability
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            print(f"📝 Writing {len(contents)} bytes to temp file: {tmp.name}")
            tmp.write(contents)
            tmp_path = tmp.name

        print("🧠 Running AI Inference...")
        # Run prediction in a thread pool to avoid blocking the event loop
        fault_class, location = await run_in_threadpool(detector.predict, tmp_path)
        print(f"✅ Prediction complete: {fault_class} at {location}")

        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass

        return {
            "success": True,
            "fault_class": fault_class.replace("_", " ").title(),
            "location": location,
            "is_normal": "normal" in fault_class.lower()
        }
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

@app.get("/classes")
async def get_classes():
    """Get list of fault classes the model can detect"""
    if detector is None:
        ensure_latest_detector()
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": list(detector.classes),
        "fault_mapping": config.FAULT_MAPPING
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
