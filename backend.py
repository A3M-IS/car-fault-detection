from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import os
import torch
from inference import EngineFaultDetector
import config

app = FastAPI(title="Car Engine Fault Detection API", version="1.0.0")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = None

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global detector
    model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
    le_path = os.path.join(config.MODELS_DIR, "label_encoder.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print(f"⚠️ Model files not found at {model_path} or {le_path}")
        print("Please run: python main.py")
        detector = None
    else:
        try:
            detector = EngineFaultDetector(model_path, le_path)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            detector = None

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
    """
    Upload audio file and get fault prediction
    
    - **file**: WAV or MP3 audio file (max 5MB, 0.5-10 seconds)
    
    Returns:
    - fault_class: Detected fault category
    - location: Engine component location
    - confidence: Model confidence score
    """
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first using: python main.py"
        )
    
    # Validate file size
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: 5MB, Got: {len(contents)/1024/1024:.1f}MB"
        )
    
    try:
        # Write bytes to a temporary file because librosa expects a file path
        import tempfile
        _, ext = os.path.splitext(file.filename or '')
        if not ext:
            ext = '.wav'

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Run prediction using temp file path
        fault_class, location = detector.predict(tmp_path)

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return {
            "success": True,
            "fault_class": fault_class.replace("_", " ").title(),
            "location": location,
            "is_normal": "normal" in fault_class.lower()
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing audio: {str(e)}"
        )

@app.get("/classes")
async def get_classes():
    """Get list of fault classes the model can detect"""
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
