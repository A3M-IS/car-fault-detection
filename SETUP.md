# 🚗 Car Engine Fault Detection - Frontend & Backend Setup

## Quick Start

### 1. Install Backend Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model (FIRST TIME ONLY)
```bash
python main.py
```
This creates `models/best_model.pth` and `models/label_encoder.joblib`

### 3. Start Backend (Terminal 1)
```bash
python backend.py
```
Backend runs on: http://localhost:8000

### 4. Setup Frontend
```bash
cd frontend
npm install
```

### 5. Start Frontend (Terminal 2)
```bash
cd frontend
npm run dev
```
Frontend runs on: http://localhost:3000

---

## File Structure
```
.
├── backend.py                 # FastAPI backend
├── main.py                   # Model training
├── inference.py              # Inference engine
├── requirements.txt          # Python dependencies
└── frontend/                 # Next.js frontend
    ├── package.json
    ├── next.config.js
    ├── pages/
    │   ├── _app.js
    │   └── index.js
    ├── styles/
    │   └── globals.css
    └── utils/
        └── api.js
```

## API Endpoints

- `GET /` - Health check
- `POST /predict` - Upload audio and get prediction
- `GET /classes` - Get available fault classes

## Notes
- Main goal: Training the model for accuracy and fault detection
- Frontend is secondary UI for testing predictions
- Keep structure simple and focused
