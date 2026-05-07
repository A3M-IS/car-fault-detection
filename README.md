# Engine Neural Diagnostic System 🚗🔬

An advanced automotive diagnostic platform that utilizes Deep Learning (CRNN) to detect engine faults through acoustic analysis.

## 🚀 Features
- **Neural Diagnostic Engine**: CRNN model (CNN + LSTM) trained on diversified engine sound profiles.
- **AI-Powered UI**: Modern, glassmorphism-based dashboard built with Next.js.
- **Real-time Capture**: Record engine sounds directly from the browser for instant analysis.
- **Robust Regularization**: Implements Mixup augmentation, Spatial Dropout, and Label Smoothing to prevent overfitting.

## 🛠️ Tech Stack
- **Backend**: Python, PyTorch, Librosa, FastAPI (FastAPI implementation in `backend.py`).
- **Frontend**: Next.js 14, React 18, Lucide Icons, Vanilla CSS.
- **Processing**: Mel-Spectrogram extraction at 16kHz.

## 📦 Getting Started

### 1. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Merge diversified dataset (Optional if data already present)
python merge_unprocessed_data.py

# Train the model
python main.py

# Start the API server
python backend.py
```

### 2. Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## 📁 Project Structure
- `/data`: Training/Testing audio samples (Excluded from Git).
- `/frontend`: Next.js application.
- `/models`: Saved PyTorch model weights (Excluded from Git).
- `/results`: Training logs and confusion matrices.
- `model.py`: EngineCRNN architecture.
- `preprocessing.py`: Audio augmentation and feature extraction logic.

## ⚠️ Note on Data
The raw audio data and trained model weights are excluded from Git to maintain a lightweight repository. Please ensure you have the dataset locally in the `data/` folder before training.
