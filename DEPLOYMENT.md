# 🚀 Live Deployment Guide

Follow these steps to deploy your **Engine Neural Diagnostic System** to the cloud.

## 🛠️ Prerequisites
1. A **GitHub** repository with your project code.
2. A **Vercel** account (for Frontend).
3. A **Render.com** account (for Backend).

---

## 🛰️ Step 1: Deploy the AI Backend (Render)
The backend handles the PyTorch model and sound analysis.

1.  **Create a New Web Service** on Render.
2.  Connect your GitHub repository.
3.  **Configure Environment**:
    *   **Runtime**: `Python`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `gunicorn -w 2 -k uvicorn.workers.UvicornWorker backend:app --bind 0.0.0.0:$PORT --timeout 120`
4.  **Add Environment Variables** (Advanced section):
    *   `ALLOWED_ORIGINS`: `https://your-frontend-url.vercel.app` (You will update this after Step 2).
5.  **Wait for deployment**: Once finished, copy your Backend URL (e.g., `https://car-ai-api.onrender.com`).

---

## 🎨 Step 2: Deploy the Frontend (Vercel)
The dashboard provides the user interface.

1.  **Import your Project** on Vercel.
2.  **Configure the Build**:
    *   **Root Directory**: `frontend`
    *   **Framework Preset**: `Next.js`
3.  **Add Environment Variables**:
    *   `NEXT_PUBLIC_API_URL`: Paste your **Backend URL** from Step 1.
4.  **Deploy**: Hit the deploy button. Once finished, copy your Frontend URL.

---

## 🔗 Step 3: Final Sync
1.  Go back to your **Render Dashboard**.
2.  Update the `ALLOWED_ORIGINS` variable with your real **Vercel URL**.
3.  Restart the Render service.

---

## ✅ Post-Deployment Checklist
- [ ] **Model Weights**: Ensure `models/best_model.pth` is pushed to GitHub.
- [ ] **Health Check**: Visit `https://your-api-url.onrender.com/` to see if it says "status: running".
- [ ] **CORS**: Try uploading a file from the Vercel app to verify communication.

**Note**: Since Render's free tier "sleeps" after 15 minutes of inactivity, the first check-up might take 30-60 seconds to wake up the server.
