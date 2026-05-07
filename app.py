import streamlit as st
import torch
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import librosa.display
from inference import EngineFaultDetector
import config
import preprocessing
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Car Engine Fault Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        border: 1px solid #4caf50;
    }
    .result-card {
        background-color: #1e1e1e;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #333;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-normal {
        color: #4caf50;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .status-fault {
        color: #f44336;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .diagnostic-info {
        color: #9e9e9e;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Detector
@st.cache_resource
def load_detector():
    model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
    le_path = os.path.join(config.MODELS_DIR, "label_encoder.joblib")
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        return None
    return EngineFaultDetector(model_path, le_path)

detector = load_detector()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/car-service.png", width=100)
    st.title("Settings")
    st.info("This system uses a Deep Learning (CRNN) model to analyze engine sounds and detect mechanical faults.")
    
    if detector is None:
        st.error("Model files not found! Please train the model first by running `python main.py`.")
    else:
        st.success("Model loaded successfully!")

# Main UI
st.title("🚗 Car Engine Fault Detection System")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Engine Audio")
    st.info("""
    **💡 Best Practices:**
    *   **Format:** .wav or .mp3
    *   **Length:** 1 to 5 seconds
    *   **Content:** Clean engine sound without music or loud speech.
    *   **Size:** Max 5MB
    """)
    
    uploaded_file = st.file_uploader("Choose an engine recording...", type=["wav", "mp3"], help="Upload the audio file you want the AI to analyze.")
    
    if uploaded_file is not None:
        # 1. Check File Size
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ File too large! Please upload a file smaller than 5MB. (Current: {uploaded_file.size/1024/1024:.1f}MB)")
            uploaded_file = None
        else:
            # 2. Check Duration
            try:
                duration = librosa.get_duration(path=uploaded_file)
                if duration < 0.5:
                    st.warning("⚠️ Recording is very short. It might not contain enough diagnostic data.")
                elif duration > 10:
                    st.error(f"❌ Recording is too long ({duration:.1f}s). Please upload a shorter clip (under 10 seconds) focused on the engine sound.")
                    uploaded_file = None
                else:
                    st.success(f"✅ Audio Validated: {duration:.1f}s | {uploaded_file.size/1024:.1f}KB")
            except Exception as e:
                st.error(f"❌ Invalid audio file: {e}")
                uploaded_file = None

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Analyze Button
        if st.button("🚀 Run Diagnostic"):
            if detector:
                with st.spinner("Analyzing audio fingerprint..."):
                    # Process prediction
                    # Save temporarily for librosa (it likes files or bytes)
                    # For simplicity, we'll pass the uploaded_file directly as librosa handles it
                    fault, location = detector.predict(uploaded_file)
                    
                    # Store results in session state
                    st.session_state.result = (fault, location)
            else:
                st.error("Cannot run diagnostic: Model not loaded.")

with col2:
    st.subheader("📊 Visual Analysis")
    if uploaded_file is not None:
        # Load audio for visualization
        y, sr = librosa.load(uploaded_file, sr=config.SAMPLE_RATE, duration=config.DURATION)
        
        # Plot Waveform
        fig_wave, ax_wave = plt.subplots(figsize=(10, 3), facecolor='#0e1117')
        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#2e7d32')
        ax_wave.set_title("Time-Domain Waveform", color='white')
        ax_wave.set_axis_off()
        st.pyplot(fig_wave)
        
        # Plot Spectrogram (what the model sees)
        mel_spec = preprocessing.extract_mel_spectrogram(uploaded_file)
        if mel_spec is not None:
            fig_spec, ax_spec = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
            img = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=sr, hop_length=config.HOP_LENGTH, ax=ax_spec, cmap='viridis')
            ax_spec.set_title("Frequency-Domain (Mel Spectrogram)", color='white')
            st.pyplot(fig_spec)

# Prediction Result Display
if 'result' in st.session_state and uploaded_file is not None:
    fault, location = st.session_state.result
    
    st.markdown("---")
    st.subheader("🏁 Diagnostic Result")
    
    is_normal = "normal" in fault.lower()
    status_class = "status-normal" if is_normal else "status-fault"
    status_text = "SYSTEM HEALTHY" if is_normal else "FAULT DETECTED"
    icon = "✅" if is_normal else "🚨"
    
    st.markdown(f"""
        <div class="result-card">
            <div class="{status_class}">{icon} {status_text}</div>
            <div style="font-size: 1.8rem; color: white; margin-top: 10px;">
                <strong>Identified Category:</strong> {fault.replace('_', ' ').title()}
            </div>
            <div class="diagnostic-info">
                <strong>Target System:</strong> {location}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if not is_normal:
        st.warning(f"**Action Required:** Inspect the **{location}** immediately. The acoustic pattern matches characteristics of **{fault.replace('_', ' ')}**.")
    else:
        st.balloons()
        st.success("The engine sound profile appears consistent with normal operating parameters.")

st.markdown("---")
st.caption("Developed with Antigravity AI | Version 1.0.0")
