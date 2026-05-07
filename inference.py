import torch
import joblib
import os
import preprocessing
import model
import config
import numpy as np

class EngineFaultDetector:
    def __init__(self, model_path, le_path):
        # Load Label Encoder
        self.le = joblib.load(le_path)
        self.classes = self.le.classes_
        
        # Load Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = model.build_model(len(self.classes))
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
        
    def predict(self, audio_file):
        """Predict the fault class and location."""
        # Preprocess
        feature = preprocessing.extract_mel_spectrogram(audio_file)
        if feature is None:
            return "Error", "Unknown"
            
        # Prepare for model (batch_size=1, channel=1, H, W)
        feature = np.expand_dims(feature, axis=(0, 1))
        feature_tensor = torch.tensor(feature, dtype=torch.float32).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.net(feature_tensor)
            _, predicted = outputs.max(1)
            
        class_idx = predicted.item()
        fault_class = self.classes[class_idx]
        
        # Map to fault location
        location = config.FAULT_MAPPING.get(fault_class, "Unknown")
        
        return fault_class, location

def run_inference_demo(audio_file):
    detector = EngineFaultDetector(
        os.path.join(config.MODELS_DIR, "best_model.pth"),
        os.path.join(config.MODELS_DIR, "label_encoder.joblib")
    )
    fault, location = detector.predict(audio_file)
    print(f"\nInference for {audio_file}:")
    print(f"Detected Fault: {fault}")
    print(f"Potential Location: {location}")
    return fault, location
