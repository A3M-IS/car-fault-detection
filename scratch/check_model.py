import torch
import sys
import os
sys.path.append(os.getcwd())

import model
import config
import joblib
import preprocessing
import numpy as np

def test_model():
    model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
    le_path = os.path.join(config.MODELS_DIR, "label_encoder.joblib")
    
    if not os.path.exists(model_path):
        print("Model file not found.")
        return
        
    le = joblib.load(le_path)
    classes = le.classes_
    print(f"Classes: {classes}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.build_model(len(classes)).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    
    # Test on 5 random samples
    data_dir = os.path.join(config.DATASET_DIR, "train")
    count = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav") and count < 10:
                path = os.path.join(root, file)
                feat = preprocessing.extract_mel_spectrogram(path)
                if feat is not None:
                    feat = np.expand_dims(feat, axis=(0, 1))
                    feat_t = torch.tensor(feat, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        out = net(feat_t)
                        probs = torch.softmax(out, dim=1)
                        conf, pred = out.max(1)
                        print(f"Actual: {os.path.basename(root)} -> Predicted: {classes[pred.item()]} ({probs[0][pred.item()].item()*100:.1f}%)")
                        count += 1
                if count >= 10: break
        if count >= 10: break

if __name__ == "__main__":
    test_model()
