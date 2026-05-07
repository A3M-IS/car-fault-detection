import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import utils
import config
import os

def evaluate_model(model, loader, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=classes)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(os.path.join(config.RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        
    # Plot Confusion Matrix
    utils.plot_confusion_matrix(
        all_labels, all_preds, classes, 
        save_path=os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    )
    
    return all_labels, all_preds
