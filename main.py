import data_loader
import train
import evaluate
import utils
import config
import os
import inference
import torch

def main():
    print("--- Car Engine Fault Detection System (Optimized CRNN) ---")
    
    # 1. Prepare Data
    print("\n[1/4] Preparing Data...")
    train_loader, val_loader, test_loader, classes, class_weights = data_loader.get_dataloaders()
    
    # 2. Training
    print("\n[2/4] Starting Training...")
    net, history = train.run_training(train_loader, val_loader, len(classes), class_weights=class_weights)
    
    # Plot Training History
    utils.plot_history(history, save_path=os.path.join(config.RESULTS_DIR, "training_history.png"))
    
    # 3. Evaluation
    print("\n[3/4] Evaluating Model on Test Set...")
    # Load best model for evaluation
    best_model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        net.load_state_dict(torch.load(best_model_path))
        
    evaluate.evaluate_model(net, test_loader, classes)
    
    # 4. Inference Demo
    print("\n[4/4] Running Inference Demo...")
    # For demo, let's pick the first file from the test folder
    test_dir = os.path.join(config.DATASET_DIR, "test")
    test_files, _ = utils.get_audio_files(test_dir)
    
    if test_files:
        sample_file = test_files[0]
        inference.run_inference_demo(sample_file)

if __name__ == "__main__":
    main()
