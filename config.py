import os

# Paths
DATASET_DIR = r"c:\Machine Learning Project\car diagnostics dataset"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Audio Preprocessing
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
N_MELS = 128
HOP_LENGTH = 125 # Changed from 512 to get 128 frames for 1s @ 16kHz
N_FFT = 1024
IMG_SIZE = (128, 128) # Padded/Resized to match user recommendation

# Dataset Split
TRAIN_SIZE = 0.80
TEST_SIZE = 0.20
VAL_SIZE = 0.0  # Optional, can split from TRAIN_SIZE if needed

# Training Hyperparameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 3e-4 # Slightly lower for better convergence
WEIGHT_DECAY = 5e-3 # Increased from 1e-4 to penalize large weights (regularization)
LABEL_SMOOTHING = 0.1 # Increased from 0.05

# Early Stopping and Scheduler
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 7
LR_FACTOR = 0.5

# CRNN Architecture
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
DROPOUT = 0.5 # Increased from 0.3 to reduce overfitting

# Data Augmentation
USE_AUGMENTATION = True
USE_MIXUP = True
NOISE_FACTOR = 0.01 # Lowered for stability
TIME_SHIFT_MAX = 0.1
TIME_MASK_PARAM = 4 # ~12%
FREQ_MASK_PARAM = 10 # ~8%
MIXUP_ALPHA = 0.2

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fault mapping (example based on subfolders)
FAULT_MAPPING = {
    "normal_brakes": "Normal",
    "worn_out_brakes": "Braking system",
    "no oil_serpentine belt": "Oil & Belt system",
    "power steering combined_no oil": "Power steering & Oil system",
    "power steering combined_no oil_serpentine belt": "Multiple systems",
    "power steering combined_serpentine belt": "Power steering & Belt system",
    "low_oil": "Oil system",
    "normal_engine_idle": "Normal",
    "power_steering": "Power steering system",
    "serpentine_belt": "Belt system",
    "bad_ignition": "Combustion chamber",
    "dead_battery": "Battery / Electrical",
    "normal_engine_startup": "Normal"
}
