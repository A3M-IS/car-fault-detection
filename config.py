import os

# Paths
DATASET_DIR = "data"
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
EPOCHS = 80  # Reduced to allow early stopping to trigger (was 100)
LEARNING_RATE = 3e-4 # Slightly lower for better convergence
WEIGHT_DECAY = 1e-2 # Increased to 1e-2 from 5e-3 for stronger L2 regularization
LABEL_SMOOTHING = 0.1 # Increased from 0.05

# Early Stopping and Scheduler
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 7
LR_FACTOR = 0.5

# CRNN Architecture
LSTM_HIDDEN_SIZE = 64  # Reduced from 128 to constrain model capacity and prevent overfitting
LSTM_LAYERS = 2
DROPOUT = 0.5 # Increased from 0.3 to reduce overfitting

# Data Augmentation
USE_AUGMENTATION = True
USE_MIXUP = True
USE_BALANCED_SAMPLING = True
NOISE_FACTOR = 0.015 # Increased from 0.01 for stronger noise augmentation
TIME_SHIFT_MAX = 0.3  # Increased from 0.1 for more aggressive shifting
TIME_MASK_PARAM = 8   # Increased from 4 (~25% of time masked)
FREQ_MASK_PARAM = 20  # Increased from 10 (~16% of frequency masked)
MIXUP_ALPHA = 0.3     # Increased from 0.2 for stronger mixup effect

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fault mapping (Grouped to reduce acoustic overlap and include new data)
FAULT_MAPPING = {
    "normal_brakes": "Normal",
    "worn_out_brakes": "Braking system",
    "no oil_serpentine belt": "Belt & Accessory System",
    "power steering combined_no oil": "Belt & Accessory System",
    "power steering combined_no oil_serpentine belt": "Belt & Accessory System",
    "power steering combined_serpentine belt": "Belt & Accessory System",
    "low_oil": "Oil system",
    "normal_engine_idle": "Normal",
    "power_steering": "Belt & Accessory System",
    "serpentine_belt": "Belt & Accessory System",
    "bad_ignition": "Combustion & Ignition",
    "dead_battery": "Battery / Electrical",
    "normal_engine_startup": "Normal",
    # New from unprocessedData
    "air_leak": "Combustion & Ignition",
    "oil_cap_off": "Oil system",
    "background_noise": "Normal"
}

# Canonical source labels used by the merge script.
MERGE_LABEL_ALIASES = {
    "normal engine inside cabin": "normal_engine_idle",
    "normal engine": "normal_engine_idle",
    "idling": "normal_engine_idle",
    "air leak engine inside cabin": "air_leak",
    "air leak": "air_leak",
    "oil cap off engine inside cabin": "oil_cap_off",
    "oil cap off": "oil_cap_off",
    "background noise": "background_noise",
}
