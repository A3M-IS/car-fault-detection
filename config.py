import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATASET_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FEATURE_CACHE_DIR = os.path.join(RESULTS_DIR, "feature_cache")

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
BATCH_SIZE = 48
EPOCHS = 60
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 5e-3
LABEL_SMOOTHING = 0.05
USE_CLASS_WEIGHTED_LOSS = False

# Early Stopping and Scheduler
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 4
LR_FACTOR = 0.5

# CRNN Architecture
LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 2
DROPOUT = 0.5

# Data Augmentation
USE_AUGMENTATION = True
USE_MIXUP = True
USE_BALANCED_SAMPLING = True
NOISE_FACTOR = 0.01
TIME_SHIFT_MAX = 0.15
TIME_MASK_PARAM = 6
FREQ_MASK_PARAM = 14
MIXUP_ALPHA = 0.2
USE_PITCH_SHIFT_AUG = False
USE_TIME_STRETCH_AUG = False

# Runtime speed options
NUM_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))
ENABLE_FEATURE_CACHE = True

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

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
