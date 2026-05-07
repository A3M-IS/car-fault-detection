import librosa
import numpy as np
import config

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_y = y + noise_factor * noise
    return augmented_y

def time_shift(y, shift_max=0.2):
    shift = np.random.randint(int(len(y) * shift_max))
    return np.roll(y, shift)

def pitch_shift(y, sr, n_steps=2):
    """Shift pitch by n_steps semitones."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate=1.1):
    """Stretch time by rate."""
    return librosa.effects.time_stretch(y=y, rate=rate)

def apply_spec_augment(mel_spec, time_mask_param=6, freq_mask_param=15):
    """Time and Frequency masking on spectrogram."""
    n_mels, time_frames = mel_spec.shape
    
    # Time Masking
    if time_mask_param > 0:
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, time_frames - t)
        mel_spec[:, t0:t0+t] = 0
        
    # Frequency Masking
    if freq_mask_param > 0:
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, n_mels - f)
        mel_spec[f0:f0+f, :] = 0
        
    return mel_spec

def extract_mel_spectrogram(file_path, augment=False):
    """
    Load audio file, preprocess it, and extract Mel Spectrogram.
    Optional: Apply audio-level and spectrogram-level augmentation.
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
        
        # Audio-level augmentation (only if augment=True)
        if augment:
            if np.random.random() > 0.5:
                y = add_noise(y, config.NOISE_FACTOR)
            if np.random.random() > 0.5:
                y = time_shift(y, config.TIME_SHIFT_MAX)
            if np.random.random() > 0.4:
                # Pitch shift +/- 2 semitones
                steps = np.random.uniform(-2, 2)
                y = pitch_shift(y, config.SAMPLE_RATE, steps)
            if np.random.random() > 0.4:
                # Time stretch 0.9x to 1.1x
                rate = np.random.uniform(0.9, 1.1)
                y = time_stretch(y, rate)
        
        # Normalize amplitude
        y = librosa.util.normalize(y)
        
        # Ensure it's exactly 1 second
        target_len = config.SAMPLE_RATE * config.DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, int(target_len - len(y))))
        else:
            y = y[:int(target_len)]
            
        # Extract Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=config.N_MELS, 
            n_fft=config.N_FFT, 
            hop_length=config.HOP_LENGTH
        )
        
        # Convert to decibel scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Spectrogram-level augmentation (only if augment=True)
        if augment:
            if np.random.random() > 0.5:
                mel_spec_db = apply_spec_augment(mel_spec_db, config.TIME_MASK_PARAM, config.FREQ_MASK_PARAM)
        
        # Ensure fixed shape
        if mel_spec_db.shape[1] < config.IMG_SIZE[1]:
            pad_width = config.IMG_SIZE[1] - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)))
        else:
            mel_spec_db = mel_spec_db[:, :config.IMG_SIZE[1]]
            
        # Normalize: mean=0, std=1 (Standardization)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        
        return mel_spec_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def preprocess_audio_batch(file_paths, augment=False):
    """Process a batch of audio files."""
    features = []
    valid_paths = []
    for path in file_paths:
        feat = extract_mel_spectrogram(path, augment=augment)
        if feat is not None:
            features.append(feat)
            valid_paths.append(path)
    return np.array(features), valid_paths
