import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import utils
import preprocessing
import config
import joblib
import os

class EngineAudioDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess with optional augmentation
        feature = preprocessing.extract_mel_spectrogram(file_path, augment=self.augment)
        if feature is None:
            feature = np.zeros((config.N_MELS, config.IMG_SIZE[1]))
            
        # Add channel dimension: (1, 128, 128)
        feature = np.expand_dims(feature, axis=0)
        
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def prepare_data():
    """Load files from train/test directories and encode labels."""
    train_dir = os.path.join(config.DATASET_DIR, "train")
    test_dir = os.path.join(config.DATASET_DIR, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("Data not organized yet. Organizing now...")
        import organize_dataset
        organize_dataset.organize_dataset()
        
    train_files, train_labels_folders = utils.get_audio_files(train_dir)
    test_files, test_labels_folders = utils.get_audio_files(test_dir)
    
    # Apply FAULT_MAPPING from config
    train_labels_raw = [config.FAULT_MAPPING.get(l, l) for l in train_labels_folders]
    test_labels_raw = [config.FAULT_MAPPING.get(l, l) for l in test_labels_folders]
    
    if not train_files or not test_files:
        raise ValueError("Training or Testing files not found.")
    
    # Label Encoding
    le = LabelEncoder()
    # Use grouped labels
    all_labels = sorted(list(set(train_labels_raw + test_labels_raw)))
    le.fit(all_labels)
    
    train_labels = le.transform(train_labels_raw)
    test_labels = le.transform(test_labels_raw)
    
    # Save Label Encoder
    joblib.dump(le, os.path.join(config.MODELS_DIR, "label_encoder.joblib"))
    
    # Optional: Split a small validation set from training data (e.g., 10% of train)
    # Since user asked for 80/20, we'll keep the test set as the 20%.
    # We can split the 80% into 70% train and 10% val if we want, or just 80% train.
    # Let's do 90/10 split of the 80% for validation.
    
    train_files_split, val_files, train_labels_split, val_labels = train_test_split(
        train_files, train_labels, test_size=0.1, 
        random_state=42, stratify=train_labels
    )
    
    # Calculate Class Weights for Imbalance Handling
    classes = le.classes_
    class_counts = np.bincount(train_labels_split)
    total_samples = len(train_labels_split)
    # weights = total / (num_classes * count)
    class_weights = total_samples / (len(classes) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    sampler = None
    if config.USE_BALANCED_SAMPLING:
        sample_weights = class_weights[train_labels_split].double()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
    
    print(f"Dataset Split: Train={len(train_files_split)}, Val={len(val_files)}, Test={len(test_files)}")
    print(f"Class weights: {class_weights}")
    
    return (train_files_split, train_labels_split), (val_files, val_labels), (test_files, test_labels), classes, class_weights, sampler

def get_dataloaders():
    """Create PyTorch DataLoaders."""
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels), classes, class_weights, sampler = prepare_data()
    
    train_ds = EngineAudioDataset(train_files, train_labels, augment=config.USE_AUGMENTATION)
    val_ds = EngineAudioDataset(val_files, val_labels, augment=False)
    test_ds = EngineAudioDataset(test_files, test_labels, augment=False)
    
    use_workers = max(0, int(config.NUM_WORKERS))
    persistent = use_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False if sampler is not None else True,
        sampler=sampler,
        num_workers=use_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=use_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=use_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
    )
    
    return train_loader, val_loader, test_loader, classes, class_weights
