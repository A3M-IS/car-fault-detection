import os
import shutil
from sklearn.model_selection import train_test_split
import config
import utils

def organize_dataset():
    """
    Physically split the dataset into train and test directories.
    """
    base_dir = config.DATASET_DIR
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    # Create train and test directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all audio files and labels
    audio_files, labels = utils.get_audio_files(base_dir)
    
    # Filter out files already in train or test (to avoid recursion issues)
    filtered_files = []
    filtered_labels = []
    for f, l in zip(audio_files, labels):
        if "train" not in f and "test" not in f:
            filtered_files.append(f)
            filtered_labels.append(l)
            
    if not filtered_files:
        print("No files found to organize (or already organized).")
        return

    # Split dataset
    train_files, test_files, train_labels, test_labels = train_test_split(
        filtered_files, filtered_labels, 
        test_size=config.TEST_SIZE, 
        random_state=42, 
        stratify=filtered_labels
    )
    
    # Function to move files
    def move_files(files, target_root):
        for f in files:
            # Get relative path to maintain class structure
            # Root is config.DATASET_DIR
            # f is absolute path
            rel_path = os.path.relpath(f, base_dir)
            target_path = os.path.join(target_root, rel_path)
            
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(f, target_path) # Copy instead of move to be safe
            
    print(f"Organizing {len(train_files)} files into {train_dir}...")
    move_files(train_files, train_dir)
    
    print(f"Organizing {len(test_files)} files into {test_dir}...")
    move_files(test_files, test_dir)
    
    print("Dataset organization complete.")

if __name__ == "__main__":
    organize_dataset()
