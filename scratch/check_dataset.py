import os
from collections import Counter

dataset_dir = r"c:\Machine Learning Project\car diagnostics dataset"

def check_dataset(dir_path):
    stats = Counter()
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".wav"):
                label = os.path.basename(root)
                stats[label] += 1
    return stats

if __name__ == "__main__":
    if os.path.exists(dataset_dir):
        print(f"Checking dataset at {dataset_dir}...")
        stats = check_dataset(dataset_dir)
        print("\nClass Distribution:")
        for label, count in stats.items():
            print(f"  {label}: {count} files")
    else:
        print(f"Directory {dataset_dir} not found.")
