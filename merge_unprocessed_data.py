import os
import shutil
import json
import re
from collections import Counter
from tqdm import tqdm
import config

# Paths
UNPROCESSED_DATA_DIR = r"c:\anw\project\car-fault-detection\car-fault-detection\unprocessedData"
PROJECT_DATA_DIR = r"c:\anw\project\car-fault-detection\car-fault-detection\data"
RESULTS_DIR = r"c:\anw\project\car-fault-detection\car-fault-detection\results"
MERGE_SUMMARY_PATH = os.path.join(RESULTS_DIR, "merge_summary.json")

# Target directories (relative to data/train/)
TARGET_MAP = {
    "normal_engine_idle": os.path.join("idle state", "normal_engine_idle"),
    "air_leak": os.path.join("idle state", "air_leak"),
    "oil_cap_off": os.path.join("idle state", "oil_cap_off"),
    "background_noise": os.path.join("idle state", "background_noise")
}

SUPPORTED_SOURCE_LABELS = set(TARGET_MAP.keys())


def normalize_text(value):
    value = value.replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", value)


def sanitize_filename(filename):
    base, ext = os.path.splitext(os.path.basename(filename))
    base = normalize_text(base)
    base = re.sub(r"[^a-z0-9.-]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return f"{base or 'sample'}{ext.lower() or '.wav'}"


def resolve_aim_label(raw_label):
    normalized = normalize_text(raw_label)
    return config.MERGE_LABEL_ALIASES.get(normalized)

def merge_ai_mechanic():
    print("Processing ai-mechanic-export...")
    export_dir = os.path.join(UNPROCESSED_DATA_DIR, "ai-mechanic-export")
    labels_file = os.path.join(export_dir, "info.labels")
    
    if not os.path.exists(labels_file):
        print(f"Labels file not found: {labels_file}")
        return

    with open(labels_file, 'r') as f:
        data = json.load(f)
        
    count = 0
    skipped = Counter()
    copied_by_label = Counter()
    for file_info in tqdm(data['files'], desc="Merging AI Mechanic data"):
        if file_info.get("category") != "training":
            skipped["non_training_category"] += 1
            continue

        src_rel_path = file_info['path']
        label = file_info['label']['label']
        src_path = os.path.join(export_dir, src_rel_path)
        
        if not os.path.exists(src_path):
            skipped["missing_source_file"] += 1
            continue
            
        target_key = resolve_aim_label(label)
        if target_key not in SUPPORTED_SOURCE_LABELS:
            skipped[normalize_text(label) or "unmapped"] += 1
            continue

        target_subfolder = TARGET_MAP[target_key]
        if target_subfolder:
            target_dir = os.path.join(PROJECT_DATA_DIR, "train", target_subfolder)
            os.makedirs(target_dir, exist_ok=True)
            
            filename = f"aim__{sanitize_filename(src_path)}"
            target_path = os.path.join(target_dir, filename)
            
            if not os.path.exists(target_path):
                shutil.copy2(src_path, target_path)
                count += 1
                copied_by_label[target_key] += 1
                
    print(f"Added {count} samples from AI Mechanic export.")
    print(f"Skipped AI Mechanic files: {dict(skipped)}")
    return copied_by_label, skipped

def merge_engine_models():
    print("Processing engine model variety (unprocessedData/data)...")
    src_data_dir = os.path.join(UNPROCESSED_DATA_DIR, "data")
    if not os.path.exists(src_data_dir):
        return

    target_dir = os.path.join(PROJECT_DATA_DIR, "train", TARGET_MAP["normal_engine_idle"])
    os.makedirs(target_dir, exist_ok=True)
    
    count = 0
    max_per_type = 10
    skipped = Counter()
    copied_by_engine = Counter()
    
    for engine_type in os.listdir(src_data_dir):
        engine_path = os.path.join(src_data_dir, engine_type)
        if not os.path.isdir(engine_path):
            continue
            
        idle_files = []
        for root, dirs, files in os.walk(engine_path):
            for file in files:
                if "_idle_" in file.lower() and file.endswith(".wav"):
                    idle_files.append(os.path.join(root, file))
                elif file.endswith(".wav"):
                    skipped["non_idle_files"] += 1
        
        idle_files = sorted(idle_files)
        if len(idle_files) > max_per_type:
            skipped["capped_idle_files"] += len(idle_files) - max_per_type
        for f in idle_files[:max_per_type]:
            filename = f"engine_model_{sanitize_filename(engine_type + '_' + os.path.basename(f))}"
            target_path = os.path.join(target_dir, filename)
            if not os.path.exists(target_path):
                shutil.copy2(f, target_path)
                count += 1
                copied_by_engine[engine_type] += 1
                
    print(f"Added {count} diversified normal idle samples from engine models.")
    print(f"Engine-model source coverage: {dict(copied_by_engine)}")
    print(f"Skipped engine-model files: {dict(skipped)}")
    return copied_by_engine, skipped


def build_merge_summary(aim_copied, aim_skipped, engine_copied, engine_skipped):
    summary = {
        "ai_mechanic": {
            "copied": dict(aim_copied),
            "skipped": dict(aim_skipped),
        },
        "engine_models": {
            "copied": dict(engine_copied),
            "skipped": dict(engine_skipped),
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(MERGE_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Merge summary written to {MERGE_SUMMARY_PATH}")

if __name__ == "__main__":
    aim_copied, aim_skipped = merge_ai_mechanic()
    engine_copied, engine_skipped = merge_engine_models()
    build_merge_summary(aim_copied, aim_skipped, engine_copied, engine_skipped)
    print("Data merging complete.")
