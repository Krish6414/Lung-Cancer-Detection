import os
from pathlib import Path
from PIL import Image
import shutil

# Define paths
RAW_BASE = Path("data/raw/ct_scans/LungcancerDataSet/Data")
PROCESSED_BASE = Path("data/processed/ct_scans")
IMG_SIZE = (224, 224)

# Class name normalization mapping
CLASS_NAME_MAP = {
    "adenocarcinoma": "adenocarcinoma",
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": "adenocarcinoma",
    "BenginCases": "benign",
    "Bengin cases": "benign",
    "large.cell.carcinoma": "large_cell_carcinoma",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": "large_cell_carcinoma",
    "MalignantCases": "malignant",
    "Malignant cases": "malignant",
    "normal": "normal",
    "squamous.cell.carcinoma": "squamous_cell_carcinoma",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": "squamous_cell_carcinoma",
}

def process_and_save(image_path, save_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(save_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {image_path.name} – {e}")

def preprocess_split(split):
    split_dir = RAW_BASE / split
    for class_folder in os.listdir(split_dir):
        full_class_path = split_dir / class_folder
        if not full_class_path.is_dir():
            continue

        label = CLASS_NAME_MAP.get(class_folder, None)
        if label is None:
            print(f"[SKIP] Unknown folder: {class_folder}")
            continue

        for file in os.listdir(full_class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = full_class_path / file
                dst_path = PROCESSED_BASE / split / label / file
                process_and_save(src_path, dst_path)

def main():
    for split in ['train', 'valid', 'test']:
        print(f"[INFO] Processing split: {split}")
        preprocess_split(split)

if __name__ == "__main__":
    main()
