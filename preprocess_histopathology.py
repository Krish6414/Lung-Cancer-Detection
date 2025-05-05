import os
import shutil
import random
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

# Paths
RAW_DIR = Path("data/raw/histopathology/lung_image_sets")
PROCESSED_DIR = Path("data/processed/histopathology")
IMG_SIZE = (224, 224)

# Class label mapping
CLASS_MAP = {
    "lung_n": "normal",
    "lung_aca": "adenocarcinoma",
    "lung_scc": "squamous_cell_carcinoma"
}

# Create processed directories
for split in ['train', 'val', 'test']:
    for class_name in CLASS_MAP.values():
        path = PROCESSED_DIR / split / class_name
        path.mkdir(parents=True, exist_ok=True)

def process_and_save(image_path, save_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(save_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    image_paths = []
    labels = []

    # Collect images and labels
    for folder in CLASS_MAP:
        folder_path = RAW_DIR / folder
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append(folder_path / img_file)
                labels.append(CLASS_MAP[folder])

    # Split into train/val/test
    train_imgs, test_imgs, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        train_imgs, train_labels, test_size=0.1, stratify=train_labels, random_state=42)

    print(f"Total: {len(image_paths)} | Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    # Save function
    def save_images(images, labels, split):
        for img_path, label in zip(images, labels):
            filename = img_path.name
            save_path = PROCESSED_DIR / split / label / filename
            process_and_save(img_path, save_path)

    # Save all sets
    save_images(train_imgs, train_labels, "train")
    save_images(val_imgs, val_labels, "val")
    save_images(test_imgs, test_labels, "test")

if __name__ == "__main__":
    main()
