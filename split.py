import os
import shutil
import random

# Define paths
DATASET_PATH = "dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")

# Create train/val directories if not exist
for category in ["deceptive", "truthful"]:
    os.makedirs(os.path.join(TRAIN_PATH, category), exist_ok=True)
    os.makedirs(os.path.join(VAL_PATH, category), exist_ok=True)

# Function to split data
def split_data(category, split_ratio=0.8):
    src_folder = os.path.join(DATASET_PATH, category)
    images = os.listdir(src_folder)
    random.shuffle(images)
    
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for img in train_images:
        shutil.move(os.path.join(src_folder, img), os.path.join(TRAIN_PATH, category, img))

    for img in val_images:
        shutil.move(os.path.join(src_folder, img), os.path.join(VAL_PATH, category, img))

# Apply the split
split_data("deceptive")
split_data("truthful")

print("✅ Dataset successfully split into 'train' and 'val'!")