import os
import shutil
import random
import glob
import json
from PIL import Image

# --- CONFIGURATION ---
# Paths to your current data
source_images_dir = "./resource/images/"
source_masks_dir = "./resource/labels_mask/"

# Path to your new nnU-Net raw directory (using the variables we just set)
nnunet_raw_dir = "./nnUNet_data/nnUNet_raw/"
dataset_name = "Dataset101_Dental"  # nnU-Net requires 'DatasetXXX_Name'
dataset_path = os.path.join(nnunet_raw_dir, dataset_name)

# Create the strict folder structure
imagesTr = os.path.join(dataset_path, "imagesTr")
labelsTr = os.path.join(dataset_path, "labelsTr")
imagesTs = os.path.join(dataset_path, "imagesTs")

os.makedirs(imagesTr, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)
os.makedirs(imagesTs, exist_ok=True)

# --- SPLITTING LOGIC ---
split_ratio = 0.80 # 80% Training, 20% Testing

# Get all base filenames without extensions
# Get all base filenames without extensions, ignoring those without masks
all_images = glob.glob(os.path.join(source_images_dir, "*.*"))
base_names = []
for f in all_images:
    name = os.path.splitext(os.path.basename(f))[0]
    if os.path.exists(os.path.join(source_masks_dir, name + ".png")):
        base_names.append(name)

# Randomly shuffle the list for a fair split
random.seed(42) # Keeps the split reproducible
random.shuffle(base_names)

split_index = int(len(base_names) * split_ratio)
train_names = base_names[:split_index]
test_names = base_names[split_index:]

print(f"Total files: {len(base_names)}")
print(f"Allocating {len(train_names)} to Training and {len(test_names)} to Testing...")

def copy_and_format(name_list, is_test=False):
    for name in name_list:
        img_src = os.path.join(source_images_dir, name + ".jpeg")
        if not os.path.exists(img_src):
            continue
        
        mask_src = os.path.join(source_masks_dir, name + ".png")

        # nnU-Net REQUIRES images to end in _0000 (channel identifier)
        img_dest_name = f"{name}_0000.png"
        
        # Load the image, convert to Grayscale (1 modality), and save as PNG
        img = Image.open(img_src).convert("L")
        
        if not is_test:
            # Copy to Train
            img.save(os.path.join(imagesTr, img_dest_name), format="PNG")
            shutil.copy(mask_src, os.path.join(labelsTr, name + ".png"))
        else:
            # Copy to Test (no labels needed in test set folder)
            img.save(os.path.join(imagesTs, img_dest_name), format="PNG")

copy_and_format(train_names, is_test=False)
copy_and_format(test_names, is_test=True)

# --- GENERATE DATASET.JSON ---
# IMPORTANT: These integer keys MUST match the class_id numbers used in your YOLO files!
labels_dict = {
    "background": 0,
    "Apical Lesion": 1,
    "Main Root": 2,
    "Main Canal": 3,
    "Mesial Root": 4,
    "Mesial Canal": 5,
    "Distal Root": 6,
    "Distal Canal": 7,
    "Palatal Root": 8,
    "Palatal Canal": 9,
    "Root Canal Filling": 10,
    "decay": 11
}

dataset_json = {
    "channel_names": {
        "0": "X-ray" # 0 maps to the _0000 suffix
    },
    "labels": labels_dict,
    "numTraining": len(train_names),
    "file_ending": ".png"
}

with open(os.path.join(dataset_path, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

print(f"Success! Dataset formatted at {dataset_path}")