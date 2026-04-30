import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import glob

# Set your directories
pred_dir = "/home/jiakuny1/Projects/nnUNet_data/model_predictions"
gt_dir = "/home/jiakuny1/Projects/resource/labels_mask"

all_y_true = []
all_y_pred = []

# Find all the prediction PNGs we generated earlier
pred_files = glob.glob(os.path.join(pred_dir, "*.png"))

print(f"Found {len(pred_files)} predictions. Loading images...")

for pred_path in pred_files:
    filename = os.path.basename(pred_path)
    gt_path = os.path.join(gt_dir, filename)
    
    if not os.path.exists(gt_path):
        print(f"  -> Skipping {filename}: No matching ground truth label found in labelsTs.")
        continue

    # Load images as grayscale (where pixel values = class IDs)
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

    if pred_mask is None or gt_mask is None:
        continue

    # Flatten the 2D images into 1D arrays and store them
    all_y_pred.append(pred_mask.flatten())
    all_y_true.append(gt_mask.flatten())

if not all_y_true:
    print("\nCRITICAL ERROR: No matching files found to compare!")
    exit()

print("Concatenating millions of pixels (this might take a few seconds)...")
y_true = np.concatenate(all_y_true)
y_pred = np.concatenate(all_y_pred)

print("Calculating the confusion matrix...")
classes = np.unique(np.concatenate((y_true, y_pred)))

# We normalize by 'true' so the rows show percentages instead of raw pixel counts.
# Otherwise, the background (Class 0) would have 50 million pixels and break the color scale!
cm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')

print("Drawing the heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted Class (What the Model Guessed)')
plt.ylabel('True Class (Ground Truth)')
plt.title('Pixel-wise Confusion Matrix (Normalized by True Class)')
plt.tight_layout()

# Save the output
out_path = "/home/jiakuny1/Projects/confusion_matrix_nnunet.png"
plt.savefig(out_path, dpi=300)
print(f"\nSuccess! Saved to {out_path}")