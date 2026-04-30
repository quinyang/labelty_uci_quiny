import cv2
import matplotlib.pyplot as plt
import numpy as np

# Set the base filename
# files to test with: 70 259 515 629 754 855 921
filename = "IO000001_(259)"

# We found earlier that your raw test files are actually .png!
image_path = f"/home/jiakuny1/Projects/nnUNet_data/nnUNet_raw/Dataset101_Dental/imagesTs/{filename}_0000.png"
mask_path = f"/home/jiakuny1/Projects/swin_predictions/{filename}.png"
#mask_path = f"/home/jiakuny1/Projects/nnUNet_data/model_predictions/{filename}.png"

# Load the original X-ray and the model's prediction
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Safety check
if image is None:
    print(f"CRITICAL ERROR: Could not find raw image at:\n{image_path}")
    exit()
if mask is None:
    print(f"CRITICAL ERROR: Could not find mask at:\n{mask_path}")
    exit()

# Create a side-by-side plot (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Left Graph: Original Image (Must use axes)
axes[0].imshow(image, cmap='gray')
axes[0].set_title(f"Original X-Ray: {filename}")
axes[0].axis('off')

# Right Graph: Overlay (Must use axes)
axes[1].imshow(image, cmap='gray')
# alpha=0.5 makes the mask semi-transparent, jet gives it bright colors
axes[1].imshow(mask, cmap='jet', alpha=0.5) 
axes[1].set_title("nnU-Net Prediction Overlay")
axes[1].axis('off')

# Save the visualization to your project folder
plt.tight_layout()
plt.savefig(f"/home/jiakuny1/Projects/overlay_{filename}.png")
print(f"Success! Saved visualization to overlay_{filename}.png!")