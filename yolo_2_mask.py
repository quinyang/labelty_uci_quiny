import cv2
import numpy as np
import os
import glob

image_dir = "/home/quinyang/labelty_uci_quiny/resource/images/"
label_dir = "/home/quinyang/labelty_uci_quiny/resource/labels/"
output_dir = "/home/quinyang/labelty_uci_quiny/resource/labels_mask/"

os.makedirs(output_dir, exist_ok=True)

def convert_yolo_to_mask(image_path, txt_path, output_path):
    # 1. Read the original image just to get its exact Width and Height
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    height, width = img.shape[:2]

    print(height, width)    
    
    # 2. Create a blank black digital canvas (all zeros = background)
    # Using uint8 because your class IDs (1-11) easily fit in 8-bit integers
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 3. Read the YOLO .txt file
    if not os.path.exists(txt_path):
        print(f"Label file missing for: {txt_path}")
        return
        
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        data = line.strip().split()
        if len(data) < 3:
            continue # Skip empty or invalid lines
            
        class_id = int(data[0])
        coords = list(map(float, data[1:]))
        
        # 4. Denormalize coordinates (multiply X by width, Y by height)
        # and reshape into pairs of [x, y]
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i+1] * height)
            points.append([x, y])
            
        # Convert to numpy array format required by OpenCV
        polygon = np.array([points], dtype=np.int32)
        
        # 5. Draw and fill the polygon on the mask with the class_id value
        cv2.fillPoly(mask, polygon, color=(class_id))
        
    # 6. Save the resulting mask as a PNG image
    cv2.imwrite(output_path, mask)


# --- EXECUTION LOOP ---
# Find all images and process their corresponding text files
image_files = glob.glob(os.path.join(image_dir, "*.*")) # Grabs jpg, png, etc.

for img_path in image_files:
    # 1. Get the filename with the extension (e.g., '00000001.jpg')
    filename_with_ext = os.path.basename(img_path)
    
    # 2. Split the name from the extension and explicitly grab the first part
    base_name = os.path.splitext(filename_with_ext)[0]
    
    # 3. Print it to the terminal so we can prove it's a string!
    print(f"Attempting to process: {base_name}")
    
    # 4. Create the paths (Check your variable names here! Use label_dir if that's what you defined at the top)
    txt_path = os.path.join(label_dir, base_name + ".txt")
    output_path = os.path.join(output_dir, base_name + ".png")
    
    convert_yolo_to_mask(img_path, txt_path, output_path)

print("Conversion complete! Your pixel masks are in the output directory.")