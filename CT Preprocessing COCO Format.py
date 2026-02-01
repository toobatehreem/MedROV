import numpy as np
import os
import json
from PIL import Image
import nibabel as nib

# Define paths
image_folder = '/l/users/tooba.sheikh/Datasets/RawData_AbdomenBTCV/RawData/test/images'
label_folder = '/l/users/tooba.sheikh/Datasets/RawData_AbdomenBTCV/RawData/test/labels'
output_image_folder = '/l/users/tooba.sheikh/Datasets/BioMedParse/BTCV/images/'
output_json_file = '/l/users/tooba.sheikh/Datasets/BioMedParse/BTCV/annotations.json'

# Create output directory if it doesn't exist
os.makedirs(output_image_folder, exist_ok=True)

# Mapping from label values to class names
label_mapping = {
    0: "background",
    1: "spleen",
    2: "right kidney",
    3: "left kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior vena cava",
    10: "portal vein and splenic vein",
    11: "pancreas",
    12: "right adrenal gland",
    13: "left adrenal gland"
}

# Initialize COCO annotation structure
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": [{"id": idx, "name": name} for idx, name in label_mapping.items()]
}

annotation_id = 1  # Starting annotation ID
image_id = 1  # Starting image ID

# Get all image and label files
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.nii.gz')]) 
label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.nii.gz')]) 

# Iterate over image files
for image_file in image_files:
    # Find the corresponding label file
    label_file = image_file.replace('img', 'label') 
    label_path = os.path.join(label_folder, label_file)

    # If label file doesn't exist, skip this image
    if not os.path.exists(label_path):
        print(f"Label for {image_file} not found, skipping.")
        continue

    # Load the label mask file
    label_nii = nib.load(label_path)
    label_data = label_nii.get_fdata()

    # Load the original image file
    image_path = os.path.join(image_folder, image_file)
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()

    for slice_idx in range(0, label_data.shape[2]):  # Iterating over slices
        slice_data = label_data[:, :, slice_idx]  # Extract 2D slice from the label
        original_image_slice = image_data[:, :, slice_idx]  # Extract corresponding 2D slice from the original image
        unique_labels = np.unique(slice_data)  # Get unique labels in the slice

        # Skip slices that only contain background
        if len(unique_labels) == 1 and unique_labels[0] == 0:
            continue

        # Save the original slice as a .png image
        image_filename = f'btcv_{image_id:08d}.png'
        output_image_path = os.path.join(output_image_folder, image_filename)

        # Normalize the original image slice and convert to uint8 for saving as an image
        # original_image_slice = np.clip(original_image_slice, -500, 1000)
        original_image_slice = np.clip(original_image_slice, -150, 250)
        normalized_image = ((original_image_slice - np.min(original_image_slice)) / 
                            (np.max(original_image_slice) - np.min(original_image_slice)) * 255).astype(np.uint8)
        image = Image.fromarray(normalized_image)
        image.save(output_image_path)

        # Add image metadata to COCO annotations
        coco_annotations["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": slice_data.shape[1],
            "height": slice_data.shape[0]
        })

        # Iterate over each unique label and find its bounding box
        for label in unique_labels:
            if label == 0:  # Skip background
                continue

            # Get coordinates of all pixels belonging to the current label
            label_mask = (slice_data == label)
            coords = np.argwhere(label_mask)  # Find the coordinates of non-zero points

            if coords.size > 0:
                # Find bounding box (min_x, min_y, max_x, max_y)
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                
                # Create annotation for the current label and slice
                bbox = [int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y)]  # COCO format: [x, y, width, height]
                coco_annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],  # Area is width * height
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1  # Increment image ID after processing each slice

# Save the COCO annotations to a .json file
with open(output_json_file, 'w') as f:
    json.dump(coco_annotations, f, indent=4)

print(f"COCO annotations saved to {output_json_file}")
