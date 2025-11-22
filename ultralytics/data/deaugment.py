import torch
import matplotlib.pyplot as plt
import cv2  # For drawing bounding boxes on 
import numpy as np

def undo_flips(image, bboxes, horizontal_flip=False, vertical_flip=False):
    """
    Undoes the horizontal and vertical flips on an image and bounding boxes.
    
    Args:
        image (numpy.ndarray): The flipped image.
        bboxes (numpy.ndarray): Bounding boxes in the format [x1, y1, x2, y2].
        horizontal_flip (bool): Whether a horizontal flip was applied.
        vertical_flip (bool): Whether a vertical flip was applied.
    
    Returns:
        tuple: The restored image and bounding boxes.
    """
    _, H, W = image.shape  # Get batch size, height, and width

    # Undo horizontal flip
    if horizontal_flip:
        image = torch.flip(image, dims=[-1])  # Flip width dimension
        # bboxes[:, :, [0, 2]] = W - bboxes[:, :, [2, 0]]  # Flip x1 and x2 for all bounding boxes in batch
        # bboxes[:, [0, 2]] = W - bboxes[:, [2, 0]]
        if bboxes.numel() > 0:  # Check if bboxes is not empty
            bboxes[:, [0, 2]] = W - bboxes[:, [2, 0]]
    
    # Undo vertical flip
    if vertical_flip:
        image = torch.flip(image, dims=[-2])  # Flip height dimension
        bboxes[:, :, [1, 3]] = H - bboxes[:, :, [3, 1]]  # Flip y1 and y2 for all bounding boxes in batch

    return image, bboxes



def draw_bboxes(image, bboxes, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on an image.
    
    Args:
        image (np.ndarray): The image to draw on (HWC format).
        bboxes (torch.Tensor): Bounding boxes in the format [x1, y1, x2, y2].
        color (tuple): RGB color for the bounding boxes.
        thickness (int): Thickness of the bounding box lines.
    
    Returns:
        np.ndarray: The image with bounding boxes drawn on it.
    """
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return image

def draw_transformed_coords(img, transformed_coords):
    for coords in transformed_coords:
        points = np.int32(coords).reshape((-1, 1, 2))
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    return img


def draw_pred_bboxes(image, bboxes, color=(255, 0, 0), thickness=2, font_scale=0.5, font_thickness=1):
    """
    Draw bounding boxes with class and confidence on an image.
    
    Args:
        image (np.ndarray): The image to draw on (HWC format).
        bboxes (torch.Tensor or np.ndarray): Bounding boxes in the format [x1, y1, x2, y2, confidence, class].
        color (tuple): RGB color for the bounding boxes.
        thickness (int): Thickness of the bounding box lines.
        font_scale (float): Font scale for the text.
        font_thickness (int): Thickness of the text font.
    
    Returns:
        np.ndarray: The image with bounding boxes and labels drawn on it.
    """
    for bbox in bboxes:
        x1, y1, x2, y2, confidence, class_label = bbox[:6]
        # Draw rectangle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Create label text with class and confidence
        label = f"Class: {int(class_label)}, Conf: {confidence:.2f}"
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_origin = (int(x1), int(y1) - text_height - 4)  # Position above the box
        
        # Draw text background for better visibility
        cv2.rectangle(image, (text_origin[0], text_origin[1] - text_height - 2), 
                      (text_origin[0] + text_width, text_origin[1] + baseline), color, -1)
        
        # Put text on the image
        cv2.putText(image, label, (text_origin[0], text_origin[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    return image


def undo_affine(image, M_inv, ori_shape=(1280,1280)):
    """
    Undo affine transformations using the inverse matrix and original shape.
    Args:
        image (np.ndarray): The transformed image (after random perspective).
        M_inv (np.ndarray): Inverse transformation matrix.
        ori_shape (tuple): Original shape to map back to (height, width).
    Returns:
        np.ndarray: Image mapped back to the original mosaic space.
    """
    original_img_back = cv2.warpAffine(image, M_inv[:2], dsize=ori_shape, borderValue=(114, 114, 114))

    return original_img_back


def apply_bboxes(bboxes, M):
    """
    Apply affine transformation to bounding boxes.

    This function applies an affine transformation to a set of bounding boxes using the provided
    transformation matrix.

    Args:
        bboxes (np.ndarray): Bounding boxes in xyxy format with shape (N, 4), where N is the number
            of bounding boxes.
        M (np.ndarray): Affine transformation matrix with shape (3, 3).

    Returns:
        np.ndarray: Transformed bounding boxes in xyxy format with shape (N, 4).
    """
    n = len(bboxes)
    if n == 0:
        return bboxes

    xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
    xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ M.T  # transform
    xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # perspective rescale or affine

    # Create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

def reverse_bboxes(bboxes, M_inv):
    """
    Reverse the transformation of bounding boxes using the inverse matrix.
    Args:
        labels (dict): Dictionary containing transformed bounding boxes and other label information.
        M_inv (np.ndarray): Inverse transformation matrix.
    Returns:
        np.ndarray: Original bounding boxes mapped to the original space.
    """
    original_bboxes = apply_bboxes(bboxes, M_inv)
    return original_bboxes


def calculate_overlap(bbox, mosaic_coords):
    """Calculate the overlap area between a bounding box and mosaic coordinates."""
    x_min, y_min, x_max, y_max = bbox
    mosaic_x_min, mosaic_y_min, mosaic_x_max, mosaic_y_max = mosaic_coords

    # Calculate the intersection rectangle
    inter_x_min = max(x_min, mosaic_x_min)
    inter_y_min = max(y_min, mosaic_y_min)
    inter_x_max = min(x_max, mosaic_x_max)
    inter_y_max = min(y_max, mosaic_y_max)

    # Check if there is an overlap
    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)  # Area of intersection
    return 0


def draw_bboxes_mosaic(image, bboxes, color=(255, 0, 0), thickness=2):
    """
    Draw bounding boxes on an image.
    
    Args:
        image (np.ndarray): The image to draw on (HWC format).
        bboxes (np.ndarray or list): Bounding boxes in the format [x1, y1, x2, y2].
        color (tuple): RGB color for the bounding boxes.
        thickness (int): Thickness of the bounding box lines.
    
    Returns:
        np.ndarray: The image with bounding boxes drawn on it.
    """
    # Ensure bboxes is a NumPy array
    bboxes = np.array(bboxes, dtype=np.float32)
    
    # Reshape if a single bounding box is provided (shape (4,))
    if bboxes.ndim == 1 and bboxes.shape[0] == 4:
        bboxes = np.expand_dims(bboxes, axis=0)  # Convert to shape (1, 4)
    
    # Loop through each bounding box and draw it
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return image


# def map_bbox_to_patch(ori_bbox, mosaic_coords, source_coords):
#     """
#     Maps the bounding box coordinates from the mosaic image back to the patch in the original image.
#     Ensures proper clipping to patch dimensions.

#     Args:
#         ori_bbox (list): [x1, y1, x2, y2] bounding box coordinates in the mosaic image.
#         mosaic_coords (list): [x1, y1, x2, y2] coordinates of the patch in the mosaic image.
#         source_coords (list): [x1, y1, x2, y2] coordinates of the patch in the original image.

#     Returns:
#         list: Adjusted bounding box coordinates [x1, y1, x2, y2] in the patch.
#     """
#     # Extract mosaic patch boundaries and original patch boundaries
#     mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = mosaic_coords
#     source_x1, source_y1, source_x2, source_y2 = source_coords

#     # Step 1: Adjust ori_bbox from mosaic image coordinates to patch coordinates within the mosaic image
#     patch_bbox = [
#         ori_bbox[0] - mosaic_x1,  # Shift x1 based on the mosaic coordinates
#         ori_bbox[1] - mosaic_y1,  # Shift y1 based on the mosaic coordinates
#         ori_bbox[2] - mosaic_x1,  # Shift x2 based on the mosaic coordinates
#         ori_bbox[3] - mosaic_y1,  # Shift y2 based on the mosaic coordinates
#     ]
    
#     # Step 2: Ensure the bounding box stays within the source patch's size
#     patch_width = source_x2 - source_x1
#     patch_height = source_y2 - source_y1

#     # Clip the bounding box coordinates to stay inside the patch boundaries
#     clipped_bbox = [
#         max(0, min(patch_width, patch_bbox[0])),  # Clip x1
#         max(0, min(patch_height, patch_bbox[1])),  # Clip y1
#         max(0, min(patch_width, patch_bbox[2])),  # Clip x2
#         max(0, min(patch_height, patch_bbox[3])),  # Clip y2
#     ]

#     # If the bounding box has become invalid (e.g., x1 >= x2 or y1 >= y2), return a null box
#     if clipped_bbox[0] >= clipped_bbox[2] or clipped_bbox[1] >= clipped_bbox[3]:
#         return [0, 0, 0, 0]  # No valid bounding box

#     # Step 3: Adjust the bounding box to fit within the original image's coordinates
#     # Calculate the scaling factors between mosaic patch and source patch
#     mosaic_patch_width = mosaic_x2 - mosaic_x1
#     mosaic_patch_height = mosaic_y2 - mosaic_y1
#     scale_x = (source_x2 - source_x1) / mosaic_patch_width
#     scale_y = (source_y2 - source_y1) / mosaic_patch_height

#     # Adjust the coordinates of the bounding box back to the original image space
#     adjusted_bbox = [
#         source_x1 + scale_x * clipped_bbox[0],  # Adjust x1 based on scaling
#         source_y1 + scale_y * clipped_bbox[1],  # Adjust y1 based on scaling
#         source_x1 + scale_x * clipped_bbox[2],  # Adjust x2 based on scaling
#         source_y1 + scale_y * clipped_bbox[3],  # Adjust y2 based on scaling
#     ]

#     return adjusted_bbox


# def map_bbox_to_patch(ori_bbox, mosaic_coords, source_coords):
#     """
#     Maps the bounding box coordinates from the mosaic image to the corresponding patch
#     in the original image without scaling, just by placing it directly.
    
#     Args:
#         ori_bbox (list): [x1, y1, x2, y2] bounding box coordinates in the mosaic image.
#         mosaic_coords (list): [x1, y1, x2, y2] coordinates of the patch in the mosaic image.
#         source_coords (list): [x1, y1, x2, y2] coordinates of the patch in the original image.
        
#     Returns:
#         list: Directly mapped bounding box coordinates in the original patch.
#     """
#     # Step 1: Extract coordinates of the mosaic and the original patch
#     mosaic_x1, mosaic_y1, mosaic_x2, mosaic_y2 = mosaic_coords
#     source_x1, source_y1, source_x2, source_y2 = source_coords

#     # Step 2: Shift the original bounding box coordinates to the patch's coordinates in the original image
#     mapped_bbox = [
#         ori_bbox[0] - mosaic_x1 + source_x1,  # x1 shifted from mosaic to patch in original
#         ori_bbox[1] - mosaic_y1 + source_y1,  # y1 shifted from mosaic to patch in original
#         ori_bbox[2] - mosaic_x1 + source_x1,  # x2 shifted from mosaic to patch in original
#         ori_bbox[3] - mosaic_y1 + source_y1   # y2 shifted from mosaic to patch in original
#     ]
    
#     return mapped_bbox


# def transform_mosaic_coords(mosaic_metadata, M, final_image_size):
#     transformed_coords = []
#     for meta in mosaic_metadata:
#         x1, y1, x2, y2 = meta['mosaic_coords']
#         coords = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32).T
#         transformed = M @ coords  # Apply the affine transformation
#         transformed = transformed[:2] / transformed[2]  # Normalize if perspective
#         transformed = transformed.T  # Reshape back to (4, 2)
#         transformed_coords.append(transformed)
#     return transformed_coords

def polygon_to_bbox(polygon):
    """
    Convert a polygon (list of 4 points) to a bounding box [x1, y1, x2, y2].
    
    Args:
        polygon (numpy.ndarray): An array of shape (4, 2) representing [x, y] points.

    Returns:
        list: Bounding box in [x1, y1, x2, y2] format.
    """
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)
    return [x1, y1, x2, y2]

def adjust_for_horizontal_flip(mosaic_coords, image_width=640):
    """
    Adjust mosaic coordinates after a horizontal flip.

    Args:
        mosaic_coords (list): A list of bounding box coordinates in [x1, y1, x2, y2] format.
        image_width (int): The width of the image after applying the flip.

    Returns:
        list: Adjusted bounding box coordinates after the horizontal flip.
    """
    adjusted_coords = []
    for coord in mosaic_coords:
        x1, y1, x2, y2 = coord
        # Flip x-coordinates
        flipped_x1 = image_width - x2
        flipped_x2 = image_width - x1
        adjusted_coords.append([flipped_x1, y1, flipped_x2, y2])
    return adjusted_coords



def transform_mosaic_coords(mosaic_metadata, M, final_image_size):
    transformed_coords = []
    for meta in mosaic_metadata:
        x1, y1, x2, y2 = meta['mosaic_coords']
        # Define the four corners of the box
        coords = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32).T
        # Apply the affine transformation
        transformed = M @ coords  # Shape will be (3, 4) because of the homogenous coordinates
        transformed = transformed[:2] / transformed[2]  # Normalize by the third row
        transformed = transformed.T  # Shape back to (4, 2)
        # Calculate the bounding box from the transformed coordinates
        x1 = min(transformed[:, 0])
        y1 = min(transformed[:, 1])
        x2 = max(transformed[:, 0])
        y2 = max(transformed[:, 1])
        transformed_coords.append([x1, y1, x2, y2])
    return transformed_coords


def draw_pred_bboxes1(image, bboxes, color=(255, 0, 0), thickness=2, font_scale=0.5, font_thickness=1):
    """
    Draw bounding boxes with class and confidence on an image.
    
    Args:
        image (np.ndarray): The image to draw on (HWC format).
        bboxes (torch.Tensor or np.ndarray): Bounding boxes in the format [x1, y1, x2, y2, confidence, class].
        color (tuple): RGB color for the bounding boxes.
        thickness (int): Thickness of the bounding box lines.
        font_scale (float): Font scale for the text.
        font_thickness (int): Thickness of the text font.
    
    Returns:
        np.ndarray: The image with bounding boxes and labels drawn on it.
    """
    
    for bbox in bboxes:
        print("bbox inside: ", bbox)
        # Ensure bbox is converted to a NumPy array if it's a tensor
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.detach().cpu().numpy()
        
        # Ensure bbox is a 1D array or list
        if isinstance(bbox, np.ndarray):
            bbox = bbox.flatten().tolist()  # Convert to a flat list
        
        # Extract the values, ensuring they are scalars
        x1, y1, x2, y2, confidence, class_label = [float(val) if isinstance(val, (int, float)) else float(val[0]) for val in bbox[:6]]
        
        # Draw rectangle
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        
        # Create label text with class and confidence
        label = f"Class: {int(class_label)}, Conf: {confidence:.2f}"
        
        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_origin = (int(x1), max(0, int(y1) - text_height - 4))  # Ensure text is not out of bounds
        
        # Draw text background for better visibility
        cv2.rectangle(image, (text_origin[0], text_origin[1] - text_height - 2), 
                      (text_origin[0] + text_width, text_origin[1] + baseline), color, -1)
        
        # Put text on the image
        cv2.putText(image, label, (text_origin[0], text_origin[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    return image

def is_patch_majority_black(img_patch, threshold=50):
    """
    Check if more than 'threshold' percent of pixels in the patch are background (black).
    
    Args:
    - img_patch (numpy array): Extracted image patch.
    - threshold (float): Percentage threshold to determine if background dominates.
    
    Returns:
    - (bool, float): True if background > threshold%, else False. Also returns the percentage.
    """
    if img_patch.size == 0:
        return True, 100.0  # If the patch is empty, consider it fully background
    
    # Convert to grayscale if needed
    if len(img_patch.shape) == 3:
        img_patch_gray = cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)
    else:
        img_patch_gray = img_patch  # Already grayscale

    # Define black pixel threshold (e.g., intensity <= 10)
    black_pixels = np.sum(img_patch_gray <= 10)
    total_pixels = img_patch_gray.size

    # Compute background percentage
    black_ratio = (black_pixels / total_pixels) * 100

    return black_ratio > threshold, black_ratio