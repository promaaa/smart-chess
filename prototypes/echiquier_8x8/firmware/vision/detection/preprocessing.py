"""
Image preprocessing and data augmentation for chessboard detection.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import random


def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess a single frame for CNN inference.
    
    Args:
        frame: Input BGR image from camera (H, W, 3)
        target_size: Target dimensions (height, width)
        normalize: Whether to normalize pixel values to [0, 1]
        
    Returns:
        Preprocessed image ready for model input (3, H, W) in RGB
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to target dimensions
    resized = cv2.resize(rgb, (target_size[1], target_size[0]))
    
    # Normalize to [0, 1]
    if normalize:
        processed = resized.astype(np.float32) / 255.0
    else:
        processed = resized.astype(np.float32)
    
    # Transpose to channel-first format (H, W, C) -> (C, H, W)
    processed = np.transpose(processed, (2, 0, 1))
    
    return processed


def preprocess_batch(
    frames: List[np.ndarray],
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess a batch of frames.
    
    Args:
        frames: List of BGR images
        target_size: Target dimensions
        
    Returns:
        Batch array of shape (N, 3, H, W)
    """
    processed = [preprocess_frame(f, target_size) for f in frames]
    return np.stack(processed, axis=0)


def denormalize_corners(
    corners: np.ndarray,
    original_size: Tuple[int, int]
) -> np.ndarray:
    """
    Convert normalized corner coordinates [0,1] back to pixel coordinates.
    
    Args:
        corners: Normalized corners of shape (4, 2) in [0, 1] range
        original_size: Original image dimensions (height, width)
        
    Returns:
        Pixel coordinates of shape (4, 2)
    """
    h, w = original_size
    pixel_corners = corners.copy()
    pixel_corners[:, 0] *= w  # x coordinates
    pixel_corners[:, 1] *= h  # y coordinates
    return pixel_corners


def normalize_corners(
    corners: np.ndarray,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """
    Convert pixel corner coordinates to normalized [0,1] range.
    
    Args:
        corners: Pixel coordinates of shape (4, 2)
        image_size: Image dimensions (height, width)
        
    Returns:
        Normalized corners in [0, 1] range
    """
    h, w = image_size
    normalized = corners.copy().astype(np.float32)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    return normalized


# ============================================================================
# DATA AUGMENTATION FOR TRAINING
# ============================================================================

def augment_image(
    image: np.ndarray,
    corners: np.ndarray,
    augment_config: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random augmentations to image and adjust corner coordinates.
    
    Args:
        image: Input BGR image
        corners: Corner coordinates of shape (4, 2) in pixel space
        augment_config: Optional configuration dict for augmentation parameters
        
    Returns:
        Tuple of (augmented_image, adjusted_corners)
    """
    if augment_config is None:
        augment_config = {
            'rotation_range': 15,
            'brightness_range': 0.3,
            'contrast_range': 0.3,
            'perspective_strength': 0.1,
            'flip_horizontal': True
        }
    
    aug_image = image.copy()
    aug_corners = corners.copy()
    
    # Random brightness adjustment
    if random.random() > 0.5:
        delta = random.uniform(-augment_config['brightness_range'], 
                               augment_config['brightness_range'])
        aug_image = np.clip(aug_image.astype(np.float32) + delta * 255, 0, 255).astype(np.uint8)
    
    # Random contrast adjustment
    if random.random() > 0.5:
        factor = 1 + random.uniform(-augment_config['contrast_range'],
                                    augment_config['contrast_range'])
        mean = np.mean(aug_image)
        aug_image = np.clip((aug_image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    # Random rotation
    if random.random() > 0.5 and augment_config['rotation_range'] > 0:
        angle = random.uniform(-augment_config['rotation_range'],
                               augment_config['rotation_range'])
        aug_image, aug_corners = rotate_image_and_corners(aug_image, aug_corners, angle)
    
    # Random horizontal flip
    if augment_config['flip_horizontal'] and random.random() > 0.5:
        aug_image, aug_corners = flip_horizontal(aug_image, aug_corners)
    
    # Random perspective transform
    if random.random() > 0.7:
        strength = augment_config['perspective_strength']
        aug_image, aug_corners = random_perspective(aug_image, aug_corners, strength)
    
    return aug_image, aug_corners


def rotate_image_and_corners(
    image: np.ndarray,
    corners: np.ndarray,
    angle: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image and adjust corner coordinates.
    
    Args:
        image: Input image
        corners: Corner coordinates (4, 2)
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image and adjusted corners
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # Transform corners
    ones = np.ones((corners.shape[0], 1))
    corners_homogeneous = np.hstack([corners, ones])
    rotated_corners = M @ corners_homogeneous.T
    rotated_corners = rotated_corners.T
    
    return rotated, rotated_corners


def flip_horizontal(
    image: np.ndarray,
    corners: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flip image horizontally and adjust corners.
    """
    h, w = image.shape[:2]
    flipped = cv2.flip(image, 1)
    
    flipped_corners = corners.copy()
    flipped_corners[:, 0] = w - corners[:, 0]
    
    # Reorder corners to maintain consistent ordering
    # Assuming corners are ordered: top-left, top-right, bottom-right, bottom-left
    # After flip: top-right, top-left, bottom-left, bottom-right
    flipped_corners = flipped_corners[[1, 0, 3, 2]]
    
    return flipped, flipped_corners


def random_perspective(
    image: np.ndarray,
    corners: np.ndarray,
    strength: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random perspective transformation.
    
    Args:
        image: Input image
        corners: Corner coordinates (4, 2)
        strength: Maximum perspective distortion (0-1)
        
    Returns:
        Transformed image and corners
    """
    h, w = image.shape[:2]
    
    # Original corner positions
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Add random perturbations
    max_shift = strength * min(h, w)
    dst_pts = src_pts + np.random.uniform(-max_shift, max_shift, (4, 2)).astype(np.float32)
    
    # Compute perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    # Apply to image
    warped = cv2.warpPerspective(image, M, (w, h))
    
    # Transform corners
    ones = np.ones((corners.shape[0], 1))
    corners_h = np.hstack([corners, ones])
    warped_corners = (M @ corners_h.T).T
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:3]  # Dehomogenize
    
    return warped, warped_corners


def add_noise(image: np.ndarray, noise_std: float = 10.0) -> np.ndarray:
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Add Gaussian blur to image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
