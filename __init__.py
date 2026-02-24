"""
Image Tampering Localization Package
A complete research project for detecting and localizing manipulated regions in digital images
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Patch-based CNN for Image Tampering Localization"

# Import main components for easier access
from .model import get_model, TamperingCNN, ImprovedTamperingCNN
from .dataset import TamperingDataset, get_data_loaders, get_image_mask_pairs
from .patch_utils import extract_patches, extract_test_patches, reconstruct_probability_map
from .metrics import compute_iou, compute_pixel_accuracy, evaluate_single_image
from .visualize import visualize_single_prediction, plot_training_history

__all__ = [
    'get_model',
    'TamperingCNN',
    'ImprovedTamperingCNN',
    'TamperingDataset',
    'get_data_loaders',
    'get_image_mask_pairs',
    'extract_patches',
    'extract_test_patches',
    'reconstruct_probability_map',
    'compute_iou',
    'compute_pixel_accuracy',
    'evaluate_single_image',
    'visualize_single_prediction',
    'plot_training_history',
]
