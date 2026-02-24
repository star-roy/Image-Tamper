"""
Configuration file for image tampering localization project
"""

import torch

# Image preprocessing
IMAGE_SIZE = 256  # Resize all images to this size
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Patch extraction
PATCH_SIZE = 32
STRIDE = 8  # Reduced from 16 for smoother detection
TAMPERED_THRESHOLD = 0.3  # 30% of patch pixels must be tampered to label as tampered

# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
TRAIN_VAL_SPLIT = 0.8
RANDOM_SEED = 42

# Inference
PROB_THRESHOLD = 0.5  # Threshold to convert probability map to binary mask (increased for better precision)
GAUSSIAN_KERNEL_SIZE = 5  # Kernel size for Gaussian smoothing
GAUSSIAN_SIGMA = 1.5  # Sigma for Gaussian smoothing
MIN_AREA_THRESHOLD = 50  # Minimum area (pixels) for detected regions (reduced for finer detection)

# Morphological operations for mask refinement
MORPH_KERNEL_SIZE = 3  # Reduced from 5 for finer details

# Improved model settings
IMPROVED_MODEL_TYPE = 'unet_resnet34'  # Options: 'unet_resnet34', 'unet_resnet18', 'attention_unet'
IMPROVED_BATCH_SIZE = 8  # Batch size for training improved models
IMPROVED_LEARNING_RATE = 1e-4  # Learning rate for improved models
IMPROVED_EPOCHS = 50  # Number of epochs for improved models

# Model architecture
CONV1_CHANNELS = 32
CONV2_CHANNELS = 64
CONV3_CHANNELS = 128
FC_HIDDEN = 128

# Paths (update these based on your dataset location)
DATASET_ROOT = "./dataset"  # Root directory containing images and masks
IMAGES_DIR = "images"  # Subdirectory name for images
MASKS_DIR = "masks"    # Subdirectory name for masks
OUTPUT_DIR = "./output"  # Directory to save results
MODEL_SAVE_PATH = "./output/best_model.pth"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
