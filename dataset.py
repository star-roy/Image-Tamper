"""
Dataset loader for image tampering detection
Supports CASIA v2, Columbia, and similar datasets with image-mask pairs
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from sklearn.model_selection import train_test_split
import config


class TamperingDataset(Dataset):
    """
    Dataset class for loading tampered images and their ground truth masks
    """
    
    def __init__(self, image_paths, mask_paths, transform=None, mask_transform=None):
        """
        Args:
            image_paths: List of paths to images
            mask_paths: List of paths to corresponding masks
            transform: Transformations to apply to images
            mask_transform: Transformations to apply to masks
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform
        
        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            # Default: resize and convert to tensor
            mask = transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE))(mask)
            mask = transforms.ToTensor()(mask)
            # Binarize mask (0 or 1)
            mask = (mask > 0.5).float()
        
        return image, mask, img_path


class PatchDataset(Dataset):
    """
    Dataset class that returns patches extracted from images
    """
    
    def __init__(self, patches, labels):
        """
        Args:
            patches: Tensor of patches (N, C, H, W)
            labels: Tensor of labels (N,)
        """
        self.patches = patches
        self.labels = labels
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx], self.labels[idx]


def get_image_mask_pairs(dataset_root, images_dir='images', masks_dir='masks'):
    """
    Find all image-mask pairs in the dataset directory
    
    Args:
        dataset_root: Root directory of dataset
        images_dir: Subdirectory containing images
        masks_dir: Subdirectory containing masks
    
    Returns:
        image_paths: List of image file paths
        mask_paths: List of mask file paths
    """
    images_path = os.path.join(dataset_root, images_dir)
    masks_path = os.path.join(dataset_root, masks_dir)
    
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images directory not found: {images_path}")
    if not os.path.exists(masks_path):
        raise FileNotFoundError(f"Masks directory not found: {masks_path}")
    
    # Get all image files
    image_files = sorted([f for f in os.listdir(images_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    
    image_paths = []
    mask_paths = []
    
    for img_file in image_files:
        img_path = os.path.join(images_path, img_file)
        
        # Try different mask naming conventions
        possible_mask_names = [
            img_file,  # Same name
            img_file.rsplit('.', 1)[0] + '_mask.png',
            img_file.rsplit('.', 1)[0] + '_gt.png',
            img_file.rsplit('.', 1)[0] + '.png',
        ]
        
        mask_found = False
        for mask_name in possible_mask_names:
            mask_path = os.path.join(masks_path, mask_name)
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)
                mask_found = True
                break
        
        if not mask_found:
            print(f"Warning: No mask found for {img_file}")
    
    print(f"Found {len(image_paths)} image-mask pairs")
    return image_paths, mask_paths


def get_data_loaders(dataset_root, images_dir='images', masks_dir='masks', 
                    train_split=0.8, batch_size=32, random_seed=42):
    """
    Create train and validation data loaders
    
    Args:
        dataset_root: Root directory of dataset
        images_dir: Subdirectory containing images
        masks_dir: Subdirectory containing masks
        train_split: Fraction of data for training
        batch_size: Batch size for data loaders
        random_seed: Random seed for reproducibility
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Get image-mask pairs
    image_paths, mask_paths = get_image_mask_pairs(dataset_root, images_dir, masks_dir)
    
    # Split into train and validation
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, train_size=train_split, random_state=random_seed
    )
    
    # Define transformations
    image_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = TamperingDataset(train_imgs, train_masks, 
                                    transform=image_transform, 
                                    mask_transform=mask_transform)
    val_dataset = TamperingDataset(val_imgs, val_masks, 
                                  transform=image_transform, 
                                  mask_transform=mask_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def create_patch_dataloader(patches, labels, batch_size=64, shuffle=True):
    """
    Create a DataLoader for patch-based training
    
    Args:
        patches: Tensor of patches
        labels: Tensor of labels
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    dataset = PatchDataset(patches, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader
