"""
Improved Training Pipeline for Image Tampering Detection
Uses U-Net architecture with better loss functions and optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import config
from model_improved import get_improved_model, count_parameters
from dataset import get_data_loaders
import matplotlib.pyplot as plt


class DiceLoss(nn.Module):
    """Dice Loss for better segmentation performance"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss for robust training"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss - excellent for imbalanced segmentation"""
    
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # weight for false negatives
        self.beta = beta   # weight for false positives
        self.gamma = gamma  # focusing parameter
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (predictions * targets).sum()
        FP = ((1 - targets) * predictions).sum()
        FN = (targets * (1 - predictions)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


def calculate_iou(predictions, targets, threshold=0.5):
    """Calculate Intersection over Union"""
    pred_mask = (predictions > threshold).float()
    target_mask = targets
    
    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum() - intersection
    
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for images, masks, _ in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate metrics
        batch_iou = calculate_iou(outputs.detach(), masks)
        
        running_loss += loss.item()
        running_iou += batch_iou
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{batch_iou:.4f}'
        })
    
    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    
    return avg_loss, avg_iou


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation')
        
        for images, masks, _ in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            batch_iou = calculate_iou(outputs, masks)
            
            running_loss += loss.item()
            running_iou += batch_iou
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou:.4f}'
            })
    
    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    
    return avg_loss, avg_iou


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot IoU
    axes[1].plot(history['train_iou'], label='Train IoU')
    axes[1].plot(history['val_iou'], label='Val IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('Training and Validation IoU')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")


def train_improved_model(
    dataset_root=None,
    model_type='unet_resnet34',
    loss_type='focal_tversky',
    num_epochs=50,
    batch_size=8,
    learning_rate=1e-4,
    save_path=None,
    pretrained=True
):
    """
    Complete training pipeline for improved model
    
    Args:
        dataset_root: Root directory of dataset
        model_type: Model architecture ('unet_resnet34', 'unet_resnet18', 'attention_unet')
        loss_type: Loss function ('combined', 'focal_tversky', 'dice')
        num_epochs: Number of training epochs
        batch_size: Batch size 
        learning_rate: Learning rate
        save_path: Path to save best model
        pretrained: Use pretrained encoder
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Set paths
    if dataset_root is None:
        dataset_root = config.DATASET_ROOT
    if save_path is None:
        save_path = "./output/best_model_improved.pth"
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load dataset
    print("\n=== Loading Dataset ===")
    train_loader, val_loader = get_data_loaders(
        dataset_root,
        images_dir=config.IMAGES_DIR,
        masks_dir=config.MASKS_DIR,
        train_split=config.TRAIN_VAL_SPLIT,
        batch_size=batch_size,
        random_seed=config.RANDOM_SEED
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n=== Creating Model ===")
    print(f"Model type: {model_type}")
    print(f"Pretrained encoder: {pretrained}")
    
    model = get_improved_model(model_type, pretrained=pretrained)
    model = model.to(device)
    
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Define loss function
    print("\n=== Setting Up Training ===")
    print(f"Loss function: {loss_type}")
    
    if loss_type == 'combined':
        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    elif loss_type == 'focal_tversky':
        criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
    elif loss_type == 'dice':
        criterion = DiceLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }
    
    best_val_iou = 0.0
    patience_counter = 0
    early_stop_patience = 15
    
    # Training loop
    print("\n=== Training Started ===")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Early stopping patience: {early_stop_patience}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"{'-'*60}")
        
        # Train
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Print results
        print(f"\nResults:")
        print(f"  Train - Loss: {train_loss:.4f} | IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_iou)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.2e}", end='')
        if current_lr < old_lr:
            print(f" (reduced from {old_lr:.2e})", end='')
        print()
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_type': model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, save_path)
            
            print(f"  ✓ Best model saved! Val IoU: {val_iou:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"{'='*60}")
            break
        
        print(f"{'-'*60}")
    
    # Save training history plot
    plot_path = save_path.replace('.pth', '_history.png')
    plot_training_history(history, plot_path)
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"Training history: {plot_path}")
    print(f"{'='*60}\n")
    
    return model, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved tampering detection model')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Path to dataset root directory')
    parser.add_argument('--model', type=str, default='unet_resnet34',
                       choices=['unet_resnet34', 'unet_resnet18', 'unet_resnet50', 'attention_unet'],
                       help='Model architecture')
    parser.add_argument('--loss', type=str, default='focal_tversky',
                       choices=['combined', 'focal_tversky', 'dice'],
                       help='Loss function')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (reduce if OOM)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Do not use pretrained encoder')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save model')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_improved_model(
        dataset_root=args.dataset,
        model_type=args.model,
        loss_type=args.loss,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        pretrained=not args.no_pretrained
    )
