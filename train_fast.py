"""
Quick Training Script - Train in 15-20 minutes
FOR EXAM EVALUATION
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import config
from model_fast import get_fast_model, count_parameters
from dataset import get_data_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0
    
    for images, masks, _ in tqdm(train_loader, desc='Training'):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Calculate IoU
        pred = (outputs > 0.5).float()
        intersection = (pred * masks).sum()
        union = pred.sum() + masks.sum() - intersection
        iou = (intersection + 1e-8) / (union + 1e-8)
        
        total_loss += loss.item()
        total_iou += iou.item()
    
    return total_loss / len(train_loader), total_iou / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc='Validation'):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate IoU
            pred = (outputs > 0.5).float()
            intersection = (pred * masks).sum()
            union = pred.sum() + masks.sum() - intersection
            iou = (intersection + 1e-8) / (union + 1e-8)
            
            total_loss += loss.item()
            total_iou += iou.item()
    
    return total_loss / len(val_loader), total_iou / len(val_loader)


def train_fast():
    """Quick training for exam"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"FAST TRAINING - EXAM VERSION")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader = get_data_loaders(
        config.DATASET_ROOT,
        images_dir=config.IMAGES_DIR,
        masks_dir=config.MASKS_DIR,
        train_split=0.9,  # Use more for training, less for validation
        batch_size=32,  # Even larger batch - faster training
        random_seed=42
    )
    
    # Create model
    print("\nCreating model...")
    model = get_fast_model().to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training - just 5 epochs for exam (10-15 minutes total)
    num_epochs = 5
    best_iou = 0
    save_path = "./output/best_model_improved.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training for {num_epochs} epochs (~10-15 min)")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        print(f"Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_type': 'fast_unet',
                'model_state_dict': model.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, save_path)
            print(f"✓ Best model saved! IoU: {val_iou:.4f}")
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"Best Val IoU: {best_iou:.4f}")
    print(f"Model saved: {save_path}")
    print(f"{'='*60}\n")
    
    return model


if __name__ == "__main__":
    train_fast()
