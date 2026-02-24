"""
Fast Inference Script - Works with fast trained model
"""

import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import os
from tqdm import tqdm
import config
from model_fast import get_fast_model
import matplotlib.pyplot as plt


def load_fast_model(model_path, device='cpu'):
    """Load fast model"""
    model = get_fast_model()
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model - Val IoU: {checkpoint.get('val_iou', 0):.4f}")
    return model


def preprocess_image(image_path, target_size=256):
    """Preprocess image"""
    original_image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    image_tensor = transform(original_image).unsqueeze(0)
    return image_tensor, original_image


def postprocess_mask(prob_map, threshold=0.5):
    """Clean up mask"""
    binary_mask = (prob_map > threshold).astype(np.uint8)
    
    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    return binary_mask


def predict_single(model_path, image_path, output_dir, threshold=0.5):
    """Quick prediction"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model...")
    model = load_fast_model(model_path, device)
    
    print(f"Processing: {image_path}")
    image_tensor, original_image = preprocess_image(image_path)
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        prob_map = output.squeeze().cpu().numpy()
    
    binary_mask = postprocess_mask(prob_map, threshold)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Resize to original
    orig_size = original_image.size
    prob_resized = cv2.resize(prob_map, orig_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(binary_mask, orig_size, interpolation=cv2.INTER_NEAREST)
    
    # Save mask
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, (mask_resized * 255).astype(np.uint8))
    
    # Save probability map
    prob_path = os.path.join(output_dir, f"{base_name}_prob.png")
    prob_vis = (prob_resized * 255).astype(np.uint8)
    prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
    cv2.imwrite(prob_path, prob_vis)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(prob_resized, cmap='jet')
    axes[1].set_title('Probability')
    axes[1].axis('off')
    
    axes[2].imshow(mask_resized, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    # Overlay
    overlay = np.array(original_image).copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    mask_bool = mask_resized > 0
    overlay[mask_bool] = cv2.addWeighted(overlay[mask_bool], 0.6, red[mask_bool], 0.4, 0)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    vis_path = os.path.join(output_dir, f"{base_name}_vis.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  Mask: {mask_path}")
    print(f"  Visualization: {vis_path}")
    
    return prob_map, binary_mask


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results_improved')
    parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    predict_single(args.model, args.image, args.output_dir, args.threshold)
