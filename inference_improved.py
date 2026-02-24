"""
Improved Inference for Image Tampering Detection
Uses U-Net architecture for direct pixel-level predictions
Much faster and more accurate than patch-based approach
"""

import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import os
from tqdm import tqdm
import config
from model_improved import get_improved_model
import matplotlib.pyplot as plt


def load_improved_model(model_path, device='cpu'):
    """
    Load trained improved model from checkpoint
    
    Args:
        model_path: Path to saved model
        device: Device to load model on
    
    Returns:
        model: Loaded model
        model_info: Model information
    """
    # Check if file exists
    if not os.path.exists(model_path):
        # Try alternative paths
        alt_path = model_path.replace('best_model_improved.pth', 'best_model.pth')
        if os.path.exists(alt_path):
            print(f"\n⚠️  WARNING: Improved model not found!")
            print(f"Found old model at: {alt_path}")
            print(f"This is the old patch-based model with poor accuracy.")
            print(f"\nTo use the improved model, train it first:")
            print(f"  python train_fast.py  (10-15 minutes)\n")
            raise FileNotFoundError(f"Improved model not found: {model_path}\nTrain with: python train_fast.py")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}\nTrain with: python train_fast.py")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model type from checkpoint
    model_type = checkpoint.get('model_type', 'unet_resnet34')
    
    # Load appropriate model
    if model_type == 'fast_unet':
        from model_fast import get_fast_model
        model = get_fast_model()
    else:
        model = get_improved_model(model_type, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model: {model_type}")
    print(f"Validation IoU: {checkpoint.get('val_iou', 'N/A'):.4f}")
    print(f"Validation Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    model_info = {
        'type': model_type,
        'val_iou': checkpoint.get('val_iou', 0),
        'val_loss': checkpoint.get('val_loss', 0),
        'epoch': checkpoint.get('epoch', 0)
    }
    
    return model, model_info


def preprocess_image(image_path, target_size=256):
    """
    Load and preprocess image for inference
    
    Args:
        image_path: Path to image file
        target_size: Target image size
    
    Returns:
        image_tensor: Preprocessed image tensor (1, C, H, W)
        original_image: Original PIL image
        original_size: Original image size (W, H)
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size  # (W, H)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD)
    ])
    
    image_tensor = transform(original_image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_image, original_size


def postprocess_mask(prob_map, threshold=0.5, apply_morphology=True, 
                     kernel_size=3, min_area=50):
    """
    Post-process probability map to binary mask
    
    Args:
        prob_map: Probability map (H, W)
        threshold: Threshold for binarization
        apply_morphology: Apply morphological operations
        kernel_size: Kernel size for morphology
        min_area: Minimum area for connected components
    
    Returns:
        binary_mask: Binary mask (H, W)
    """
    # Threshold
    binary_mask = (prob_map > threshold).astype(np.uint8)
    
    if apply_morphology:
        # Morphological operations to clean up mask
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Opening: removes small noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Closing: fills small holes
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small connected components
    if min_area > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # Keep only large components
        filtered_mask = np.zeros_like(binary_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered_mask[labels == i] = 1
        
        binary_mask = filtered_mask
    
    return binary_mask


def predict_tampering(model, image_tensor, device='cpu', threshold=0.5):
    """
    Predict tampering mask for a single image
    
    Args:
        model: Trained U-Net model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        device: Device
        threshold: Threshold for binary mask
    
    Returns:
        prob_map: Probability map (H, W)
        binary_mask: Binary tampering mask (H, W)
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)  # (1, 1, H, W)
        
        # Convert to numpy
        prob_map = outputs.squeeze().cpu().numpy()  # (H, W)
    
    # Post-process to binary mask
    binary_mask = postprocess_mask(
        prob_map,
        threshold=threshold,
        apply_morphology=True,
        kernel_size=config.MORPH_KERNEL_SIZE,
        min_area=config.MIN_AREA_THRESHOLD
    )
    
    return prob_map, binary_mask


def create_visualization(original_image, prob_map, binary_mask):
    """
    Create visualization combining original image, probability map, mask, and overlay
    
    Args:
        original_image: PIL Image
        prob_map: Probability map (H, W)
        binary_mask: Binary mask (H, W)
    
    Returns:
        vis_image: Combined visualization
    """
    # Resize maps to original image size
    orig_w, orig_h = original_image.size
    
    prob_map_resized = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Convert original image to numpy
    orig_np = np.array(original_image)
    
    # Create probability heatmap
    prob_vis = (prob_map_resized * 255).astype(np.uint8)
    prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
    prob_vis = cv2.cvtColor(prob_vis, cv2.COLOR_BGR2RGB)
    
    # Create mask visualization
    mask_vis = (mask_resized * 255).astype(np.uint8)
    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)
    
    # Create overlay
    overlay = orig_np.copy()
    red_overlay = np.zeros_like(orig_np)
    red_overlay[:, :, 0] = 255  # Red channel
    
    # Apply red overlay where mask is positive
    mask_bool = mask_resized > 0
    overlay[mask_bool] = cv2.addWeighted(
        orig_np[mask_bool], 0.6, red_overlay[mask_bool], 0.4, 0
    )
    
    # Combine all visualizations
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(orig_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(prob_vis)
    axes[1].set_title('Tampering Probability', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(mask_vis, cmap='gray')
    axes[2].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return vis_image


def inference_single_image(model_path, image_path, output_dir=None, 
                          threshold=None, show_visualization=True):
    """
    Run inference on a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        output_dir: Directory to save results (None = don't save)
        threshold: Probability threshold (None = use config)
        show_visualization: Whether to display visualization
    
    Returns:
        prob_map: Probability map
        binary_mask: Binary mask
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model, model_info = load_improved_model(model_path, device)
    
    # Set threshold
    if threshold is None:
        threshold = config.PROB_THRESHOLD
    
    print(f"\nProcessing image: {image_path}")
    
    # Preprocess image
    image_tensor, original_image, original_size = preprocess_image(
        image_path, config.IMAGE_SIZE
    )
    
    print(f"Image size: {original_size[0]}x{original_size[1]}")
    print(f"Threshold: {threshold}")
    
    # Predict
    print("\nPredicting tampering...")
    prob_map, binary_mask = predict_tampering(model, image_tensor, device, threshold)
    
    # Calculate statistics
    tampered_pixels = binary_mask.sum()
    total_pixels = binary_mask.size
    tampered_ratio = tampered_pixels / total_pixels
    
    print(f"\nResults:")
    print(f"  Tampered pixels: {tampered_pixels:,} ({tampered_ratio*100:.2f}%)")
    print(f"  Max probability: {prob_map.max():.4f}")
    print(f"  Mean probability: {prob_map.mean():.4f}")
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save binary mask
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        
        # Resize to original size
        mask_resized = cv2.resize(mask_uint8, original_size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mask_path, mask_resized)
        
        # Save probability map
        prob_path = os.path.join(output_dir, f"{base_name}_prob.png")
        prob_resized = cv2.resize(prob_map, original_size, interpolation=cv2.INTER_LINEAR)
        prob_vis = (prob_resized * 255).astype(np.uint8)
        prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
        cv2.imwrite(prob_path, prob_vis)
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{base_name}_visualization.png")
        vis_image = create_visualization(original_image, prob_map, binary_mask)
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  Mask: {mask_path}")
        print(f"  Probability map: {prob_path}")
        print(f"  Visualization: {vis_path}")
    
    # Show visualization
    if show_visualization:
        vis_image = create_visualization(original_image, prob_map, binary_mask)
        plt.figure(figsize=(16, 4))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return prob_map, binary_mask


def inference_directory(model_path, input_dir, output_dir, threshold=None):
    """
    Run inference on all images in a directory
    
    Args:
        model_path: Path to trained model
        input_dir: Directory containing input images
        output_dir: Directory to save results
        threshold: Probability threshold
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model, model_info = load_improved_model(model_path, device)
    
    # Set threshold
    if threshold is None:
        threshold = config.PROB_THRESHOLD
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, 'masks')
    probs_dir = os.path.join(output_dir, 'probability_maps')
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(probs_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get all images
    image_files = [f for f in os.listdir(input_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    print(f"\nFound {len(image_files)} images to process")
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {threshold}\n")
    
    # Process each image
    for img_file in tqdm(image_files, desc='Processing images'):
        img_path = os.path.join(input_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        
        try:
            # Preprocess
            image_tensor, original_image, original_size = preprocess_image(
                img_path, config.IMAGE_SIZE
            )
            
            # Predict
            prob_map, binary_mask = predict_tampering(model, image_tensor, device, threshold)
            
            # Resize to original size
            prob_resized = cv2.resize(prob_map, original_size, interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Save results
            mask_path = os.path.join(masks_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, (mask_resized * 255).astype(np.uint8))
            
            prob_path = os.path.join(probs_dir, f"{base_name}_prob.png")
            prob_vis = (prob_resized * 255).astype(np.uint8)
            prob_vis = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
            cv2.imwrite(prob_path, prob_vis)
            
            vis_path = os.path.join(vis_dir, f"{base_name}_visualization.png")
            vis_image = create_visualization(original_image, prob_map, binary_mask)
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    print(f"\n✓ Processing complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved inference for tampering detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image (for single image inference)')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Path to input directory (for batch inference)')
    parser.add_argument('--output_dir', type=str, default='./results_improved',
                       help='Path to output directory')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Probability threshold (default: from config)')
    parser.add_argument('--no_viz', action='store_true',
                       help='Do not show visualization')
    
    args = parser.parse_args()
    
    if args.image:
        # Single image inference
        inference_single_image(
            args.model,
            args.image,
            args.output_dir,
            args.threshold,
            show_visualization=not args.no_viz
        )
    elif args.input_dir:
        # Batch inference
        inference_directory(
            args.model,
            args.input_dir,
            args.output_dir,
            args.threshold
        )
    else:
        print("Error: Please specify either --image or --input_dir")
        parser.print_help()
