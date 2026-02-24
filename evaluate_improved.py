"""
Evaluation script for improved model
Calculates IoU, Precision, Recall, F1-Score on test dataset
"""

import torch
import numpy as np
import os
from tqdm import tqdm
import cv2
from inference_improved import load_improved_model, preprocess_image, predict_tampering
import config


def calculate_metrics(pred_mask, gt_mask, threshold=0.5):
    """Calculate evaluation metrics"""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    # True Positives, False Positives, False Negatives, True Negatives
    TP = np.sum((pred_binary == 1) & (gt_binary == 1))
    FP = np.sum((pred_binary == 1) & (gt_binary == 0))
    FN = np.sum((pred_binary == 0) & (gt_binary == 1))
    TN = np.sum((pred_binary == 0) & (gt_binary == 0))
    
    # IoU
    intersection = TP
    union = TP + FP + FN
    iou = intersection / (union + 1e-8)
    
    # Pixel Accuracy
    pixel_acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    
    # Precision
    precision = TP / (TP + FP + 1e-8)
    
    # Recall
    recall = TP / (TP + FN + 1e-8)
    
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return {
        'iou': iou,
        'pixel_acc': pixel_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_improved_model(model_path, images_dir, masks_dir, threshold=0.5, save_results=True):
    """
    Evaluate improved model on dataset
    
    Args:
        model_path: Path to trained model
        images_dir: Directory containing test images
        masks_dir: Directory containing ground truth masks
        threshold: Probability threshold
        save_results: Save detailed results to file
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model, model_info = load_improved_model(model_path, device)
    
    # Get image files
    image_files = sorted([f for f in os.listdir(images_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    print(f"\nEvaluating on {len(image_files)} images")
    print(f"Threshold: {threshold}\n")
    
    all_metrics = []
    detailed_results = []
    
    # Process each image
    for img_file in tqdm(image_files, desc='Evaluating'):
        img_path = os.path.join(images_dir, img_file)
        base_name = os.path.splitext(img_file)[0]
        
        # Find corresponding mask
        possible_mask_names = [
            f"{base_name}_mask.png",
            f"{base_name}_gt.png",
            f"{base_name}.png",
            img_file
        ]
        
        mask_path = None
        for mask_name in possible_mask_names:
            potential_path = os.path.join(masks_dir, mask_name)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path is None:
            print(f"Warning: No mask found for {img_file}")
            continue
        
        try:
            # Load ground truth mask
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"Warning: Could not load mask {mask_path}")
                continue
            
            # Preprocess image
            image_tensor, _, original_size = preprocess_image(img_path, config.IMAGE_SIZE)
            
            # Predict
            prob_map, binary_mask = predict_tampering(model, image_tensor, device, threshold)
            
            # Resize prediction to match ground truth
            if gt_mask.shape != binary_mask.shape:
                binary_mask = cv2.resize(binary_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
            
            # Calculate metrics
            metrics = calculate_metrics(binary_mask, gt_mask, threshold=0.5)
            all_metrics.append(metrics)
            
            # Store detailed results
            detailed_results.append({
                'image': img_file,
                'iou': metrics['iou'],
                'pixel_acc': metrics['pixel_acc'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            })
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    # Calculate average metrics
    avg_metrics = {
        'iou': np.mean([m['iou'] for m in all_metrics]),
        'pixel_acc': np.mean([m['pixel_acc'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1': np.mean([m['f1'] for m in all_metrics])
    }
    
    std_metrics = {
        'iou': np.std([m['iou'] for m in all_metrics]),
        'pixel_acc': np.std([m['pixel_acc'] for m in all_metrics]),
        'precision': np.std([m['precision'] for m in all_metrics]),
        'recall': np.std([m['recall'] for m in all_metrics]),
        'f1': np.std([m['f1'] for m in all_metrics])
    }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {model_info['type']}")
    print(f"Number of images: {len(all_metrics)}")
    print(f"Threshold: {threshold}")
    print("-"*60)
    print(f"{'Metric':<20} {'Mean':>10} {'Std Dev':>10}")
    print("-"*60)
    print(f"{'IoU':<20} {avg_metrics['iou']:>10.4f} {std_metrics['iou']:>10.4f}")
    print(f"{'Pixel Accuracy':<20} {avg_metrics['pixel_acc']:>10.4f} {std_metrics['pixel_acc']:>10.4f}")
    print(f"{'Precision':<20} {avg_metrics['precision']:>10.4f} {std_metrics['precision']:>10.4f}")
    print(f"{'Recall':<20} {avg_metrics['recall']:>10.4f} {std_metrics['recall']:>10.4f}")
    print(f"{'F1-Score':<20} {avg_metrics['f1']:>10.4f} {std_metrics['f1']:>10.4f}")
    print("="*60)
    
    # Save results
    if save_results:
        output_dir = os.path.dirname(model_path)
        results_file = os.path.join(output_dir, 'evaluation_improved.txt')
        
        with open(results_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("IMPROVED MODEL EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Model: {model_info['type']}\n")
            f.write(f"Model path: {model_path}\n")
            f.write(f"Number of images: {len(all_metrics)}\n")
            f.write(f"Threshold: {threshold}\n\n")
            
            f.write(f"{'Metric':<20} {'Mean':>10} {'Std Dev':>10}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'IoU':<20} {avg_metrics['iou']:>10.4f} {std_metrics['iou']:>10.4f}\n")
            f.write(f"{'Pixel Accuracy':<20} {avg_metrics['pixel_acc']:>10.4f} {std_metrics['pixel_acc']:>10.4f}\n")
            f.write(f"{'Precision':<20} {avg_metrics['precision']:>10.4f} {std_metrics['precision']:>10.4f}\n")
            f.write(f"{'Recall':<20} {avg_metrics['recall']:>10.4f} {std_metrics['recall']:>10.4f}\n")
            f.write(f"{'F1-Score':<20} {avg_metrics['f1']:>10.4f} {std_metrics['f1']:>10.4f}\n")
            f.write("="*60 + "\n\n")
            
            f.write("PER-IMAGE RESULTS:\n")
            f.write("-"*60 + "\n")
            for result in detailed_results:
                f.write(f"\n{result['image']}:\n")
                f.write(f"  IoU: {result['iou']:.4f}\n")
                f.write(f"  Pixel Accuracy: {result['pixel_acc']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall: {result['recall']:.4f}\n")
                f.write(f"  F1-Score: {result['f1']:.4f}\n")
        
        print(f"\nDetailed results saved to: {results_file}")
    
    return avg_metrics, std_metrics, detailed_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate improved tampering detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--images_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--masks_dir', type=str, required=True,
                       help='Directory containing ground truth masks')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results file')
    
    args = parser.parse_args()
    
    avg_metrics, std_metrics, detailed = evaluate_improved_model(
        args.model,
        args.images_dir,
        args.masks_dir,
        args.threshold,
        save_results=not args.no_save
    )
