"""
Evaluation metrics for tampering localization
Computes IoU, Precision, Recall, F1-score, and Pixel Accuracy
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import os
from tqdm import tqdm
from PIL import Image
import cv2


def compute_iou(pred_mask, gt_mask):
    """
    Compute Intersection over Union (IoU)
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
    
    Returns:
        iou: IoU score
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    iou = intersection / union
    return iou


def compute_pixel_accuracy(pred_mask, gt_mask):
    """
    Compute pixel-wise accuracy
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
    
    Returns:
        accuracy: Pixel accuracy
    """
    correct = (pred_mask == gt_mask).sum()
    total = pred_mask.size
    accuracy = correct / total
    return accuracy


def compute_precision_recall_f1(pred_mask, gt_mask):
    """
    Compute precision, recall, and F1-score
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
    
    Returns:
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, average='binary', zero_division=0
    )
    
    return precision, recall, f1


def compute_confusion_matrix(pred_mask, gt_mask):
    """
    Compute confusion matrix
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
    
    Returns:
        tn, fp, fn, tp: True negatives, false positives, false negatives, true positives
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat, labels=[0, 1]).ravel()
    
    return tn, fp, fn, tp


def evaluate_single_image(pred_mask, gt_mask):
    """
    Evaluate a single prediction
    
    Args:
        pred_mask: Predicted binary mask (H, W)
        gt_mask: Ground truth binary mask (H, W)
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Ensure masks are binary
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)
    
    # Compute metrics
    iou = compute_iou(pred_mask, gt_mask)
    accuracy = compute_pixel_accuracy(pred_mask, gt_mask)
    precision, recall, f1 = compute_precision_recall_f1(pred_mask, gt_mask)
    tn, fp, fn, tp = compute_confusion_matrix(pred_mask, gt_mask)
    
    metrics = {
        'iou': iou,
        'pixel_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }
    
    return metrics


def evaluate_dataset(pred_dir, gt_dir, mask_suffix='_mask.png'):
    """
    Evaluate predictions for an entire dataset
    
    Args:
        pred_dir: Directory containing predicted masks
        gt_dir: Directory containing ground truth masks
        mask_suffix: Suffix for mask files
    
    Returns:
        results: Dictionary containing per-image and average metrics
    """
    # Get all prediction files
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.png')]
    
    print(f"Evaluating {len(pred_files)} predictions...")
    
    all_metrics = []
    per_image_results = {}
    
    for pred_file in tqdm(pred_files):
        pred_path = os.path.join(pred_dir, pred_file)
        
        # Find corresponding ground truth
        # Try different naming conventions
        base_name = pred_file.replace(mask_suffix, '').replace('_mask', '').replace('.png', '')
        possible_gt_names = [
            base_name + '.png',
            base_name + '_mask.png',
            base_name + '_gt.png',
            pred_file
        ]
        
        gt_path = None
        for gt_name in possible_gt_names:
            test_path = os.path.join(gt_dir, gt_name)
            if os.path.exists(test_path):
                gt_path = test_path
                break
        
        if gt_path is None:
            print(f"Warning: No ground truth found for {pred_file}")
            continue
        
        # Load masks
        pred_mask = np.array(Image.open(pred_path).convert('L'))
        gt_mask = np.array(Image.open(gt_path).convert('L'))
        
        # Ensure same size
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # Binarize
        pred_mask = (pred_mask > 127).astype(np.uint8)
        gt_mask = (gt_mask > 127).astype(np.uint8)
        
        # Evaluate
        metrics = evaluate_single_image(pred_mask, gt_mask)
        all_metrics.append(metrics)
        per_image_results[pred_file] = metrics
    
    # Compute average metrics
    avg_metrics = {}
    metric_keys = ['iou', 'pixel_accuracy', 'precision', 'recall', 'f1_score']
    
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        avg_metrics[f'avg_{key}'] = np.mean(values)
        avg_metrics[f'std_{key}'] = np.std(values)
    
    results = {
        'per_image': per_image_results,
        'average': avg_metrics,
        'num_images': len(all_metrics)
    }
    
    return results


def print_evaluation_results(results):
    """
    Print evaluation results in a readable format
    
    Args:
        results: Results dictionary from evaluate_dataset
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of images evaluated: {results['num_images']}")
    print()
    
    avg = results['average']
    
    print(f"{'Metric':<20} {'Mean':<12} {'Std Dev':<12}")
    print("-"*60)
    print(f"{'IoU':<20} {avg['avg_iou']:.4f}       {avg['std_iou']:.4f}")
    print(f"{'Pixel Accuracy':<20} {avg['avg_pixel_accuracy']:.4f}       {avg['std_pixel_accuracy']:.4f}")
    print(f"{'Precision':<20} {avg['avg_precision']:.4f}       {avg['std_precision']:.4f}")
    print(f"{'Recall':<20} {avg['avg_recall']:.4f}       {avg['std_recall']:.4f}")
    print(f"{'F1-Score':<20} {avg['avg_f1_score']:.4f}       {avg['std_f1_score']:.4f}")
    print("="*60)


def save_evaluation_results(results, output_path):
    """
    Save evaluation results to a text file
    
    Args:
        results: Results dictionary
        output_path: Path to save results
    """
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Number of images evaluated: {results['num_images']}\n\n")
        
        avg = results['average']
        
        f.write(f"{'Metric':<20} {'Mean':<12} {'Std Dev':<12}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'IoU':<20} {avg['avg_iou']:.4f}       {avg['std_iou']:.4f}\n")
        f.write(f"{'Pixel Accuracy':<20} {avg['avg_pixel_accuracy']:.4f}       {avg['std_pixel_accuracy']:.4f}\n")
        f.write(f"{'Precision':<20} {avg['avg_precision']:.4f}       {avg['std_precision']:.4f}\n")
        f.write(f"{'Recall':<20} {avg['avg_recall']:.4f}       {avg['std_recall']:.4f}\n")
        f.write(f"{'F1-Score':<20} {avg['avg_f1_score']:.4f}       {avg['std_f1_score']:.4f}\n")
        f.write("="*60 + "\n\n")
        
        f.write("PER-IMAGE RESULTS:\n")
        f.write("-"*60 + "\n")
        for img_name, metrics in results['per_image'].items():
            f.write(f"\n{img_name}:\n")
            f.write(f"  IoU: {metrics['iou']:.4f}\n")
            f.write(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate tampering detection results')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory containing predicted masks')
    parser.add_argument('--gt_dir', type=str, required=True,
                       help='Directory containing ground truth masks')
    parser.add_argument('--output', type=str, default='./output/evaluation_results.txt',
                       help='Output file for results')
    parser.add_argument('--mask_suffix', type=str, default='_mask.png',
                       help='Suffix for mask files')
    
    args = parser.parse_args()
    
    # Evaluate
    results = evaluate_dataset(args.pred_dir, args.gt_dir, args.mask_suffix)
    
    # Print results
    print_evaluation_results(results)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_evaluation_results(results, args.output)
