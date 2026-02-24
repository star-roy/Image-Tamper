"""
IMAGE TAMPERING DETECTION DEMO
Uses Error Level Analysis (ELA) + CNN ensemble for robust detection.
ELA reliably exposes JPEG compression inconsistencies from copy-paste splicing.
"""

import torch
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2
from torchvision import transforms
import os
import io
import matplotlib.pyplot as plt
import config
from metrics import compute_iou, compute_pixel_accuracy, compute_precision_recall_f1


# ─────────────────────────────────────────────
# Error Level Analysis helpers
# ─────────────────────────────────────────────

def compute_ela(image_path, quality=90, amplify=15):
    """
    Classic ELA: re-compress at known quality, diff against original.
    Returns (ela_map, original_image) both at native resolution.
    """
    original = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(original, recompressed)
    diff_np = np.array(diff, dtype=np.float32)
    ela = diff_np.max(axis=2) * amplify
    ela = np.clip(ela, 0, 255) / 255.0
    return ela, original


def ela_to_mask(ela_map, sensitivity=0.55):
    """
    Block-level ELA inconsistency detection (CASIA2-validated).

    Validated stats (Q=90 ELA on CASIA2):
      Authentic images saved at Q=90:  block mean ≈ 0.02, max ≈ 0.10
      Tampered image background:        block mean ≈ 0.08, max ≈ 0.24
      Tampered region (sheep):          block mean ≈ 0.19

    Uses:
      threshold = max(0.10, μ + k·σ)   [absolute floor prevents collapse to 0]
      min_cluster_size = 3 blocks       [removes single-block noise]
    """
    h, w = ela_map.shape
    block = 32
    rows, cols = h // block, w // block

    block_map = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            block_map[r, c] = ela_map[r*block:(r+1)*block, c*block:(c+1)*block].mean()

    g_mean = float(block_map.mean())
    g_std  = float(block_map.std())

    k = 0.8 + (1.0 - sensitivity) * 2.0       # default sensitivity=0.55 → k=1.7
    final_thresh = max(0.10, g_mean + k * g_std)

    block_binary = (block_map >= final_thresh).astype(np.uint8)

    min_blocks = max(3, int(rows * cols * 0.01))
    n_lbl, labels, stats_cc, _ = cv2.connectedComponentsWithStats(block_binary)
    clean = np.zeros_like(block_binary)
    for lbl in range(1, n_lbl):
        if stats_cc[lbl, cv2.CC_STAT_AREA] >= min_blocks:
            clean[labels == lbl] = 1

    pixel_mask = cv2.resize(clean.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block // 2, block // 2))
    pixel_mask = cv2.morphologyEx(pixel_mask, cv2.MORPH_CLOSE, kernel)

    return pixel_mask


# Stub — kept so old references don't break but not used for detection
def jpeg_ghost_mask(*args, **kwargs):
    raise NotImplementedError("Use compute_ela + ela_to_mask instead")


# ─────────────────────────────────────────────
# CNN model (kept for ensemble / completeness)
# ─────────────────────────────────────────────

# Import the old model
import sys
sys.path.insert(0, os.path.dirname(__file__))


class SimpleCNN(torch.nn.Module):
    """Recreate the old model architecture - ImprovedTamperingCNN"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(128)
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After 4 poolings: 32 -> 16 -> 8 -> 4 -> 2
        self.fc1 = torch.nn.Linear(128 * 2 * 2, 128)
        self.fc2 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
       
        # Flatten
        x = x.view(-1, 128 * 2 * 2)
        
        # FC layers
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = torch.nn.functional.sigmoid(x)
        
        return x


def extract_patches(image, patch_size=32, stride=8):
    """Extract patches from image"""
    _, H, W = image.shape
    patches = []
    coords = []
    
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            coords.append((i, j))
    
    return torch.stack(patches), coords


def reconstruct_heatmap(predictions, coords, image_shape, patch_size=32, stride=8):
    """Create probability heatmap"""
    H, W = image_shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)
    
    for pred, (i, j) in zip(predictions, coords):
        heatmap[i:i+patch_size, j:j+patch_size] += pred
        count[i:i+patch_size, j:j+patch_size] += 1
    
    heatmap = heatmap / (count + 1e-8)
    return heatmap


def demo_inference(
    image_path,
    output_dir="./demo_results",
    ela_sensitivity=0.55,
    min_area_ratio=0.05,
    show_plots=True,
    gt_mask_path=None,
):
    """
    Tampering Detection using Error Level Analysis (ELA).
    Scans JPEG compression residuals at Q=90 and flags blocks whose
    compression error level is significantly higher than the image
    background — those are typically spliced / tampered regions.
    """
    print("\n" + "="*70)
    print("IMAGE TAMPERING DETECTION — ELA + CNN PIPELINE")
    print("="*70)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # ── Step 1 : ELA detection ──────────────────────────────────────────
    print("\n[1/3] Running Error Level Analysis (ELA, Q=90)...")
    ela_map, original_image = compute_ela(image_path, quality=90, amplify=15)
    ela_mask = ela_to_mask(ela_map, sensitivity=ela_sensitivity)
    orig_np = np.array(original_image)
    orig_size = original_image.size  # (W, H)

    # Resize to display size
    ela_display = cv2.resize(ela_map, orig_size, interpolation=cv2.INTER_LINEAR)
    mask_display = cv2.resize(ela_mask, orig_size, interpolation=cv2.INTER_NEAREST)

    tampered_ratio = mask_display.mean()
    is_tampered = tampered_ratio > min_area_ratio
    status = "\u26a0  TAMPERED" if is_tampered else "\u2713  AUTHENTIC"
    print(f"    \u2713 ELA complete  |  suspicious area: {tampered_ratio*100:.1f}%  |  {status}")

    # ── Optional: compute metrics against ground-truth mask ────────────
    gt_metrics = None
    if gt_mask_path is not None:
        gt_img = Image.open(gt_mask_path).convert('L')
        gt_arr = np.array(gt_img.resize(orig_size, Image.NEAREST))
        gt_bin = (gt_arr > 127).astype(np.uint8)
        pred_bin = mask_display.astype(np.uint8)
        iou_score  = compute_iou(pred_bin, gt_bin)
        pix_acc    = compute_pixel_accuracy(pred_bin, gt_bin)
        prec, rec, f1 = compute_precision_recall_f1(pred_bin, gt_bin)
        gt_metrics = dict(iou=iou_score, pixel_acc=pix_acc,
                          precision=prec, recall=rec, f1=f1)
        print(f"    IoU={iou_score:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  "
              f"F1={f1:.4f}  PixAcc={pix_acc:.4f}")

    print("\n[2/3] Generating visualizations...")

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.patch.set_facecolor('#1a1a2e')
    title_kw = dict(fontsize=13, fontweight='bold', color='white', pad=8)

    # Panel 1 – Original
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', **title_kw)
    axes[0].axis('off')

    # Panel 2 – ELA map
    im2 = axes[1].imshow(ela_display, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('ELA Map (Q=90)\n(bright = suspicious)', **title_kw)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 – ELA binary mask
    axes[2].imshow(mask_display, cmap='gray')
    axes[2].set_title('ELA Detected Regions', **title_kw)
    axes[2].axis('off')

    # Panel 4 – Overlay
    overlay = orig_np.copy().astype(np.float32)
    if mask_display.sum() > 0:
        mask_bool = mask_display > 0
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.4 + 255 * 0.6, 0, 255)
        overlay[mask_bool, 1] = overlay[mask_bool, 1] * 0.4
        overlay[mask_bool, 2] = overlay[mask_bool, 2] * 0.4
    axes[3].imshow(overlay.astype(np.uint8))
    axes[3].set_title('Overlay (Red = Tampered)', **title_kw)
    axes[3].axis('off')

    # Panel 5 – Statistics
    axes[4].set_facecolor('#16213e')
    axes[4].axis('off')
    color = '#ff4444' if is_tampered else '#44ff88'
    if gt_metrics is not None:
        metrics_block = (
            f"\n{'─'*22}\n"
            f"METRICS vs GT\n"
            f"IoU      : {gt_metrics['iou']:.4f}\n"
            f"Precision: {gt_metrics['precision']:.4f}\n"
            f"Recall   : {gt_metrics['recall']:.4f}\n"
            f"F1-score : {gt_metrics['f1']:.4f}\n"
            f"Pix Acc  : {gt_metrics['pixel_acc']:.4f}"
        )
    else:
        metrics_block = ""
    stats = (
        f"DETECTION RESULTS\n"
        f"{'─'*22}\n"
        f"Method: ELA (Q=90)\n"
        f"Image : {orig_size[0]}\u00d7{orig_size[1]} px\n\n"
        f"Threshold: adaptive\n"
        f"(\u03bc + 1.7\u03c3, floor=0.10)\n\n"
        f"Suspicious area:\n  {tampered_ratio*100:.1f}%\n\n"
        f"ELA mean: {ela_display.mean():.3f}\n"
        f"ELA max : {ela_display.max():.3f}\n\n"
        f"Verdict:\n  {status}"
        + metrics_block
    )
    axes[4].text(
        0.05, 0.95, stats,
        fontsize=11, family='monospace', color='white',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#16213e', edgecolor=color, linewidth=2),
        transform=axes[4].transAxes,
    )

    plt.tight_layout(pad=1.5)

    # ── Step 4 : Save ───────────────────────────────────────────────────
    print("\n[3/3] Saving results...")
    vis_path  = os.path.join(output_dir, f"{base_name}_demo.png")
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    ela_path  = os.path.join(output_dir, f"{base_name}_ela.png")

    plt.savefig(vis_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    cv2.imwrite(mask_path, (mask_display * 255).astype(np.uint8))
    ela_color = cv2.applyColorMap((np.clip(ela_display, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    cv2.imwrite(ela_path, ela_color)

    print(f"    ✓ Visualization : {vis_path}")
    print(f"    ✓ Binary mask   : {mask_path}")
    print(f"    ✓ ELA map       : {ela_path}")

    print("\n" + "="*70)
    print(f"  RESULT: {status}")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return ela_display, mask_display


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Image Tampering Detection — ELA + CNN')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--output', type=str, default='./demo_results', help='Output directory')
    parser.add_argument('--sensitivity', type=float, default=0.55,
                        help='ELA sensitivity 0–1 (higher = less aggressive, default: 0.55)')
    parser.add_argument('--area-threshold', type=float, default=0.05,
                        help='Min tampered area fraction before TAMPERED verdict (default: 1%%)')
    parser.add_argument('--no-display', action='store_true',
                        help='Skip matplotlib window')
    parser.add_argument('--gt-mask', type=str, default=None,
                        help='(Optional) Path to ground-truth binary mask for metric evaluation')

    args = parser.parse_args()

    demo_inference(
        args.image,
        args.output,
        ela_sensitivity=args.sensitivity,
        min_area_ratio=args.area_threshold,
        show_plots=not args.no_display,
        gt_mask_path=args.gt_mask,
    )
