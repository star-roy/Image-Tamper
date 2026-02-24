# Image Tampering Localization Using Patch-Based Convolutional Neural Networks

A complete research project for detecting and localizing manipulated regions in digital images using deep learning.

## 🎯 Project Overview

This system detects and localizes various types of image tampering including:
- Copy-move manipulation
- Splicing
- Object removal (inpainting)
- AI-based content editing (generative fill)

The system outputs a binary tampering mask highlighting altered regions using a lightweight patch-based CNN approach.

## 📋 Features

- **Modular Architecture**: Clean, well-documented code organized into logical modules
- **Patch-Based Approach**: Extracts overlapping patches for fine-grained localization
- **Lightweight CNN**: CPU-compatible model suitable for student research
- **Complete Pipeline**: From data loading to visualization
- **Comprehensive Evaluation**: IoU, Precision, Recall, F1-score, Pixel Accuracy
- **Visualization Tools**: Side-by-side comparisons, heatmaps, overlays

## 🏗️ Project Structure

```
Project/
├── config.py              # Configuration parameters
├── dataset.py             # Dataset loader for image-mask pairs
├── patch_utils.py         # Patch extraction and reconstruction utilities
├── model.py               # CNN architecture definitions
├── train.py               # Training pipeline
├── inference.py           # Inference on test images
├── metrics.py             # Evaluation metrics
├── visualize.py           # Visualization utilities
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Installation

### 1. Clone or download this project

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- PyTorch >= 1.9.0
- OpenCV >= 4.5.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- scikit-learn >= 0.24.0
- Pillow >= 8.0.0
- tqdm >= 4.60.0

### 3. Prepare your dataset

Organize your dataset in the following structure:

```
dataset/
├── images/              # Original/tampered images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── masks/               # Ground truth masks (0=authentic, 1=tampered)
    ├── image1.png
    ├── image2.png
    └── ...
```

**Supported datasets:**
- CASIA v2 Tampered Dataset
- Columbia Image Splicing Dataset
- Any custom dataset with image-mask pairs

## 🚀 Quick Start

### 1. Configure Parameters

Edit [config.py](config.py) to set your dataset paths:

```python
DATASET_ROOT = "./dataset"  # Path to your dataset
IMAGES_DIR = "images"       # Subdirectory name for images
MASKS_DIR = "masks"         # Subdirectory name for masks
OUTPUT_DIR = "./output"     # Output directory
```

### 2. Train the Model

```bash
python train.py --dataset ./dataset --epochs 20 --batch_size 64
```

**Training arguments:**
- `--dataset`: Path to dataset root directory
- `--model`: Model type (`basic` or `improved`, default: `basic`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--no_balance`: Disable automatic data balancing
- `--save_path`: Custom path to save model

**Example:**
```bash
python train.py --dataset ./my_dataset --epochs 15 --batch_size 128 --lr 0.0005
```

### 3. Run Inference

**Single image:**
```bash
python inference.py --model ./output/best_model.pth --image ./test_image.jpg
```

**Batch inference on directory:**
```bash
python inference.py --model ./output/best_model.pth --input_dir ./test_images --output_dir ./results
```

**Inference arguments:**
- `--model`: Path to trained model (required)
- `--image`: Path to single test image
- `--input_dir`: Directory of test images (for batch processing)
- `--output_dir`: Output directory for results (default: `./output/inference`)
- `--model_type`: Model architecture (`basic` or `improved`)
- `--device`: Device to use (`cpu` or `cuda`)
- `--no_viz`: Disable visualization for single image

### 4. Evaluate Results

```bash
python metrics.py --pred_dir ./results/masks --gt_dir ./dataset/masks --output ./evaluation.txt
```

**Evaluation arguments:**
- `--pred_dir`: Directory with predicted masks (required)
- `--gt_dir`: Directory with ground truth masks (required)
- `--output`: Output file for results
- `--mask_suffix`: Suffix for mask files (default: `_mask.png`)

### 5. Visualize Results

**Single image visualization:**
```bash
python visualize.py --mode single --image ./test.jpg --gt_mask ./gt_mask.png --pred_mask ./pred_mask.png --save ./viz.png
```

**Grid visualization (multiple samples):**
```bash
python visualize.py --mode grid --image_dir ./images --gt_dir ./gt_masks --pred_dir ./pred_masks --num_samples 5 --save ./grid.png
```

## 📊 Pipeline Details

### 1. Data Loading
- Automatically finds image-mask pairs
- Supports multiple naming conventions
- 80/20 train/validation split by default
- Resizes images to 256×256
- Normalizes using ImageNet statistics

### 2. Patch Extraction
- Extracts overlapping 32×32 patches with stride 16
- Labels patches as tampered if ≥30% pixels are manipulated
- Balances dataset to handle class imbalance
- Returns patches with coordinates for reconstruction

### 3. CNN Architecture

**Basic Model (Lightweight):**
```
Conv(3→32) + ReLU + MaxPool
Conv(32→64) + ReLU + MaxPool
Conv(64→128) + ReLU + MaxPool
Flatten
FC(128) + ReLU + Dropout
FC(1) + Sigmoid
```

**Parameters:** ~180K trainable parameters  
**Loss:** Binary Cross-Entropy  
**Optimizer:** Adam

### 4. Training
- Batch size: 64 (configurable)
- Learning rate: 0.001 (configurable)
- Epochs: 10-20 recommended
- Automatic validation during training
- Saves best model based on validation accuracy
- Progress bars using tqdm

### 5. Inference
- Extracts patches from test image
- Predicts tampering probability per patch
- Reconstructs full probability map by averaging overlaps
- Applies threshold (0.5) to generate binary mask
- Morphological operations (opening + closing) for noise removal

### 6. Evaluation
- **IoU (Intersection over Union)**: Overlap between prediction and ground truth
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### 7. Visualization
- Side-by-side comparison of original, ground truth, prediction
- Probability heatmaps with color mapping
- Red overlay on original image
- Difference maps (TP/FP/FN visualization)

## 🎛️ Configuration Parameters

Key parameters in [config.py](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | 256 | Resize images to this size |
| `PATCH_SIZE` | 32 | Size of extracted patches |
| `STRIDE` | 16 | Stride for overlapping patches |
| `TAMPERED_THRESHOLD` | 0.3 | Min ratio to label patch as tampered |
| `BATCH_SIZE` | 64 | Training batch size |
| `LEARNING_RATE` | 0.001 | Adam optimizer learning rate |
| `NUM_EPOCHS` | 20 | Number of training epochs |
| `PROB_THRESHOLD` | 0.5 | Threshold for binary mask |
| `MORPH_KERNEL_SIZE` | 5 | Kernel size for morphological ops |

## 📈 Expected Results

**Training Performance:**
- Training time: ~10-30 minutes on CPU (depends on dataset size)
- Validation accuracy: ~85-95% (patch-level)

**Localization Performance:**
- IoU: 0.6-0.8 (varies by dataset and tampering type)
- F1-Score: 0.7-0.85
- Best for: Copy-move, splicing
- Challenging for: Subtle edits, AI-generated content

## 🔬 Research Extensions

**Potential improvements:**
1. **Data Augmentation**: Add rotation, flipping, color jittering
2. **Advanced Architectures**: Use ResNet, EfficientNet backbones
3. **Multi-Scale Patches**: Combine predictions from multiple patch sizes
4. **Attention Mechanisms**: Add spatial attention layers
5. **Self-Supervised Learning**: Pre-train on authentic images
6. **Error Level Analysis**: Incorporate JPEG compression artifacts
7. **Frequency Domain**: Add DCT-based features

## 💡 Usage Tips

1. **Small Dataset**: Use data augmentation and reduce model complexity
2. **Class Imbalance**: The system automatically balances data, but you can adjust the ratio in `patch_utils.py`
3. **Poor Localization**: Try reducing stride (more overlap) or adjusting tampered threshold
4. **Overfitting**: Add more dropout, reduce model size, or get more training data
5. **CPU Training**: Expect 10-30 min for 20 epochs on typical datasets
6. **GPU Acceleration**: Set `device='cuda'` in training arguments if GPU available

## 📝 Code Examples

### Train with custom parameters:
```python
from train import train_model

model, history = train_model(
    dataset_root='./my_dataset',
    model_type='basic',
    num_epochs=15,
    batch_size=128,
    learning_rate=0.0005,
    balance_data=True,
    save_path='./models/my_model.pth'
)
```

### Inference on single image:
```python
from inference import inference_single_image

prob_map, binary_mask = inference_single_image(
    model_path='./output/best_model.pth',
    image_path='./test.jpg',
    model_type='basic',
    device='cpu',
    visualize=True
)
```

### Evaluate predictions:
```python
from metrics import evaluate_dataset, print_evaluation_results

results = evaluate_dataset(
    pred_dir='./predictions',
    gt_dir='./ground_truth'
)

print_evaluation_results(results)
```

## 🐛 Troubleshooting

**Issue:** `FileNotFoundError: Images directory not found`
- **Solution:** Check that `DATASET_ROOT`, `IMAGES_DIR`, and `MASKS_DIR` in config.py point to correct locations

**Issue:** Low validation accuracy
- **Solution:** Train longer, increase model capacity, check data quality, ensure masks are correct

**Issue:** Poor localization on test images
- **Solution:** Reduce stride for more overlap, adjust probability threshold, check if test distribution matches training

**Issue:** Out of memory error
- **Solution:** Reduce batch size, use smaller image size, process fewer images at once

**Issue:** Masks are noisy
- **Solution:** Increase morphological kernel size, adjust probability threshold, post-process with larger kernels

## 📚 Dataset References

**CASIA v2:**
- Contains 12,614 images (7,491 authentic, 5,123 tampered)
- Various tampering types: splicing, copy-move
- Download: Search for "CASIA v2 dataset"

**Columbia Image Splicing:**
- 933 authentic and 912 spliced images
- High-quality ground truth masks
- Download: [Columbia DVMM Research](http://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)

## 🤝 Contributing

This is a research project template. Feel free to:
- Modify architectures
- Add new features
- Experiment with different approaches
- Share improvements

## 📄 License

This project is for educational and research purposes.

## ✉️ Contact

For questions or issues, please create an issue in the project repository.

## 🎓 Citation

If you use this code in your research, please cite:

```
@misc{image_tampering_localization,
  title={Image Tampering Localization Using Patch-Based CNNs},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/tampering-localization}}
}
```

## 🌟 Acknowledgments

This project implements patch-based tampering detection inspired by various research papers in image forensics and deep learning.

---

**Happy Research! 🔍🖼️**
