# Clothing Segmentation System

This repository contains a deep learning pipeline for human parsing and clothing segmentation using **DeepLabV3** with a **ResNet-101** backbone trained on the mattmdjaga/human_parsing_dataset.

## 🎯 Features

- **18-class clothing segmentation** (background, hat, hair, face, upper_body, arms, gloves, coat, jacket, shirt, sweater, skirt, pants, shoes, bag, scarf, dress)
- **High-accuracy predictions** with combined Cross-Entropy + Dice Loss
- **Single image & batch processing** modes
- **Visualization tools** with overlay and colored masks
- **GPU acceleration** support (CUDA/CPU)
- **Easy-to-use command-line interface**

## 📦 Installation

### Requirements
- Python 3.7+
- CUDA 11.0+ (for GPU acceleration, optional)

### Setup
```bash
pip install torch torchvision
pip install albumentations pillow matplotlib
pip install numpy tqdm
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Single Image Inference
```bash
python inference.py --image path/to/image.jpg --output_dir ./results
```

### With Model Checkpoint
```bash
python inference.py --image path/to/image.jpg --model deeplab_model.pt --output_dir ./results
```

### Batch Processing
```bash
python inference.py --input_dir ./images --output_dir ./results --save_mask --save_viz
```

### Save Segmentation Masks
```bash
python inference.py --image path/to/image.jpg --output_dir ./masks --save_mask
```

## 📋 Command-Line Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--image` | str | Path to single image for inference |
| `--input_dir` | str | Input directory for batch processing |
| `--output_dir` | str | Output directory for results (default: `./output`) |
| `--model` | str | Path to model checkpoint (optional) |
| `--device` | str | Device to use: `cuda` or `cpu` (default: auto-detect) |
| `--save_mask` | flag | Save raw segmentation masks |
| `--save_viz` | flag | Save visualization images (default: enabled) |

## 👕 Clothing Classes (18 Categories)

| ID | Class | ID | Class |
|----|----|----|----|
| 0 | Background | 9 | Jacket |
| 1 | Hat | 10 | Shirt |
| 2 | Hair | 11 | Sweater |
| 3 | Face | 12 | Skirt |
| 4 | Upper Body | 13 | Pants |
| 5 | Right Arm | 14 | Shoes |
| 6 | Left Arm | 15 | Bag |
| 7 | Glove | 16 | Scarf |
| 8 | Coat | 17 | Dress |

## 🏗️ Architecture & Training

### Model Architecture
- **Backbone:** ResNet-101 (ImageNet pretrained)
- **Decoder:** Atrous Spatial Pyramid Pooling (ASPP)
- **Output:** 18-class segmentation mask (512×512)

### Training Configuration
- **Dataset:** mattmdjaga/human_parsing_dataset (70/15/15 train/val/test split)
- **Loss Function:** 0.6 × Cross-Entropy + 0.4 × Dice Loss
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Augmentation:** Resize (512×512), Horizontal Flip, Normalization
- **Batch Size:** 8
- **Epochs:** 20

### Metrics
- **Pixel Accuracy:** Per-pixel classification accuracy
- **mIoU:** Mean Intersection-over-Union across all classes

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Pixel Accuracy | 93.85%+ |
| mIoU | 0.5402+ |
| Inference Speed | ~200ms per image (GPU) |

## 📁 Project Structure

```
.
├── inference.py           # Inference script
├── deeplabv3-clothes.ipynb # Training notebook
├── README.md              # This file
└── inspect_dataset.ipynb  # For dataset exploration
```

## Usage Examples

### Python API
```python
from inference import DeepLabV3Inference

# Initialize
model = DeepLabV3Inference(
    model_path="model.pt",
    device="cuda"
)

# Single prediction
mask, original = model.predict("image.jpg")

# Visualize
model.visualize_prediction("image.jpg", save_path="result.png")

# Batch process
model.batch_predict(
    input_dir="./images",
    output_dir="./results",
    save_masks=True,
    save_viz=True
)
```

## Dataset Information

- **Dataset:** [mattmdjaga/human_parsing_dataset](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset)
- **Total Samples:** ~17,000+ images
- **Resolution:** Variable (resized to 512×512 during training)
- **Annotations:** Pixel-level clothing category labels

## Model Details

The model uses DeepLabV3, a state-of-the-art semantic segmentation architecture that combines:
- **Atrous Convolutions** for multi-scale feature extraction
- **ASPP Module** for capturing contextual information at multiple scales
- **ResNet-101 Backbone** for robust feature extraction

## Tips for Best Results

1. **Image Quality:** Higher resolution images generally produce better segmentation
2. **Clothing Visibility:** Ensure clothing is clearly visible in the image
3. **Background:** Cluttered backgrounds may affect accuracy
4. **GPU Usage:** Use `--device cuda` for significantly faster inference

## License

This project is provided as-is for research and development purposes.


