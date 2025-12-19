import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os
from huggingface_hub import hf_hub_download

"""
DeepLabV3 Clothing Segmentation Inference Script

Model Architecture:
- Backbone: ResNet-101 (ImageNet pretrained)
- Decoder: Atrous Spatial Pyramid Pooling (ASPP)
- Output: 18-class segmentation mask (512×512)
- Classifier: Modified final layer to output 18 classes

Training Configuration (from notebook):
- Dataset: mattmdjaga/human_parsing_dataset (70/15/15 train/val/test split)
- Loss: 0.6 × CrossEntropyLoss + 0.4 × DiceLoss
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Augmentation: Resize(512x512), HorizontalFlip, Normalization
- Batch Size: 8
- Epochs: 20
- Metrics: Pixel Accuracy, mIoU

Inference Features:
- Single image or batch processing
- GPU/CPU support
- Color-coded visualization with 18 clothing classes
- Save segmentation masks and overlays
- HuggingFace Hub integration for model download
"""
    
    # Class labels for clothing parsing
CLASS_LABELS = {
    0: 'background',
    1: 'hat',
    2: 'hair',
    3: 'face',
    4: 'upper_body',
    5: 'right_arm',
    6: 'left_arm',
    7: 'glove',
    8: 'coat',
    9: 'jacket',
    10: 'shirt',
    11: 'sweater',
    12: 'skirt',
    13: 'pants',
    14: 'shoes',
    15: 'bag',
    16: 'scarf',
    17: 'dress'
}

# Color palette for visualization (18 classes)
PALETTE = [
    [0, 0, 0],        # background
    [255, 0, 0],      # hat
    [255, 85, 0],     # hair
    [255, 170, 0],    # face
    [255, 255, 0],    # upper_body
    [170, 255, 0],    # right_arm
    [85, 255, 0],     # left_arm
    [0, 255, 0],      # glove
    [0, 255, 85],     # coat
    [0, 255, 170],    # jacket
    [0, 255, 255],    # shirt
    [0, 170, 255],    # sweater
    [0, 85, 255],     # skirt
    [0, 0, 255],      # pants
    [85, 0, 255],     # shoes
    [170, 0, 255],    # bag
    [255, 0, 255],    # scarf
    [255, 0, 170],    # dress
]


class DeepLabV3Inference:
    def __init__(self, model_path=None, device=None, num_classes=18, hf_repo=None):
        """
        Initialize the inference model
        
        Args:
            model_path: Path to the saved model checkpoint
            device: torch device ('cuda' or 'cpu')
            num_classes: Number of output classes (18 for clothing segmentation)
            hf_repo: HuggingFace repo ID to download model from (e.g., 'oshaban/deeplabv3_clothes')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        
        # Load model architecture - matches notebook architecture exactly
        self.model = deeplabv3_resnet101(weights="DEFAULT")
        
        # Modify classifier for custom number of classes (18 clothing categories)
        # This matches the notebook: model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Also modify aux_classifier if it exists (for auxiliary loss during training)
        if hasattr(self.model, 'aux_classifier') and self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        
        # Move to device before loading weights (important for CUDA compatibility)
        self.model = self.model.to(self.device)
        
        # Load checkpoint - prioritize: model_path > HuggingFace > pretrained
        if model_path and os.path.exists(model_path):
            # Load from local path
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded from local path: {model_path}")
        
        elif hf_repo:
            # Download from HuggingFace Hub
            try:
                model_file = hf_hub_download(
                    repo_id=hf_repo,
                    filename="deeplab_model_checkpoint/model.pt",
                    cache_dir=".cache"
                )
                checkpoint = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                print(f"✅ Model loaded from HuggingFace repo: {hf_repo}")
            except Exception as e:
                print(f"⚠️ Failed to download from HuggingFace: {e}")
                print(f"   Using ImageNet pretrained weights instead")
        
        else:
            # Use ImageNet pretrained weights (backbone only, classifier is random)
            print("ℹ️ Using ImageNet pretrained ResNet101 backbone (classifier randomly initialized)")
        
        self.model.eval()
        
        # Preprocessing transform - matches notebook augmentation exactly
        # Resize to 512x512, Normalize with ImageNet stats, convert to tensor
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed tensor and original image
        """
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        
        # Store original for later use
        original_size = image_np.shape[:2]
        
        augmented = self.transform(image=image_np)
        image_tensor = augmented["image"].unsqueeze(0).to(self.device)
        
        return image_tensor, image, original_size
    
    @torch.no_grad()
    def predict(self, image_path):
        """
        Generate segmentation prediction
        
        Args:
            image_path: Path to input image
            
        Returns:
            Prediction mask (H, W)
        """
        image_tensor, original_image, original_size = self.preprocess_image(image_path)
        
        outputs = self.model(image_tensor)["out"]
        prediction = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
        # Resize prediction back to original size
        prediction_resized = Image.fromarray(prediction.astype(np.uint8))
        prediction_resized = prediction_resized.resize(
            (original_size[1], original_size[0]),
            Image.NEAREST
        )
        
        return np.array(prediction_resized), original_image
    
    def colorize_mask(self, mask):
        """
        Convert grayscale mask to RGB using color palette
        
        Args:
            mask: Segmentation mask (H, W)
            
        Returns:
            RGB image (H, W, 3)
        """
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_idx, color in enumerate(PALETTE):
            colored[mask == class_idx] = color
        
        return colored
    
    def visualize_prediction(self, image_path, save_path=None, show=True):
        """
        Visualize prediction alongside original image
        
        Args:
            image_path: Path to input image
            save_path: Path to save visualization (optional)
            show: Whether to display the plot
        """
        prediction, original_image = self.predict(image_path)
        colored_mask = self.colorize_mask(prediction)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Segmentation mask
        axes[1].imshow(colored_mask)
        axes[1].set_title("Segmentation Mask")
        axes[1].axis("off")
        
        # Overlay
        overlay = np.uint8(0.6 * np.array(original_image) + 0.4 * colored_mask)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        if show:
            plt.show()
        
        return colored_mask
    
    def save_mask(self, image_path, output_path):
        """
        Save segmentation mask as image
        
        Args:
            image_path: Path to input image
            output_path: Path to save mask
        """
        prediction, _ = self.predict(image_path)
        colored_mask = self.colorize_mask(prediction)
        
        Image.fromarray(colored_mask).save(output_path)
        print(f"✅ Mask saved to {output_path}")
    
    def batch_predict(self, input_dir, output_dir, save_masks=True, save_viz=True):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            save_masks: Whether to save segmentation masks
            save_viz: Whether to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [
            f for f in os.listdir(input_dir)
            if Path(f).suffix.lower() in image_extensions
        ]
        
        print(f"Processing {len(image_files)} images...")
        
        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            base_name = Path(image_file).stem
            
            try:
                # Save mask
                if save_masks:
                    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
                    self.save_mask(image_path, mask_path)
                
                # Save visualization
                if save_viz:
                    viz_path = os.path.join(output_dir, f"{base_name}_viz.png")
                    self.visualize_prediction(
                        image_path,
                        save_path=viz_path,
                        show=False
                    )
                
                print(f"✅ Processed: {image_file}")
            
            except Exception as e:
                print(f"❌ Failed to process {image_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description="DeepLabV3 Clothing Segmentation Inference")
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to single image for inference'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Input directory for batch processing'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Path to local model checkpoint'
    )
    parser.add_argument(
        '--hf_repo',
        type=str,
        default='oshaban/deeplabv3_clothes',
        help='HuggingFace repo ID to download model from (default: oshaban/deeplabv3_clothes)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--save_mask',
        action='store_true',
        help='Save segmentation masks'
    )
    parser.add_argument(
        '--save_viz',
        action='store_true',
        default=True,
        help='Save visualizations'
    )
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = DeepLabV3Inference(
        model_path=args.model,
        device=args.device,
        hf_repo=args.hf_repo
    )
    
    # Single image inference
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Visualize
        base_name = Path(args.image).stem
        viz_path = os.path.join(args.output_dir, f"{base_name}_viz.png")
        inference.visualize_prediction(args.image, save_path=viz_path, show=False)
        
        # Save mask if requested
        if args.save_mask:
            mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
            inference.save_mask(args.image, mask_path)
    
    # Batch inference
    elif args.input_dir:
        if not os.path.exists(args.input_dir):
            print(f"❌ Input directory not found: {args.input_dir}")
            return
        
        inference.batch_predict(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            save_masks=args.save_mask,
            save_viz=args.save_viz
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
