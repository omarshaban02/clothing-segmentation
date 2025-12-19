# Clothing Segmentation System (CYSHIELD Assessment)


This repository contains a deep learning pipeline for human parsing and clothing segmentation using **DeepLabV3** with a **ResNet-101** backbone.
## 🚀 Quick Start


1. **Clone the repo:**
```bash
git clone https://github.com/omarshaban02/clothing-segmentation.git
cd clothing-segmentation

```


2. **Install Dependencies:**
```bash
pip install torch torchvision numpy opencv-python Pillow matplotlib

```


3. **Run Inference:**
Place your image as `input.jpg` and run:
```python
# Execute the inference script provided in the repo
python inference.py --input input.jpg --output result.png

```




## 📊 Methodology


* **Architecture:** DeepLabV3 (ASPP + ResNet-101)
* **Loss:** Combined Weighted Cross-Entropy + Dice Loss
* **Dataset:** Hybrid ATR & iMaterialist subset


## 📈 Results


* **mIoU:** 0.5111 (Epoch 1) / `[PLACEHOLDER: Final]`
* **Pixel Accuracy:** 92.96% (Epoch 1) / `[PLACEHOLDER: Final]`


