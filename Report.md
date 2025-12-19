### 1: Dataset Choice and Reasoning**

For this task, a hybrid dataset approach was utilized by combining subsets of the **ATR (Deep Human Parsing)** and **iMaterialist (Fashion)** datasets, sourced via the Hugging Face Hub.
**Reasoning for Selection:**
* **Resource Efficiency:** Standard datasets like DeepFashion2 are extremely large and difficult to manage on cloud-based platforms like Google Colab or Kaggle without high-tier storage. By using the Hugging Face `datasets` library, data was streamed/loaded efficiently within RAM limits.
* **Class Granularity:** The ATR dataset provides a specific 18-class map (including Background, Hat, Hair, Face, Upper-clothes, etc.) which is ideal for "Human Parsing"—a subset of segmentation that understands the structural relationship between the body and the clothes.
* **Quality vs. Size Trade-off:** While ATR provides smaller, more manageable images, some masks can be "coarse." To balance this, a subset of iMaterialist was integrated to introduce more complex fashion items and improve the model's ability to generalize to high-fashion garments.

---

### 2: Model Architecture**

The system implements **DeepLabV3** with a **ResNet-101** backbone.
**Architectural Decisions:**
* **Pivoting from Transformers:** Initial experiments with *SegFormer* and *Mask2Former* resulted in "Out of Memory" (OOM) errors on the T4 GPU. DeepLabV3 was selected as a more memory-efficient alternative that still maintains high-tier performance.
* **Atrous Spatial Pyramid Pooling (ASPP):** This is the core strength of DeepLabV3. It uses atrous (dilated) convolutions at different rates to capture multi-scale context. In clothing segmentation, this is vital because it allows the model to "see" a small accessory like a belt while simultaneously understanding a large garment like a dress.
* **ResNet-101 Backbone:** A deep residual network provides a powerful feature extraction layer. Using a pre-trained version (on ImageNet) allows the model to leverage existing knowledge of shapes and textures, significantly speeding up convergence.

---

### **3. Loss Function Selection & Mathematical Reasoning**

The model is trained using a **Hybrid Loss Function**, defined as the sum of **Cross-Entropy (CE) Loss** and **Dice Loss**. This combination is specifically designed to address the challenges of multi-class human parsing.

$$L_{total} = L_{CE} + L_{Dice}$$

#### **A. Weighted Cross-Entropy Loss ()**

Cross-Entropy measures the performance of a classification model whose output is a probability value between 0 and 1.

* **Formula: $L_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$** 
* **Role:** It provides smooth gradients and ensures that the model correctly identifies the category of each pixel. However, CE is "pixel-greedy"; it focuses on maximizing overall pixel accuracy. In clothing segmentation, where the "Background" class (Class 0) may occupy 70% of the image, the model can achieve high accuracy by simply predicting background everywhere, ignoring small items like "Belt" or "Sunglasses."

#### **B. Dice Loss ($L_{Dice}$)**

Dice Loss is derived from the Sørensen–Dice coefficient, which measures the overlap between two samples.

* **Formula: $L_{Dice} = 1 - \frac{2 \sum y \hat{y}}{\sum y + \sum \hat{y}}$** 
* **Role:** Unlike CE, Dice Loss is **region-based**. It calculates the intersection over union (IoU) for each class. It treats small objects (e.g., a "Scarf") with the same importance as large objects (e.g., "Pants") because it looks at the percentage of overlap rather than the raw pixel count. This directly combats the **class imbalance** problem inherent in the ATR and iMaterialist datasets.

#### **C. Why the Combination?**

Using this hybrid approach provides two distinct advantages:

1. **Gradient Stability:** CE provides a stable, convex optimization landscape that helps the model converge quickly in early epochs.
2. **Boundary & Scale Sensitivity:** Dice Loss forces the model to refine the boundaries of clothing items and ensures that even the smallest categories (like Class 8: Belt) are effectively segmented, which standard loss functions often overlook.

---

**4. Performance Analysis**
The model's performance is evaluated using two primary metrics: **Pixel Accuracy** and **Mean Intersection over Union (mIoU)**.
* **Pixel Accuracy:** Measures the percentage of pixels correctly classified.
* **mIoU:** The gold standard for segmentation; it calculates the average overlap between the predicted and ground truth masks across all 18 classes.

**Current Progress (Training in progress):**
* **Epoch 1 Results:**
* **mIoU:** 0.5111
* **Pixel Accuracy:** 92.96%
* **Loss:** 0.3851

**Analysis:** Achieving over 50% mIoU in the first epoch is a strong indicator that the **DeepLabV3 + Combined Loss** strategy is effective. Most of the early accuracy is driven by the model correctly identifying the background and large torso garments. As training continues, we expect the mIoU to rise as the model learns to distinguish between more fine-grained classes like "Left-arm" vs. "Right-arm" and smaller accessories.
`[PLACEHOLDER: Insert Final Metrics Table and Graph here]`

---

Moving to the final section of the technical report and the remaining structural deliverables.

### 5: System Limitations and Conditions**

**A. System Strengths (Capabilities)**
* **Robust Multi-Class Parsing:** Unlike simple binary foreground/background segmenters, this system identifies 18 distinct regions, including symmetrical body parts (Left vs. Right arm/shoe), which is essential for virtual try-on accuracy.
* **Contextual Integrity:** By using a ResNet-101 backbone, the model maintains high spatial resolution, allowing it to distinguish between overlapping items (e.g., a "Bag" strap over "Upper-clothes").
* **Optimized Inference:** Despite the depth of the model, the inference pipeline is optimized for standard resolution, providing a balanced trade-off between detail and processing speed.


**B. Drawbacks and Weaknesses**
* **Boundary Sharpness:** Due to the "coarse" nature of the ATR dataset labels, the model may exhibit slight "bleeding" at the edges of garments, particularly in high-contrast areas.
* **Occlusion Sensitivity:** While generally robust, heavily occluded items (e.g., a long coat covering most of the pants) can lead to fragmented segmentation masks.


**C. Capturing Conditions & Constraints (Limitations)**
To achieve the reported mIoU, the following conditions are recommended:
* **Lighting:** Uniform lighting is preferred. Harsh shadows or extreme backlighting significantly degrade pixel classification accuracy.
* **Subject Pose:** The model is trained on "Human Parsing" data; therefore, the subject should be standing or in a clear pose. Seated or highly contorted poses may lead to misclassification of limbs.
* **Background:** While the model handles diverse backgrounds, extreme clutter that mimics clothing textures (like patterned curtains) may introduce noise into the mask.