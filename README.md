# Dogs vs Cats — Foundation of AI Coursework Project

[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-89%25-blue)](https://github.com/2499146834-arch/Foundation-of-AI)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-orange)](https://www.tensorflow.org/)
[![Model](https://img.shields.io/badge/Model-3--Layer%20CNN-green)]()

Binary image classification (cat vs dog) using a **3-layer CNN trained from scratch** on the Kaggle Dogs vs Cats dataset. This project systematically evaluates the impact of optimizers, regularization, and data augmentation through controlled experiments.

**Validation Accuracy: 89%** | **Parameters: ~19M** | **Dataset: 25,000 images** | **Input: 150×150**

---

## Quick Start

```bash
# 1. Clone & enter project
git clone https://github.com/2499146834-arch/Foundation-of-AI.git
cd Foundation-of-AI

# 2. Install dependencies
pip install tensorflow matplotlib scikit-learn pillow pandas jupyter

# 3. Prepare data
# Download Kaggle Dogs vs Cats from: https://www.kaggle.com/c/dogs-vs-cats/data
# Place images under data/train/cat/, data/train/dog/, data/validation/cat/, data/validation/dog/
python split_data.py

# 4. Train baseline
python train_baseline.py

# 5. Run optimization experiments
python optimization_experiments.py

# 6. Evaluate
jupyter notebook evaluate_model.ipynb
```

---

## Project Structure

```
├── data_loader.py                    # Data pipeline & augmentation (Member A)
├── visualize_augmentation.py         # Augmentation preview tool (Member A)
├── train_baseline.py                 # Baseline CNN training script (Member B)
├── optimization_experiments.py       # 5-variant controlled experiments (Member C)
├── evaluate_model.ipynb              # Evaluation & error analysis (Member D)
├── split_data.py                     # Train/val split utility
│
├── baseline_cnn.h5                   # Best baseline model (~218 MB, Git LFS)
├── baseline_training_curves.png      # Baseline accuracy & loss curves
├── experiment_results.csv            # Optimization experiment metrics
├── experiment_comparison.png         # Variant comparison bar chart
│
├── 可视化/                            # Evaluation outputs
│   ├── confusion_matrix.png
│   ├── detailed_error_analysis.png
│   ├── 分类报告.png
│   └── error reason.txt
│
├── Dogs_vs_Cats_Report_final.docx    # Full project report (Chinese)
├── 07 AI Master.pptx                 # Presentation slides
├── 演示.mp4                           # Demo walkthrough video
│
└── 资料/                              # Planning documents
    ├── Project Topic.docx
    ├── 分工.docx
    ├── 分工第1-4部分记录.docx
    └── CDS521 Group_project term2.pdf
```

---

## Team Members & Division

| Member | Role | Key Deliverables |
|--------|------|-----------------|
| **A** | Data Preparation & Augmentation | `data_loader.py`, `visualize_augmentation.py` |
| **B** | Baseline CNN Design & Training | `train_baseline.py`, `baseline_cnn.h5` |
| **C** | Model Optimization | `optimization_experiments.py`, 5 model variants |
| **D** | Evaluation & Error Analysis | `evaluate_model.ipynb`, confusion matrix, visualizations |

---

## Model Architecture

A **3-layer CNN** built with the Keras Sequential API — designed to be deep enough to learn hierarchical features while remaining trainable on a consumer GPU.

| Layer | Output Shape | Parameters |
|-------|-------------|-----------|
| Conv2D (32, 3×3) → MaxPool (2×2) | (74, 74, 32) | 896 |
| Conv2D (64, 3×3) → MaxPool (2×2) | (36, 36, 64) | 18,496 |
| Conv2D (128, 3×3) → MaxPool (2×2) | (17, 17, 128) | 73,856 |
| Flatten | (36,992) | 0 |
| Dense (512, ReLU) | (512) | 18,940,416 |
| Dense (1, Sigmoid) | (1) | 513 |

**Total: ~19,034,179 parameters**

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Cross-Entropy |
| Input Size | 150×150×3 |
| Batch Size | 32 |
| Epochs | 20 |
| Data Augmentation | rotation ±20°, shift 20%, shear 20%, zoom 20%, horizontal flip |

---

## Optimization Experiments

To isolate the contribution of each component, **5 controlled variants** were tested against the baseline:

| Experiment | Optimizer | Regularization | Augmentation | Val Accuracy | Key Insight |
|-----------|-----------|---------------|-------------|-------------|------------|
| **Adam (baseline)** | Adam | None | ✅ | **~89%** | Best overall; adaptive LR handles noisy gradients well |
| SGD + Momentum | SGD (lr=0.01, mom=0.9) | None | ✅ | ~86% | Converges slower; higher LR sensitivity |
| Adam + Dropout | Adam | Dropout(0.5) | ✅ | ~88% | Slightly reduced overfitting but marginal gain at this depth |
| Adam + BatchNorm | Adam | BatchNormalization | ✅ | ~88% | Stabilizes training but adds compute overhead |
| Adam (no aug) | Adam | None | ❌ | ~80% | **Largest drop** — augmentation is critical for generalization |
| RMSprop | RMSprop | None | ✅ | ~87% | Similar to SGD; Adam still outperforms |

> **Key finding:** Data augmentation is the single most impactful component (~9% accuracy gain). Among optimizers, Adam provides the best convergence for this architecture.

---

## Baseline Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **89%** |
| Cat Precision | 0.89 |
| Cat Recall | 0.89 |
| Dog Precision | 0.89 |
| Dog Recall | 0.89 |
| F1-Score (both classes) | 0.89 |

The balanced precision/recall across both classes indicates no significant class bias — the model distinguishes cats and dogs equally well despite their visual similarities.

---

## Improved Version

A transfer learning version using **MobileNetV2** achieves **98.98% accuracy** with only **2.3M parameters** (88% fewer than the baseline):

[github.com/2499146834-arch/cats-vs-dogs](https://github.com/2499146834-arch/cats-vs-dogs)

Key improvements:
- **MobileNetV2 backbone** (ImageNet pretrained) replaces from-scratch CNN
- **GlobalAveragePooling2D** instead of Flatten → 88% parameter reduction
- **Two-phase training**: freeze backbone → train head → fine-tune top layers
- **ReduceLROnPlateau + EarlyStopping** for adaptive training
- **Color augmentation** (brightness, contrast, saturation)

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥2.17 | Model training & inference |
| `matplotlib` | ≥3.7 | Training curves & visualizations |
| `scikit-learn` | ≥1.3 | Metrics & confusion matrix |
| `pillow` | ≥10.0 | Image processing |
| `pandas` | ≥2.0 | Results aggregation |
| `jupyter` | ≥1.0 | Evaluation notebook |

---

## License

This project is coursework for the **Foundation of AI (CDS 521)** course. All rights reserved.
