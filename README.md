# Dogs vs Cats — Foundation of AI Coursework Project

Team project for the **Foundation of AI** course. Binary image classification using a convolutional neural network trained from scratch on the Kaggle Dogs vs Cats dataset.

**Validation Accuracy: 89%** | **Dataset: 25,000 images** | **Framework: TensorFlow / Keras**

## Project Structure

```
├── data_loader.py                    # Data loading & augmentation pipeline (Member A)
├── visualize_augmentation.py         # Augmentation visualization (Member A)
├── train_baseline.py                 # Baseline CNN training (Member B)
├── optimization_experiments.py       # Model optimization experiments (Member C)
├── evaluate_model.ipynb              # Evaluation & error analysis (Member D)
├── split_data.py                     # Train/val data split utility
├── baseline_cnn.h5                   # Saved baseline model (~218 MB, Git LFS)
├── baseline_training_curves.png      # Baseline training curves
├── Dogs_vs_Cats_Report_final.docx    # Full project report
├── 07 AI Master.pptx                 # Presentation slides
├── 演示.mp4                           # Demo walkthrough video
├── 可视化/                            # Visualizations
│   ├── confusion_matrix.png
│   ├── detailed_error_analysis.png
│   ├── 分类报告.png
│   └── error reason.txt
└── 资料/                              # Project documents
    ├── Project Topic.docx
    ├── 分工.docx                       # Task assignment
    ├── 分工第1-4部分记录.docx           # Individual reports
    └── CDS521 Group_project term2.pdf  # Course brief
```

## Team Members & Division

| Member | Task | Deliverables |
|--------|------|-------------|
| **A** | Data preparation & augmentation | `data_loader.py`, `visualize_augmentation.py` |
| **B** | Baseline CNN design & training | `train_baseline.py`, `baseline_cnn.h5` |
| **C** | Model optimization | `optimization_experiments.py`, 5 optimized models |
| **D** | Evaluation & error analysis | `evaluate_model.ipynb`, visualizations |

## Model Architecture

3-layer CNN built with Keras Sequential API:

| Layer | Output Shape | Parameters |
|-------|-------------|-----------|
| Conv2D (32, 3x3) + MaxPool | (74, 74, 32) | 896 |
| Conv2D (64, 3x3) + MaxPool | (36, 36, 64) | 18,496 |
| Conv2D (128, 3x3) + MaxPool | (17, 17, 128) | 73,856 |
| Flatten | (36,992) | 0 |
| Dense (512) | (512) | 18,940,416 |
| Dense (1, sigmoid) | (1) | 513 |

**Total: ~19,034,179 parameters**

- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy
- Input size: 150x150, Batch size: 32, Epochs: 20
- Data augmentation: rotation, shift, shear, zoom, horizontal flip

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 89% |
| Cat Precision | 0.89 |
| Cat Recall | 0.89 |
| Dog Precision | 0.89 |
| Dog Recall | 0.89 |

## Improved Version

A transfer learning version using MobileNetV2 achieving **98.98% accuracy** is available at:  
[github.com/2499146834-arch/cats-vs-dogs](https://github.com/2499146834-arch/cats-vs-dogs)
