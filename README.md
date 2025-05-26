# Image Classification Under Resource Constraints: A Comparative Study 🖼️🔬

[![Python Version](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?&style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![OpenCV](https://img.shields.io/badge/OpenCV-grey?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-blue?style=flat&logo=xgboost&logoColor=white)](https://xgboost.ai/)

## 🌟 Overview

This project provides a comparative analysis of various image classification techniques, ranging from traditional feature-based methods to modern deep learning architectures. The core focus is on evaluating these approaches under **realistic hardware constraints**, mimicking a typical local workstation setup rather than high-end server infrastructure.

We investigate the performance, training efficiency, and operational limitations of these methods on a strategically curated subset of the **Places365 dataset**. The goal is to offer practical insights for practitioners and researchers working with limited computational resources.

## 🎯 Key Objectives

*   Compare traditional methods (BoVW with SIFT/ORB, SPM, Spectral Features) against deep learning models (AlexNet, Vision Transformer).
*   Evaluate performance on a 4-category scene classification task derived from Places365, designed to test semantic distinction.
*   Analyze trade-offs in accuracy, resource consumption (RAM, VRAM), and training time within a constrained environment.
*   Provide guidance for selecting appropriate image classification tools when hardware resources are limited.

## 💾 Dataset: Places365 Subset

The original Places365 dataset contains 1.8 million images across 365 categories. To make experimentation feasible under hardware constraints, we:

1.  **Limited the total dataset size** to 100,000 images.
2.  **Mapped the 365 original categories** into 4 broad, semantically challenging categories:
    *   Indoor Residential
    *   Indoor Public/Commercial
    *   Outdoor Natural
    *   Outdoor Urban
3.  **Split the data** into 80% training (80,000 images) and 20% testing (20,000 images).
4.  **Balanced the splits** with 25,000 images per broad category (20k train, 5k test per category).
5.  Selected images **randomly** to avoid bias.

This curated dataset allows for manageable training and evaluation while still posing a significant challenge for classification algorithms due to subtle contextual differences between categories.

## 🛠️ Methodologies Explored

We implemented and evaluated the following approaches:

### 1. Traditional Feature-Based Approaches

*   **Bag of Visual Words (BoVW) & Spatial Pyramid Matching (SPM)**:
    *   **Feature Extraction**: SIFT and ORB descriptors (using OpenCV).
    *   **Visual Vocabulary**: K-Means clustering (K=1000) on extracted features (scikit-learn).
    *   **Histogram Representation**: L1 normalized histograms of visual words per image.
    *   **SPM**: 2 levels (base + 4 quadrants), weighted and concatenated.
    *   **Classifier**: XGBoost.
*   **Spectral Feature Analysis**:
    *   **Feature Extraction**: Resized (128x128) grayscale images, FFT applied, central (32x32) magnitude spectrum region extracted and flattened (1024-D vector).
    *   **Normalization**: StandardScaler.
    *   **Classifier**: XGBoost (identical parameters to BoVW models).

### 2. Deep Learning Approaches

*   **AlexNet Fine-Tuning**:
    *   **Base Model**: Pretrained AlexNet from `torchvision.models` (weights from Places365).
    *   **Architectural Change**: Replaced final classification layer for 4-category output.
    *   **Training Strategy**: Froze convolutional layers, trained only the new head.
    *   **Training Details**: CrossEntropyLoss, Adam optimizer (LR 0.0001), StepLR scheduler, 25 epochs, batch size 32.
*   **AlexNet Feature Extraction**:
    *   **Base Model**: Same pretrained AlexNet (Places365).
    *   **Feature Extraction**: Output from the second-to-last fully-connected layer (4096-D vector per image).
    *   **Classifier**: XGBoost (identical parameters to traditional models).
*   **Vision Transformer (ViT) Fine-Tuning**:
    *   **Base Model**: ViT base (16x16 patch) from `torchvision.models` (weights from ImageNet-1k, *not* Places365, for a more challenging scenario).
    *   **Architectural Change**: Replaced model head for 4-category output.
    *   **Training Strategy**: Froze transformer blocks, trained only the new head.
    *   **Training Details**: Same as AlexNet Fine-Tuning.
    *   **Preprocessing**: ImageNet ViT standards (resize smallest edge to 256, center 224x224 crop, ImageNet normalization).

### Data Handling
*   For deep learning models, a custom HDF5 dataset was used with PyTorch for efficient loading from disk, managing RAM limitations.
*   For AlexNet feature extraction, features were batched and saved to HDF5 files.

## 📊 Key Findings & Results

Detailed performance metrics, including accuracy, F1-scores, and confusion matrices, can be found in the **`/Results`** folder of this repository.

**Summary of Overall Performance:**

| Model Type                  | Feature Extractor       | Classifier | Overall Accuracy (%) | Macro F1-Score (%) |
| :-------------------------- | :---------------------- | :--------- | :------------------- | :----------------- |
| BoVW                        | SIFT                    | XGBoost    | 59.65                | 59.16              |
| BoVW+SPM                    | SIFT                    | XGBoost    | 60.29                | 59.77              |
| BoVW                        | ORB                     | XGBoost    | 43.04                | 42.95              |
| BoVW+SPM                    | ORB                     | XGBoost    | 43.68                | 43.21              |
| Spectral                    | FFT                     | XGBoost    | 50.86                | 50.72              |
| **AlexNet (Finetuned)**     | --                      | CNN Head   | **85.03**            | **84.97**          |
| AlexNet (Feature Extracted) | AlexNet FC (4096-dim)   | XGBoost    | 80.85                | 80.72              |
| ViT (Finetuned)             | --                      | ViT Head   | 80.32                | 80.21              |

**Key Takeaways:**

*   **Deep learning approaches significantly outperformed traditional methods**, with the fine-tuned AlexNet (pretrained on Places365) achieving the highest accuracy.
*   AlexNet used as a feature extractor also performed very well, demonstrating the power of its learned representations.
*   The Vision Transformer (ViT), despite being pretrained on ImageNet-1k (a different domain) and only having its head fine-tuned, showed strong performance, highlighting its generalization capabilities.
*   Traditional BoVW methods with SIFT descriptors were notably better than ORB and spectral features, but still lagged considerably behind deep learning models.
*   SPM provided marginal improvements for BoVW, likely limited by the 2-level pyramid restriction due to RAM constraints.
*   Resource limitations (RAM, VRAM) necessitated strategies like smaller batch sizes, dataset subsetting, HDF5 for data loading, and freezing layers during fine-tuning.

For a detailed discussion, including specific category confusions and model behaviors, please refer to the full experimental results and discussion sections (if available as a paper/report) and the contents of the `/Results` folder.

## 💻 Computational Environment

All experiments were conducted on a local workstation with the following specifications:

*   **CPU**: Intel Core i7-12700K
*   **GPU**: NVIDIA GeForce RTX 4070 (12GB VRAM)
*   **RAM**: 32GB DDR4
*   **OS**: Windows 11
*   **Primary Software**:
    *   Python 3.11.9
    *   OpenCV (cv2)
    *   PyTorch
    *   TensorFlow (for TFDS dataset loading, though primary DL framework is PyTorch)
    *   scikit-learn
    *   XGBoost

These specifications highlight the "constrained environment" aspect of this study.

## 🚀 Getting Started (Conceptual)

This repository primarily serves as a report and summary of the comparative study. The code for implementing these models would involve:

1.  **Dataset Preparation Scripts**: For downloading/subsetting Places365 and creating the HDF5 files.
2.  **Traditional Methods Implementation**: Scripts using OpenCV for feature extraction, scikit-learn for K-Means, and XGBoost for classification.
3.  **Deep Learning Scripts**: PyTorch scripts for:
    *   Defining custom `Dataset` and `DataLoader` for HDF5.
    *   Loading pretrained AlexNet and ViT models.
    *   Modifying model architectures (replacing heads, freezing layers).
    *   Implementing training loops (optimizer, loss function, scheduler).
    *   Feature extraction pipelines for AlexNet-XGBoost.
4.  **Evaluation Scripts**: To calculate metrics and generate confusion matrices.

*(If you plan to add the actual code, you can replace this section with setup instructions, dependencies installation (`requirements.txt`), and how to run the experiments.)*

## 🤝 Contributing

While this project documents a specific study, suggestions for future comparisons, alternative resource-constrained strategies, or improvements to the analysis are welcome via Issues or Pull Requests.

## 📄 License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.

## 🙏 Acknowledgements

*   The **Places365 dataset** team for providing the foundational dataset.
*   The developers of **PyTorch, TensorFlow, scikit-learn, OpenCV, XGBoost**, and other open-source libraries that made this research possible.

---

This README aims to provide a comprehensive yet accessible overview of the project. For deeper insights, please explore the `/Results` folder and any accompanying reports or papers.
