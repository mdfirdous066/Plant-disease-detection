# From Field Images to Fast Predictions: Efficient Deep Learning for Plant Disease Detection

[![Paper](https://img.shields.io/badge/Paper-ComSNets%202026-blue)](Paper.pdf)
[![Conference](https://img.shields.io/badge/Conference-AIoT%20Session-green)]()
[![Python](https://img.shields.io/badge/Python-3.x-yellow)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

---

## Abstract

Plant diseases significantly impact crop yield and food security, making early and accurate detection essential for sustainable agriculture. This project focuses on the application of deep learning techniques for automated plant disease detection using the **PlantDoc dataset**, which consists of diverse real-world images of diseased and healthy plant leaves.

Several convolutional neural network (CNN) architectures were evaluated, including **Xception, DenseNet121, NASNetLarge, EfficientNetV2S, EfficientNetV2M, and MobileNetV2**. Among these, the **Xception model** demonstrated the strongest baseline performance. Through fine-tuning, its accuracy improved to **58.90%** on the test set, along with consistent results in terms of precision, recall, and F1-score.

To ensure scalability and real-world applicability, **quantization techniques** were applied, resulting in a significant reduction of model size (**INT8: 21.55 MB** compared to **FP32: 83 MB**), while maintaining competitive performance.

---

## Key Features

- **Multi-Model Comparison**: Systematic evaluation of 6 state-of-the-art CNN architectures
- **Transfer Learning**: Leveraging ImageNet pretrained weights for faster convergence
- **Fine-Tuning Strategy**: Two-phase training with warmup and deep fine-tuning
- **Model Quantization**: INT8 quantization for edge deployment (~75% size reduction)
- **Real-World Focus**: Evaluated on PlantDoc dataset with diverse field conditions

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Xception** | **57.20%** | **95.40%** | **62.00%** | **56.00%** |
| EfficientNetV2S | 52.97% | 88.77% | 55.71% | 52.97% |
| DenseNet121 | 51.84% | 63.24% | 53.99% | 55.51% |
| EfficientNetV2M | 51.27% | 87.05% | 53.76% | 51.27% |
| NASNetLarge | 41.53% | 66.24% | 47.50% | 41.53% |
| MobileNetV2 | 18.22% | 35.56% | 18.58% | 18.22% |

---

## Architecture

The final fine-tuned Xception model architecture:

| Layer | Output Shape | Parameters |
|-------|--------------|------------|
| Input Layer | (None, 299, 299, 3) | 0 |
| Augmentation | (None, 299, 299, 3) | 0 |
| Xception Base | (None, 10, 10, 2048) | 20,861,480 |
| Global Average Pooling | (None, 2048) | 0 |
| Dropout | (None, 2048) | 0 |
| Dense | (None, 512) | 1,049,088 |
| Batch Normalization | (None, 512) | 2,048 |
| Dropout | (None, 512) | 0 |
| Output Dense | (None, 27) | 13,851 |

**Total Parameters**: 21,926,467 (83.64 MB)  
**Trainable Parameters**: 1,063,963 (4.06 MB)

---

## Methodology

### 1. Data Preprocessing
- Image resizing to 299×299 pixels
- Pixel normalization to [0, 1] range
- Data augmentation (rotation, flipping, zoom, brightness adjustment)

### 2. Transfer Learning
- Base model initialized with ImageNet pretrained weights
- Warmup training with frozen base layers (3-5 epochs)

### 3. Fine-Tuning
- Unfreezing last 60 layers of Xception
- Reduced learning rate with scheduling
- Early stopping and ReduceLROnPlateau callbacks

### 4. Model Quantization
- FP32 to INT8 conversion
- Model size reduction: **83 MB → 21.55 MB**
- Minimal accuracy degradation (<1%)

---

## Project Structure

```
├── KD.ipynb                    # Knowledge Distillation implementation
├── Xception.ipynb              # Xception model training
├── Xception_finetune*.ipynb    # Fine-tuning experiments
├── EfficientNet.ipynb          # EfficientNet experiments
├── ResNet50V2.ipynb            # ResNet50V2 experiments
├── fypVGG16.ipynb              # VGG16 experiments
├── efficientnetv2L.ipynb       # EfficientNetV2L experiments
├── kDonpotato.ipynb            # Knowledge distillation on potato dataset
├── Paper.pdf                   # Published paper
├── architecture.png            # Model architecture diagram
└── Latex Report - Final/       # LaTeX source for project report
```

---

## Requirements

```
tensorflow>=2.x
keras
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

## Dataset

This project uses the **PlantDoc Dataset**, an object-detection and classification dataset featuring diverse images across multiple plant species and disease types.

- **Source**: [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
- **Classes**: 27 disease categories
- **Split**: 80% Training, 20% Validation/Testing

---

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Saud008/From-Multi-crop-teacher-to-Single-crop-student-via-knowledge-distillation.git
```

2. Install dependencies:
```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn
```

3. Run the Jupyter notebooks for training and evaluation.

---

## License

This project is for academic and research purposes.
