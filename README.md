# Crop Growth Stage Detection using Computer Vision

## Overview

This project is an AI-powered agricultural monitoring system that detects crop growth stages using advanced Computer Vision and Deep Learning techniques.

The system combines:
- YOLOv8 Object Detection
- ResNet50 Transfer Learning
- Custom CNN Classification
- OpenCV Image Processing Techniques

The project focuses heavily on:
- Image preprocessing
- Noise handling
- Low-light enhancement
- Real-world agricultural robustness
- Comparative analysis of multiple deep learning models

An interactive Streamlit web application is also developed for real-time crop image prediction and preprocessing visualization.

---

# Crop Growth Stages Detected

The system detects the following crop growth stages:

- Germination
- Vegetative
- Flowering
- Harvesting

---

# Features

# Deep Learning Models
- YOLOv8 for object detection and localization
- ResNet50 for transfer learning classification
- Custom CNN for lightweight image classification

# Image Preprocessing Pipeline
- Gaussian Filtering
- Median Filtering
- Bilateral Filtering
- CLAHE Enhancement
- Gamma Correction
- Brightness Correction
- Image Sharpening
- Gaussian Noise Simulation
- Salt & Pepper Noise Handling
- Motion Blur Simulation
- Low-Light Simulation

# Interactive Streamlit Web Application
- Upload crop images
- Add Gaussian noise
- Add Salt & Pepper noise
- Remove noise
- Simulate low-light conditions
- Brighten image
- Darken image
- Visualize preprocessing filters
- Compare model predictions
- Display evaluation graphs
- Real-time prediction comparison

---

# Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- PyTorch
- TensorFlow / Keras
- Torchvision
- Streamlit
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Kaggle
- PIL (Python Imaging Library)

---

# Project Workflow

```text
Input Crop Image
        ↓
Image Preprocessing Pipeline
        ↓
Noise Handling & Enhancement
        ↓
YOLOv8 / CNN / ResNet50 Prediction
        ↓
Prediction Comparison
        ↓
Visualization Dashboard
```

---

# Project Structure

```bash
crop-growth-stage-detection/

│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── best.pt
│   ├── crop_cnn_model.pth
│   └── resnet_model.pth
│
├── graphs/
│   ├── results.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── BoxPR_curve.png
│   ├── confusion_matrix.png
│   ├── val_batch0_pred.jpg
│   ├── val_batch0_labels.jpg
│   ├── cnn_accuracy.jpeg
│   ├── cnn_confusion_matrix.jpeg
│   ├── cnn_precision_recall.jpeg
│   ├── cnn_roc_curve.jpeg
│   ├── cnn_classification_report.jpeg
│   ├── resnet_accuracy.jpeg
│   ├── resnet_confusion_matrix.jpeg
│   ├── resnet_precision_recall.jpeg
│   ├── resnet_roc_curve.jpeg
│   └── resnet_classification.png
│
├── preprocessing/
│   └── filters.py
│
├── utils/
│   └── prediction.py
│
└── dataset/
```

---

# Dataset Information

- Dataset Format: YOLOv8 Object Detection Format
- Dataset Source: Roboflow
- Number of Classes: 4
- Annotation Type: Bounding Boxes
- Image Resolution: 640×640

The YOLO dataset was later converted into a classification dataset format for:
- Custom CNN
- ResNet50

---

# Deep Learning Models Used

# 1. YOLOv8
YOLOv8 is used for:
- Object detection
- Bounding box localization
- Real-time inference

Key Features:
- High-speed detection
- High precision
- Anchor-free architecture
- Real-time performance

---

# 2. Custom CNN
The custom CNN model is used for:
- Crop growth stage classification

Architecture Includes:
- Convolution layers
- Max pooling layers
- ReLU activation
- Dropout regularization
- Dense layers
- Softmax classifier

---

# 3. ResNet50
ResNet50 uses:
- Transfer learning
- Residual connections
- Pretrained ImageNet weights

Advantages:
- Better feature extraction
- Higher accuracy
- Deep architecture
- Better generalization

---

# Image Preprocessing Techniques

The preprocessing module improves model robustness under:
- Noisy conditions
- Low-light environments
- Blurry images
- Poor contrast conditions

Filters Used:
- Gaussian Blur
- Median Filter
- Bilateral Filter
- CLAHE
- Gamma Correction
- Sharpening Filter
- Non-Local Means Denoising

Color Space Conversions:
- RGB ↔ BGR
- BGR → HSV
- HSV → BGR
- BGR → LAB
- LAB → BGR

---

# Installation & Setup Guide

# Step 1: Clone Repository

```bash
git clone https://github.com/your-username/crop-growth-stage-detection.git
```

---

# Step 2: Open Project Folder

```bash
cd crop-growth-stage-detection
```

---

# Step 3: Create Virtual Environment

```bash
python -m venv venv
```

---

# Step 4: Activate Virtual Environment

## Windows

```bash
venv\Scripts\activate
```

## Linux / Mac

```bash
source venv/bin/activate
```

---

# Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Step 6: Add Model Files

Create a folder named:

```bash
models
```

Add the following trained models inside it:

```bash
models/
│
├── best.pt
├── crop_cnn_model.pth
└── resnet_model.pth
```

---

# Step 7: Add Graph Images

Place all evaluation graphs inside:

```bash
graphs/
```

Example:
- confusion matrix
- ROC curve
- accuracy graph
- PR curve
- classification report

---

# Step 8: Run Streamlit Application

```bash
streamlit run app.py
```

---

# Step 9: Open Browser

Streamlit automatically opens:

```bash
http://localhost:8501
```

---

# Streamlit Features

The Streamlit application includes:

- Crop image upload
- Noise simulation
- Low-light simulation
- Noise removal
- Brightness enhancement
- Preprocessing visualization
- YOLOv8 prediction
- CNN prediction
- ResNet50 prediction
- Confidence comparison
- Model comparison dashboard
- Evaluation graph visualization

---

# Model Training

# YOLOv8 Training
- Trained for 20 epochs
- Input image size: 640×640
- Object detection dataset format
- Bounding box annotations

# CNN Training
- Trained for 10 epochs
- Classification dataset
- Adam optimizer
- CrossEntropyLoss

# ResNet50 Training
- Transfer learning approach
- Pretrained ImageNet weights
- Custom classifier head
- Fine-tuned dense layers

---

# Evaluation Metrics

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- mAP50
- mAP50-95
- IoU (Intersection over Union)

---

# YOLOv8 Outputs

The YOLOv8 implementation generates:
- Bounding box predictions
- Precision curves
- Recall curves
- PR curves
- Confusion matrices
- Validation predictions
- mAP evaluation metrics

---

# Results

# YOLOv8 Results
- Precision: 97%
- Recall: 96%
- mAP50: 98%
- mAP50-95: 72%

# Custom CNN Results
- Accuracy: 72%
- Good classification performance

# ResNet50 Results
- Accuracy: 80%
- Best classification performance

---

# Challenges Faced

- YOLO dataset conversion to classification dataset
- CNN overfitting
- Low-light agricultural images
- Noise handling
- Hyperparameter tuning
- Training time management
- Model compatibility issues

---

# Solutions Implemented

- Automated dataset conversion scripts
- Dropout regularization
- Data augmentation
- Advanced preprocessing pipeline
- CLAHE enhancement
- Bilateral filtering
- Gamma correction
- Noise reduction techniques

---

# Future Scope

Future improvements include:
- Drone-based crop monitoring
- Real-time video analysis
- IoT-based smart farming
- Crop disease detection
- Mobile application deployment
- Cloud deployment
- Transformer-based vision models
- Advanced image enhancement
- Smart irrigation systems
- Automated agricultural recommendation systems

---

# Authors

- Vikas Kumar Yadav
- Mahek Mehra

---

# License

This project is developed for educational and research purposes.

---