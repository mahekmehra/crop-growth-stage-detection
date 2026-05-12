# Crop Growth Stage Detection using Computer Vision

## Overview

This project is an intelligent agricultural monitoring system that detects crop growth stages using:
- YOLOv8 Object Detection
- ResNet50 Transfer Learning
- Custom CNN Classification
- Image Processing Techniques

The system also includes:
- Noise handling
- Low-light enhancement
- Image preprocessing pipeline
- Streamlit web application

The project demonstrates the practical use of Computer Vision and Deep Learning in smart agriculture systems.

---

# Crop Growth Stages Detected

The system detects the following crop growth stages:

- Germination
- Vegetative
- Flowering
- Harvesting

---

# Features

## Deep Learning Models
- YOLOv8 for object detection
- ResNet50 for transfer learning
- Custom CNN for classification

## Image Preprocessing
- Gaussian Filtering
- Median Filtering
- Bilateral Filtering
- CLAHE Enhancement
- Brightness Correction
- Image Sharpening

## Interactive Streamlit Web Application
- Upload crop images
- Add Gaussian noise
- Remove noise
- Darken image
- Brighten image
- Visualize preprocessing
- View predictions
- Compare models
- Display evaluation graphs

---

# Technologies Used

- Python
- OpenCV
- YOLOv8
- PyTorch
- TensorFlow/Keras
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- Kaggle

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
│   └── best.pt
│
├── graphs/
│   ├── results.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── BoxPR_curve.png
│   ├── confusion_matrix.png
│   ├── val_batch0_pred.jpg
│   └── val_batch0_labels.jpg
│
├── preprocessing/
│   └── filters.py
│
└── utils/
    └── prediction.py
```

---

# Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/crop-growth-stage-detection.git
```

---

## Step 2: Open Project Folder

```bash
cd crop-growth-stage-detection
```

---

## Step 3: Create Virtual Environment

```bash
python -m venv venv
```

---

## Step 4: Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit Application

```bash
streamlit run app.py
```

---

# Streamlit Features

The Streamlit application includes:
- Image Upload
- Noise Simulation
- Noise Removal
- Brightness Adjustment
- Low-Light Enhancement
- Preprocessing Visualization
- YOLOv8 Predictions
- CNN Predictions
- Evaluation Graphs

---

# Evaluation Metrics

The following metrics were used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- mAP50

---

# YOLOv8 Outputs

The YOLOv8 implementation generates:
- Bounding box predictions
- Precision curves
- Recall curves
- PR curves
- Confusion matrices
- Validation predictions

---

# Future Scope

Future improvements include:
- Drone-based crop monitoring
- Real-time video analysis
- IoT-based smart farming
- Crop disease detection
- Mobile application deployment
- Cloud deployment
- Advanced preprocessing techniques
- Transformer-based vision models

---

# Author

Vikas Kumar Yadav
Mahek Mehra

---

# License

This project is developed for educational and research purposes.