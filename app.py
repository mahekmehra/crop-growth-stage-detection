# =========================
# FILE: app.py
# =========================

import streamlit as st
import cv2
import numpy as np

from PIL import Image

from preprocessing.filters import *

from utils.prediction import *


st.set_page_config(
    page_title="Crop Growth Stage Detection",
    layout="wide"
)


# =========================
# TITLE
# =========================

st.title(
    "Crop Growth Stage Detection using Computer Vision"
)

st.markdown("""
This project detects crop growth stages using:

- YOLOv8 Object Detection
- ResNet50 Transfer Learning
- Custom CNN Classification
- Image Processing Techniques
- Noise Handling and Low-Light Enhancement
- Computer Vision Preprocessing Pipeline
""")


# =========================
# SIDEBAR
# =========================

st.sidebar.title("Image Processing Options")

add_noise_option = st.sidebar.checkbox(
    "Add Noise"
)

remove_noise_option = st.sidebar.checkbox(
    "Remove Noise"
)

darken_option = st.sidebar.checkbox(
    "Darken Image"
)

brighten_option = st.sidebar.checkbox(
    "Increase Brightness"
)


# =========================
# IMAGE UPLOAD
# =========================

uploaded = st.file_uploader(
    "Upload Crop Image",
    type=["jpg", "png", "jpeg"]
)


if uploaded:

    image = Image.open(uploaded)

    image = np.array(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )


    # =========================
    # ORIGINAL IMAGE
    # =========================

    st.header("Original Image")

    st.image(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        ),
        width=400
    )


    # =========================
    # ADD NOISE
    # =========================

    if add_noise_option:

        image = add_noise(image)

        st.subheader("Image with Gaussian Noise")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )

    if remove_noise_option:

        image = remove_gaussian_noise(image)

        st.subheader("Gaussian Noise Removed")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )


    # =========================
    # DARKEN IMAGE
    # =========================

    if darken_option:

        image = darken_image(image)

        st.subheader("Darkened Image")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )


    # =========================
    # BRIGHTEN IMAGE
    # =========================

    if brighten_option:

        image = brighten_image(image)

        st.subheader("Brightened Image")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )


    # =========================
    # PREPROCESSING SECTION
    # =========================

    st.header("Computer Vision Preprocessing Pipeline")


    col1, col2 = st.columns(2)


    with col1:

        gaussian = gaussian_filter(image)

        st.subheader("Gaussian Filter")

        st.image(
            cv2.cvtColor(
                gaussian,
                cv2.COLOR_BGR2RGB
            )
        )


        median = median_filter(image)

        st.subheader("Median Filter")

        st.image(
            cv2.cvtColor(
                median,
                cv2.COLOR_BGR2RGB
            )
        )


    with col2:

        bilateral = bilateral_filter(image)

        st.subheader("Bilateral Filter")

        st.image(
            cv2.cvtColor(
                bilateral,
                cv2.COLOR_BGR2RGB
            )
        )


        clahe_img = clahe(image)

        st.subheader("CLAHE Enhancement")

        st.image(clahe_img)


    # =========================
    # FINAL IMAGE
    # =========================

    processed = sharpen_image(image)

    st.header("Final Processed Image")

    st.image(
        cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB
        ),
        width=400
    )


    # =========================
    # MODEL PREDICTIONS
    # =========================

    st.header("Model Predictions")


    yolo_class, yolo_conf = predict_yolo(processed)

    cnn_class = predict_cnn(processed)


    col1, col2, col3 = st.columns(3)


    # =========================
    # YOLO
    # =========================

    with col1:

        st.subheader("YOLOv8")

        st.success(
            f"Prediction: {yolo_class}"
        )

        st.info(
            f"Confidence: {yolo_conf:.2f}"
        )


    # =========================
    # CNN
    # =========================

    with col2:

        st.subheader("Custom CNN")

        st.success(
            f"Prediction: {cnn_class}"
        )


    # =========================
    # RESNET PLACEHOLDER
    # =========================

    with col3:

        st.subheader("ResNet50")

        st.warning(
            "Model Evaluation Placeholder"
        )

        st.write(
            "ResNet50 comparative analysis can be added here."
        )


    # =========================
    # YOLO SECTION
    # =========================

    st.header("YOLOv8 Evaluation Results")


    # ROW 1
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Training Results")

        st.image(
            "graphs/results.png",
            width=400
        )

    with col2:

        st.subheader("Precision Curve")

        st.image(
            "graphs/BoxP_curve.png",
            width=400
        )


    # ROW 2
    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Recall Curve")

        st.image(
            "graphs/BoxR_curve.png",
            width=400
        )

    with col4:

        st.subheader("Precision Recall Curve")

        st.image(
            "graphs/BoxPR_curve.png",
            width=400
        )


    # ROW 3
    col5, col6 = st.columns(2)

    with col5:

        st.subheader("Confusion Matrix")

        st.image(
            "graphs/confusion_matrix.png",
            width=400
        )

    with col6:

        st.subheader("Ground Truth Labels")

        st.image(
            "graphs/val_batch0_labels.jpg",
            width=400
        )


    # ROW 4
    st.subheader("YOLO Predictions")

    st.image(
        "graphs/val_batch0_pred.jpg",
        width=500
    )


    # =========================
    # CNN SECTION
    # =========================

    st.header("Custom CNN Evaluation Results")


    # ROW 1
    col7, col8 = st.columns(2)

    with col7:

        st.subheader("Training vs Validation Accuracy")

        st.image(
            "graphs/cnn_accuracy.jpeg",
            width=400
        )

    with col8:

        st.subheader("CNN Confusion Matrix")

        st.image(
            "graphs/cnn_confusion_matrix.jpeg",
            width=400
        )


    # ROW 2
    col9, col10 = st.columns(2)

    with col9:

        st.subheader("Precision vs Recall")

        st.image(
            "graphs/cnn_precision_recall.jpeg",
            width=400
        )

    with col10:

        st.subheader("ROC Curve")

        st.image(
            "graphs/cnn_roc_curve.jpeg",
            width=400
        )


    # ROW 3
    col11, col12 = st.columns(2)

    with col11:

        st.subheader("CNN Sample Predictions")

        st.image(
            "graphs/sample.jpeg",
            width=400
        )

    with col12:

        st.subheader("Classification Report")

        st.image(
            "graphs/cnn_classification_report.jpeg",
            width=400
        )





    # =========================
    # RESNET PLACEHOLDER SECTION
    # =========================

    st.header("ResNet50 Placeholder Section")


    st.info("""
    ResNet50 training and evaluation graphs can be added here.

    Possible evaluation metrics:
    - Accuracy Graph
    - Loss Graph
    - Confusion Matrix
    - Precision/Recall
    - ROC Curve
    """)


    # =========================
    # FINAL COMPARISON
    # =========================

    st.header("Final Model Comparison")


    comparison_data = {

        "YOLOv8": [
            "High Detection Accuracy",
            "Fast Inference",
            "Bounding Box Localization"
        ],

        "Custom CNN": [
            "Simple Architecture",
            "Good Classification",
            "Easy to Understand"
        ],

        "ResNet50": [
            "Transfer Learning",
            "High Feature Extraction",
            "Future Improvement"
        ]
    }


    st.json(comparison_data)