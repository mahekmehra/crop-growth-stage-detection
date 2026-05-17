# =========================
# FILE: app.py
# =========================

import streamlit as st
import cv2
import numpy as np

from PIL import Image

from preprocessing.filters import *

from utils.prediction import *


# =========================
# PAGE CONFIG
# =========================

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
- Advanced Image Processing
- Noise Handling
- Low-Light Enhancement
- Computer Vision Preprocessing Pipeline
""")


# =========================
# SIDEBAR
# =========================

st.sidebar.title("Image Processing Options")

add_noise_option = st.sidebar.checkbox(
    "Add Gaussian Noise"
)

remove_noise_option = st.sidebar.checkbox(
    "Remove Gaussian Noise"
)

darken_option = st.sidebar.checkbox(
    "Darken Image"
)

brighten_option = st.sidebar.checkbox(
    "Increase Brightness"
)

salt_pepper_option = st.sidebar.checkbox(
    "Add Salt & Pepper Noise"
)

motion_blur_option = st.sidebar.checkbox(
    "Add Motion Blur"
)

low_light_option = st.sidebar.checkbox(
    "Simulate Low Light"
)

advanced_denoise_option = st.sidebar.checkbox(
    "Advanced Denoising"
)

gamma_option = st.sidebar.checkbox(
    "Gamma Correction"
)


# =========================
# IMAGE UPLOAD
# =========================

uploaded = st.file_uploader(
    "Upload Crop Image",
    type=["jpg", "png", "jpeg"]
)


# =========================
# MAIN LOGIC
# =========================

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
    # ADD GAUSSIAN NOISE
    # =========================

    if add_noise_option:

        image = add_noise(image)

        st.subheader("Gaussian Noise Added")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )

    # =========================
    # REMOVE NOISE
    # =========================

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
    # SALT & PEPPER NOISE
    # =========================

    if salt_pepper_option:

        image = add_salt_pepper_noise(image)

        st.subheader("Salt & Pepper Noise")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )

    # =========================
    # MOTION BLUR
    # =========================

    if motion_blur_option:

        image = add_motion_blur(image)

        st.subheader("Motion Blur Applied")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )

    # =========================
    # LOW LIGHT
    # =========================

    if low_light_option:

        image = low_light_simulation(image)

        st.subheader("Low Light Simulation")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )

    # =========================
    # ADVANCED DENOISING
    # =========================

    if advanced_denoise_option:

        image = advanced_denoising(image)

        st.subheader("Advanced Denoising")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )

    # =========================
    # GAMMA CORRECTION
    # =========================

    if gamma_option:

        image = gamma_correction(image)

        st.subheader("Gamma Corrected Image")

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            width=400
        )

    # =========================
    # PREPROCESSING PIPELINE
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

        sharpened = sharpen_image(image)

        st.subheader("Sharpened Image")

        st.image(
            cv2.cvtColor(
                sharpened,
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

        st.image(
            cv2.cvtColor(
                clahe_img,
                cv2.COLOR_BGR2RGB
            )
        )

        denoise = advanced_denoising(image)

        st.subheader("Advanced Denoising")

        st.image(
            cv2.cvtColor(
                denoise,
                cv2.COLOR_BGR2RGB
            )
        )

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
        width=450
    )

    # =========================
    # MODEL PREDICTIONS
    # =========================

    st.header("Model Predictions")

    yolo_class, yolo_conf = predict_yolo(processed)

    cnn_class, cnn_conf = predict_cnn(processed)

    resnet_class, resnet_conf = predict_resnet(processed)

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

        st.info(
            f"Confidence: {cnn_conf:.2f}"
        )

    # =========================
    # RESNET
    # =========================

    with col3:

        st.subheader("ResNet50")

        st.success(
            f"Prediction: {resnet_class}"
        )

        st.info(
            f"Confidence: {resnet_conf:.2f}"
        )

    # =========================
    # LIVE MODEL COMPARISON
    # =========================

    st.header("Live Model Confidence Comparison")

    comparison_chart = {

        "YOLOv8": float(yolo_conf),

        "Custom CNN": float(cnn_conf),

        "ResNet50": float(resnet_conf)
    }

    st.subheader("Confidence Score Comparison")

    st.bar_chart(comparison_chart)

    st.subheader("Prediction Summary")

    prediction_table = {

        "Model": [
            "YOLOv8",
            "Custom CNN",
            "ResNet50"
        ],

        "Prediction": [
            yolo_class,
            cnn_class,
            resnet_class
        ],

        "Confidence": [
            round(yolo_conf, 2),
            round(cnn_conf, 2),
            round(resnet_conf, 2)
        ]
    }

    st.table(prediction_table)

    best_model = max(
        comparison_chart,
        key=comparison_chart.get
    )

    st.success(
        f"Best Confidence Prediction: {best_model}"
    )

    st.subheader("Model Confidence Distribution")

    st.area_chart(comparison_chart)

    st.header("Model Training Results Comparison")
    col18,col19,col20 = st.columns(3)
    with col18:
        st.subheader("YOLOv8 Results")
        st.image("graphs/yolo_results.png",width=400)
    with col19:
        st.subheader("Custom CNN Results")
        st.image("graphs/customcnn_results.png",width=400)
    with col20:
        st.subheader("ResNet50 Results")
        st.image("graphs/resnet_results.png",width=400)


    # ========================= # YOLO SECTION # ========================= 
    st.header("YOLOv8 Evaluation Results") 
    col1, col2 = st.columns(2) 
    with col1: 
        st.subheader("Training Results") 
        st.image( "graphs/results.png", width=400 ) 
    with col2: 
        st.subheader("Precision Curve") 
        st.image( "graphs/BoxP_curve.png", width=400 ) # 
    col3, col4 = st.columns(2) 
    with col3: 
        st.subheader("Recall Curve") 
        st.image( "graphs/BoxR_curve.png", width=400 ) 
    with col4: 
        st.subheader("Precision Recall Curve") 
        st.image( "graphs/BoxPR_curve.png", width=400 )  
    col5, col6 = st.columns(2) 
    with col5: 
        st.subheader("Ground Truth Labels") 
        st.image( "graphs/val_batch0_labels.jpg", width=400 ) 
    with col6:
        st.subheader("YOLO Predictions")
        st.image( "graphs/val_batch0_pred.jpg", width=400 ) 
        
    # ========================= # CNN SECTION # ========================= 
    st.header("Custom CNN Evaluation Results")  
    col7, col8 = st.columns(2) 
    with col7: 
        st.subheader("Training vs Validation Accuracy") 
        st.image( "graphs/cnn_accuracy.jpeg", width=400 ) 
    with col8: 
        st.subheader("CNN Confusion Matrix") 
        st.image( "graphs/customcnn_confusion.png", width=400 ) 
    col9, col10 = st.columns(2) 
    with col9: 
        st.subheader("Precision vs Recall") 
        st.image( "graphs/cnn_pr.png", width=400 ) 
    with col10: 
        st.subheader("ROC Curve") 
        st.image( "graphs/cnn_roc.png", width=400 ) 
    col11, col12 = st.columns(2) 
    with col11: 
        st.subheader("CNN Sample Predictions") 
        st.image( "graphs/sample.jpeg", width=400 ) 
    with col12: 
        st.subheader("Classification Report") 
        st.image( "graphs/cnn_classification.png", width=400 ) 
        
    # ========================= # RESNET PLACEHOLDER SECTION # ========================= 
    st.header("ResNet50 Evaluation Results") 
    col13, col14 = st.columns(2) 
    with col13: 
        st.subheader("Training vs Validation Accuracy") 
        st.image( "graphs/resnet_accuracy.jpeg", width=400 ) 
    with col14: 
        st.subheader("Resnet Confusion Matrix") 
        st.image( "graphs/resnet_confusion.png", width=400 ) 
    col15, col16, col17 = st.columns(3) 
    with col15: 
        st.subheader("Precision vs Recall") 
        st.image( "graphs/resnet_pr.png", width=400 ) 
    with col16: 
        st.subheader("ROC Curve") 
        st.image( "graphs/resnet_roc.png", width=400 ) 
    with col17: 
        st.subheader("Resnet Classification Report") 
        st.image( "graphs/resnet_classification.png", width=400 )  

    st.info("""
    Observation:

    - YOLOv8 performs strong object localization.
    - ResNet50 provides advanced feature extraction.
    - Custom CNN provides lightweight classification.
    - Image preprocessing improves prediction quality.
    - CLAHE and bilateral filtering improve low-light robustness.
    - Denoising techniques improve noisy image prediction.
    """)



    # =========================
    # FINAL MODEL COMPARISON
    # =========================

    

    col21,col22 = st.columns(2)

    with col21:

        st.header("Final Model Comparison")

        comparison_data = {

            "YOLOv8": [
                "Fast Detection",
                "Bounding Box Localization",
                "High Real-Time Accuracy"
            ],

            "Custom CNN": [
                "Simple Architecture",
                "Easy Implementation",
                "Lightweight Model"
            ],

            "ResNet50": [
                "Transfer Learning",
                "Best Feature Extraction",
                "Highest Accuracy"
            ]
        }

        st.json(comparison_data)

    with col22:
        st.header("Comparison Graph")
        st.image("graphs/comparison.png",width= 600)
