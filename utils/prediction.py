from ultralytics import YOLO
import random


classes = [

    "Flowering",

    "Germination",

    "Harvesting",

    "Vegetative"
]


# LOAD YOLO MODEL
yolo_model = YOLO(
    "models/best.pt"
)


# YOLO PREDICTION
def predict_yolo(image):

    results = yolo_model(image)

    result = results[0]

    if len(result.boxes) > 0:

        class_id = int(
            result.boxes.cls[0]
        )

        confidence = float(
            result.boxes.conf[0]
        )

        return classes[class_id], confidence

    return "No Detection", 0


# CNN PLACEHOLDER
def predict_cnn(image):

    return random.choice(classes)