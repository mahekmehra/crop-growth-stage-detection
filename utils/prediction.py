from ultralytics import YOLO

import torch
import torch.nn as nn

from torchvision import models
from torchvision import transforms

import numpy as np
import cv2


# =========================
# CLASS NAMES
# =========================

classes = [

    "Flowering",

    "Germination",

    "Harvesting",

    "Vegetative"
]


# =========================
# DEVICE
# =========================

device = torch.device(

    "cuda" if torch.cuda.is_available()

    else "cpu"
)


# =========================
# LOAD YOLO MODEL
# =========================

yolo_model = YOLO(
    "models/best.pt"
)


# =========================
# CNN MODEL
# =========================

class CropCNN(nn.Module):

    def __init__(self):

        super(CropCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(
            128 * 28 * 28,
            256
        )

        self.fc2 = nn.Linear(
            256,
            4
        )

    def forward(self, x):

        x = self.pool(
            self.relu(self.conv1(x))
        )

        x = self.pool(
            self.relu(self.conv2(x))
        )

        x = self.pool(
            self.relu(self.conv3(x))
        )

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# =========================
# LOAD CNN MODEL
# =========================

cnn_model = CropCNN().to(device)

cnn_model.load_state_dict(

    torch.load(
        "models/crop_cnn_model.pth",
        map_location=device
    )
)

cnn_model.eval()


# =========================
# LOAD RESNET MODEL
# =========================

resnet_model = models.resnet50(
    pretrained=False
)

num_features = resnet_model.fc.in_features

resnet_model.fc = nn.Sequential(

    nn.Linear(
        num_features,
        256
    ),

    nn.ReLU(),

    nn.Dropout(0.5),

    nn.Linear(
        256,
        4
    )
)

resnet_model.load_state_dict(

    torch.load(
        "models/resnet_model.pth",
        map_location=device
    )
)

resnet_model = resnet_model.to(device)

resnet_model.eval()


# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize((224,224)),

    transforms.ToTensor()
])


# =========================
# YOLO PREDICTION
# =========================

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


# =========================
# CNN PREDICTION
# =========================

def predict_cnn(image):

    img = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    img = transform(img)

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = cnn_model(img)

        _, predicted = torch.max(
            outputs,
            1
        )

        confidence = torch.softmax(
            outputs,
            dim=1
        )[0][predicted].item()

    return classes[predicted.item()], confidence


# =========================
# RESNET PREDICTION
# =========================

def predict_resnet(image):

    img = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    img = transform(img)

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = resnet_model(img)

        _, predicted = torch.max(
            outputs,
            1
        )

        confidence = torch.softmax(
            outputs,
            dim=1
        )[0][predicted].item()

    return classes[predicted.item()], confidence