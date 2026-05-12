# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor()
# ])

# train_dataset = datasets.ImageFolder(
#     root="classification_dataset/train",
#     transform=transform
# )

# valid_dataset = datasets.ImageFolder(
#     root="classification_dataset/valid",
#     transform=transform
# )

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# class CropCNN(nn.Module):
#     def __init__(self):
#         super(CropCNN, self).__init__()

#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.fc1 = nn.Linear(128 * 28 * 28, 256)
#         self.fc2 = nn.Linear(256, 4)

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))

#         x = x.view(x.size(0), -1)

#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)

#         return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = CropCNN().to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# epochs = 10

# for epoch in range(epochs):
#     model.train()

#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     train_accuracy = 100 * correct / total

#     model.eval()

#     valid_correct = 0
#     valid_total = 0

#     with torch.no_grad():
#         for images, labels in valid_loader:
#             images = images.to(device)
#             labels = labels.to(device)

#             outputs = model(images)

#             _, predicted = torch.max(outputs, 1)
#             valid_total += labels.size(0)
#             valid_correct += (predicted == labels).sum().item()

#     valid_accuracy = 100 * valid_correct / valid_total

#     print(f"Epoch [{epoch+1}/{epochs}]")
#     print(f"Training Loss: {running_loss:.4f}")
#     print(f"Training Accuracy: {train_accuracy:.2f}%")
#     print(f"Validation Accuracy: {valid_accuracy:.2f}%")

# torch.save(model.state_dict(), "crop_cnn_model.pth")

# print("Model saved as crop_cnn_model.pth")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# Image Transformations
# =========================

transform = transforms.Compose([

    # Resize image
    transforms.Resize((224, 224)),

    # Random flip
    transforms.RandomHorizontalFlip(),

    # Random rotation
    transforms.RandomRotation(10),

    # Handle dark / bright images
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.3
    ),

    # Handle blurry / noisy images
    transforms.GaussianBlur(kernel_size=3),

    # Convert image to tensor
    transforms.ToTensor()
])

# =========================
# Dataset Loading
# =========================

train_dataset = datasets.ImageFolder(
    root="classification_dataset/train",
    transform=transform
)

valid_dataset = datasets.ImageFolder(
    root="classification_dataset/valid",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False
)

# =========================
# Custom CNN Model
# =========================

class CropCNN(nn.Module):

    def __init__(self):
        super(CropCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32, 64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            64, 128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(
            128 * 28 * 28,
            256
        )

        self.fc2 = nn.Linear(
            256,
            4
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(0.5)

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
# Device Configuration
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = CropCNN().to(device)

# =========================
# Loss & Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# Training
# =========================

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = 100 * correct / total

    # =========================
    # Validation
    # =========================

    model.eval()

    valid_correct = 0
    valid_total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            valid_total += labels.size(0)

            valid_correct += (
                predicted == labels
            ).sum().item()

    valid_accuracy = (
        100 * valid_correct / valid_total
    )

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Training Loss: {running_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Accuracy: {valid_accuracy:.2f}%")

# =========================
# Save Model
# =========================

torch.save(
    model.state_dict(),
    "models\crop_cnn_model.pth"
)

print("Model saved as crop_cnn_model.pth")