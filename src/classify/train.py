import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "data/classify"
MODEL_OUT = "models/classify/mobilenetv3_2w4w.pth"

BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# TRANSFORMS
# ----------------------------
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ----------------------------
# DATASET
# ----------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Class mapping:", dataset.class_to_idx)

# ----------------------------
# MODEL
# ----------------------------
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, NUM_CLASSES
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ----------------------------
# TRAIN LOOP
# ----------------------------
model.train()

for epoch in range(EPOCHS):
    total_loss = 0

    for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models/classify", exist_ok=True)
torch.save(model.state_dict(), MODEL_OUT)

print(f"✅ Model saved to {MODEL_OUT}")
