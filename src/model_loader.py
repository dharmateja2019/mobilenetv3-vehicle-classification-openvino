import torch
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
import os

# Always resolve paths relative to PROJECT ROOT
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "mobilenetv3.pth")

def load_model():
    # Ensure models/ directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("Downloading MobileNetV3 pretrained weights...")
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved at:", MODEL_PATH)
    else:
        print("Loading model from local file...")

    model = models.mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model
