import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np

MODEL_PATH = "models/classify/mobilenetv3_2w4w.pth"
CLASS_NAMES = ["2W", "4W"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, 2
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


def predict(image):
    """
    image: str (path) OR np.ndarray (H,W,C)
    """

    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = Image.open(image).convert("RGB")

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    idx = probs.argmax().item()
    return {
        "type": CLASS_NAMES[idx],
        "confidence": round(probs[idx].item(), 3)
    }
