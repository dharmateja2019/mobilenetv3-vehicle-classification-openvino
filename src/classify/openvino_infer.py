import numpy as np
from openvino.runtime import Core
from PIL import Image
from torchvision import transforms

MODEL_XML = "models/classify/openvino/mobilenetv3_2w4w.xml"
CLASS_NAMES = ["2W", "4W"]

_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

core = Core()
_compiled_models = {}

def _get_compiled_model(device):
    if device not in _compiled_models:
        _compiled_models[device] = core.compile_model(MODEL_XML, device)
    return _compiled_models[device]

def classify_openvino(image, device="CPU"):
    compiled_model = _get_compiled_model(device)
    output_layer = compiled_model.output(0)

    if isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = Image.open(image).convert("RGB")

    input_tensor = (
        _preprocess(img)
        .unsqueeze(0)
        .numpy()
        .astype(np.float32)
    )

    logits = compiled_model([input_tensor])[output_layer][0]

    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    idx = int(np.argmax(probs))
    return {
        "type": CLASS_NAMES[idx],
        "confidence": round(float(probs[idx]), 3),
        "device": device
    }
