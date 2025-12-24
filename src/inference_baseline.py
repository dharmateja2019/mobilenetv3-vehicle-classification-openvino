import torch
import cv2
from src.model_loader import load_model
from src.preprocess import preprocess
from src.postprocess import postprocess
from src.metrics import Meter
from src.color_extractor import detect_color

_model = load_model()
_model.eval()

def run_baseline(image, return_image=False):
    input_tensor = preprocess(image)
    input_tensor = torch.from_numpy(input_tensor)

    meter = Meter()
    meter.tic()

    with torch.no_grad():
        output = _model(input_tensor)

    meter.toc()

    output_np = output.detach().cpu().numpy()[0]
    label, confidence = postprocess(output_np)

    img = cv2.imread(image)
    color = detect_color(img)

    result = {
        "backend": "PyTorch FP32",
        "label": label,
        "color": color,
        "confidence": float(confidence),
        "latency": meter.latency_ms(),
        "fps": meter.fps(),
    }

    if return_image:
        text = f"{label} | {color} | {meter.latency_ms():.1f} ms | {meter.fps():.1f} FPS"
        cv2.putText(img, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        result["_image"] = img

    return result
