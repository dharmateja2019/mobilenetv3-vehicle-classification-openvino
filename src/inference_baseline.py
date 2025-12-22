import torch
from src.model_loader import load_model
from src.preprocess import preprocess
from src.postprocess import postprocess
from src.metrics import Meter


def run_baseline(image):
    """
    PyTorch FP32 inference
    """
    model = load_model()
    input_tensor = preprocess(image)

    meter = Meter()
    meter.tic()

    with torch.no_grad():
        output = model(torch.tensor(input_tensor))

    meter.toc()

    label, confidence = postprocess(output.numpy()[0])

    return {
        "backend": "PyTorch FP32",
        "label": label,
        "confidence": confidence,
        "latency": meter.latency_ms(),
        "fps": meter.fps()
    }
