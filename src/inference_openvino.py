from openvino.runtime import Core
from src.preprocess import preprocess
from src.postprocess import postprocess
from src.metrics import Meter
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_XML = os.path.join(PROJECT_ROOT, "models", "ir","mobilenetv3.xml")

def run_openvino(image, device="CPU"):
    """
    OpenVINO FP16 inference
    image: PIL Image or image path
    """
    core = Core()
    model = core.read_model(MODEL_XML)
    compiled_model = core.compile_model(model, device)

    # ✅ FIX: use `image`, not `image_path`
    input_tensor = preprocess(image)

    meter = Meter()
    meter.tic()

    result = compiled_model([input_tensor])[compiled_model.output(0)]

    meter.toc()

    label, confidence = postprocess(result[0])

    return {
        "backend": "OpenVINO",
        "label": label,
        "confidence": confidence,
        "latency": meter.latency_ms(),
        "fps": meter.fps()
    }
