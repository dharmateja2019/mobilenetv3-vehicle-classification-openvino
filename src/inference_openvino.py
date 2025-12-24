from openvino.runtime import Core
import cv2
import os

from src.preprocess import preprocess
from src.postprocess import postprocess
from src.metrics import Meter
from src.color_extractor import detect_color

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_XML = os.path.join(PROJECT_ROOT, "models", "ir", "mobilenetv3.xml")

_core = Core()
_model = _core.read_model(MODEL_XML)
_compiled_model = _core.compile_model(_model, "CPU")
_output_layer = _compiled_model.output(0)

def run_openvino(image, device="CPU", return_image=False):
    input_tensor = preprocess(image)

    meter = Meter()
    meter.tic()

    result = _compiled_model([input_tensor])[_output_layer]

    meter.toc()

    output_np = result[0]
    label, confidence = postprocess(output_np)

    img = cv2.imread(image)
    color = detect_color(img)

    output = {
        "backend": "OpenVINO FP16",
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
                    (0, 255, 255), 2)
        output["_image"] = img

    return output
