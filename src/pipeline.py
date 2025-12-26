import cv2
import os
import tempfile
import time
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.detect import VehicleDetector
from src.classify.openvino_infer import classify_openvino
from test.test_classify import predict as classify_pytorch
from src.color.color_extraction import detect_color
from src.vlm.vlm_qa import ask_question

detector = VehicleDetector()


def _run_single(
    image_path,
    backend="openvino",
    device="CPU",
    output_path=None,
    qa_question=None,
):
    image = cv2.imread(image_path)
    annotated = image.copy()

    # ---------------- CV PIPELINE TIMING ----------------
    start = time.time()
    detections = detector.detect(image)

    vehicles = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        color = detect_color(crop)

        if backend == "pytorch":
            cls = classify_pytorch(crop)
        else:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                cv2.imwrite(f.name, crop)
                cls = classify_openvino(f.name, device=device)
                os.remove(f.name)

        vehicle_type = cls["type"]

        vehicles.append({
            "type": vehicle_type,
            "color": color
        })

        label = f"{vehicle_type} | {color}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    latency_ms = (time.time() - start) * 1000
    fps = 1000 / latency_ms if latency_ms > 0 else 0
    # ---------------------------------------------------

    # ---------------- VLM (POST-PROCESSING ONLY) ----------------
    qa_answer = None
    if qa_question:
        vision_result = {
            "vehicles": vehicles
        }
        qa_answer = ask_question(vision_result, qa_question)
    # -----------------------------------------------------------

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)

    return {
        "backend": backend,
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 2),
        "vehicles": vehicles,
        "qa_answer": qa_answer,
        "annotated": annotated,
    }


def run_pipeline(
    image_path,
    backend="openvino",
    device="CPU",
    qa_question=None,
):
    if backend == "compare":
        return {
            "compare": {
                "pytorch": _run_single(
                    image_path,
                    "pytorch",
                    device,
                    "outputs/result_pytorch.jpg",
                    qa_question,
                ),
                "openvino": _run_single(
                    image_path,
                    "openvino",
                    device,
                    "outputs/result_openvino.jpg",
                    qa_question,
                ),
            }
        }

    return _run_single(
        image_path,
        backend,
        device,
        "outputs/result.jpg",
        qa_question,
    )
