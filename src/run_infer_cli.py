import argparse
import cv2
import os
import sys
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.inference_baseline import run_baseline
from src.inference_openvino import run_openvino
from src.detector_ssd_mobilenet import SSDMobileNetDetector
from src.color_extractor import detect_color
from src.vlm_reasoner import vlm_describe


DETECTOR = SSDMobileNetDetector(
    "models/ssd_mobilenet/frozen_inference_graph.pb",
    "models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
)


def crop(image, bbox):
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def refine_color_crop(crop_img, margin=0.2):
    h, w = crop_img.shape[:2]
    dx = int(w * margin)
    dy = int(h * margin)
    refined = crop_img[dy:h - dy, dx:w - dx]
    return refined if refined.size > 0 else crop_img


def run_backend(image_path, backend, use_vlm=False):
    image = cv2.imread(image_path)
    annotated = image.copy()
    detections = DETECTOR.detect(image)

    vehicles = []
    start = time.time()

    for det in detections:
        bbox = det["bbox"]
        vtype = det["label"]
        conf = det["confidence"]

        crop_img = crop(image, bbox)
        if crop_img.size == 0:
            continue

        tmp_crop = "tmp_crop.jpg"
        cv2.imwrite(tmp_crop, crop_img)

        if backend == "pytorch":
            run_baseline(tmp_crop)
        else:
            run_openvino(tmp_crop)

        color = detect_color(refine_color_crop(crop_img))

        vehicle = {
            "backend": backend,
            "type": vtype,
            "color": color,
            "confidence": round(float(conf), 3),
        }

        if use_vlm:
            vlm = vlm_describe(tmp_crop)
            if vlm.get("type") != "unknown":
                vehicle["vlm_reasoning"] = {
                    "type": vlm.get("type"),
                    "color": vlm.get("color"),
                    "description": vlm.get("description"),
                }

        vehicles.append(vehicle)

        x1, y1, x2, y2 = bbox
        overlay = f"{vtype.upper()} | {color.upper()}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            overlay,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    latency_ms = (time.time() - start) * 1000
    fps = 1000 / latency_ms if latency_ms > 0 else 0

    output = {
        "backend": backend,
        "count": len(vehicles),
        "latency_ms": round(latency_ms, 2),
        "fps": round(fps, 2),
        "vehicles": vehicles,
    }

    if use_vlm and len(vehicles) == 0:
        fallback = vlm_describe(image_path)
        if fallback.get("type") != "unknown":
            output["vlm_fallback"] = fallback

    return output, annotated


def main():
    parser = argparse.ArgumentParser("Vehicle Detection CLI")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--backend",
        choices=["pytorch", "openvino", "compare"],
        default="openvino",
    )
    parser.add_argument("--use-vlm", action="store_true")

    args = parser.parse_args()
    os.makedirs("outputs", exist_ok=True)

    if args.backend == "compare":
        pt_result, _ = run_backend(args.image, "pytorch", args.use_vlm)
        ov_result, ov_img = run_backend(args.image, "openvino", args.use_vlm)

        print(json.dumps({
            "compare": {
                "pytorch": pt_result,
                "openvino": ov_result
            }
        }, indent=2))

        cv2.imwrite("outputs/output_compare.jpg", ov_img)

    else:
        result, annotated = run_backend(args.image, args.backend, args.use_vlm)
        print(json.dumps(result, indent=2))
        cv2.imwrite(f"outputs/output_{args.backend}.jpg", annotated)


if __name__ == "__main__":
    main()
