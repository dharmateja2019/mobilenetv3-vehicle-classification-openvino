import streamlit as st
import tempfile
import cv2
import os, sys
import time
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.inference_baseline import run_baseline
from src.inference_openvino import run_openvino
from src.detector_ssd_mobilenet import SSDMobileNetDetector
from src.color_extractor import detect_color
from src.vlm_reasoner import vlm_describe


st.set_page_config(page_title="Vehicle Inference Demo", layout="centered")
st.title("🚗 Vehicle Detection & VLM Reasoning Demo")

backend = st.selectbox("Select Backend", ["pytorch", "openvino", "compare"])
device = st.selectbox("Select Device", ["CPU", "GPU", "AUTO", "FPGA"])
use_vlm = st.checkbox("Enable VLM reasoning (slow)")
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
run_btn = st.button("▶️ Run Inference")


DETECTOR = SSDMobileNetDetector(
    "models/ssd_mobilenet/frozen_inference_graph.pb",
    "models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
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


if uploaded and run_btn:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded.read())
    image_path = tmp.name

    image = cv2.imread(image_path)
    annotated = image.copy()
    detections = DETECTOR.detect(image)

    rows = []
    backend_latency = {}
    backend_fps = {}
    vlm_text = []

    for det in detections:
        bbox = det["bbox"]
        vtype = det["label"]
        conf = det["confidence"]

        crop_img = crop(image, bbox)
        if crop_img.size == 0:
            continue

        tmp_crop = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmp_crop.name, crop_img)

        backends = ["pytorch", "openvino"] if backend == "compare" else [backend]

        for b in backends:
            start = time.time()
            if b == "pytorch":
                run_baseline(tmp_crop.name)
            else:
                run_openvino(tmp_crop.name)

            latency = (time.time() - start) * 1000
            backend_latency[b] = backend_latency.get(b, 0) + latency

            color = detect_color(refine_color_crop(crop_img))

            row = {
                "Backend": b.upper(),
                "Type": vtype,
                "Color": color,
                "Confidence": round(float(conf), 3),
                "Device": device,
            }

            if use_vlm:
                vlm = vlm_describe(tmp_crop.name)
                if vlm.get("type") != "unknown":
                    row["VLM Type"] = vlm.get("type")
                    row["VLM Color"] = vlm.get("color")
                    vlm_text.append(vlm.get("description"))

            rows.append(row)

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

    for b in backend_latency:
        backend_fps[b] = round(1000 / backend_latency[b], 2)
        backend_latency[b] = round(backend_latency[b], 2)

    st.subheader("⏱ Backend Performance")
    if backend == "compare":
        c1, c2 = st.columns(2)
        with c1:
            st.metric("PyTorch Latency (ms)", backend_latency.get("pytorch"))
            st.metric("PyTorch FPS", backend_fps.get("pytorch"))
        with c2:
            st.metric("OpenVINO Latency (ms)", backend_latency.get("openvino"))
            st.metric("OpenVINO FPS", backend_fps.get("openvino"))
    else:
        st.metric(f"{backend.upper()} Latency (ms)", backend_latency.get(backend))
        st.metric(f"{backend.upper()} FPS", backend_fps.get(backend))

    st.subheader("📊 Inference Results")
    if rows:
        st.table(pd.DataFrame(rows))
    else:
        st.warning("No vehicles detected.")

    if use_vlm and vlm_text:
        st.subheader("🧠 VLM Reasoning")
        for t in vlm_text:
            st.markdown(f"- {t}")

    st.subheader("📸 Annotated Output")
    st.image(annotated, channels="BGR")

elif uploaded and not run_btn:
    st.info("Upload an image and click **Run Inference** to start.")
